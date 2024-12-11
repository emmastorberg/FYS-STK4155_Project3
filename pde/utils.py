from typing import Optional

import numpy as np
import torch.nn as nn
import torch

from pde.grid_search import config
from pde.neural_network import NN
from pde.generate_data import load_PINN_data, load_numsolver_data


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Initial condition functions

    Args:
        x (np.ndarray): rod coordinates, i.e. x-axis.

    Returns:
        np.ndarray: x-axis with initial condition applied, i.e. u(x, 0).
    """
    return np.sin(np.pi * x)


def find_dt_for_stability(dx: float) -> float:
    """Determines suitable dt value for stability from a given dx value.

    Args:
        dx (float): step length in x-direction

    Raises:
        ValueError: If dx is 0 or negative

    Returns:
        float: Suitable dt
    """
    if dx <= 0:
        raise ValueError("dx must be positive")

    dt = 0.5 * dx**2
    return dt


def analytical_solution(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Analytical solution of heat diffusion equation.

    Args:
        x (np.ndarray): x-axis (location in space)
        t (np.ndarray): time axis

    Returns:
        np.ndarray: 2D matrix containing u(x,t) as a grid for all combinations of x and t.
    """
    xx, tt = np.meshgrid(x, t, sparse=True)
    return np.sin(np.pi * xx) * np.exp(-np.pi**2 * tt)


def dict_to_matrix(data_dict: dict) -> np.ndarray:
    """Converts a dictionary of plotting data to matrix. The dictionary must have save_step = 1.

    Args:
        data_dict (dict): Dictionary of plotting data with save_step = 1.

    Raises:
        ValueError: If data_dict does not contain values for every integer, i.e. if save_step is something other than 1.

    Returns:
        np.ndarray: data organized in a 2D matrix with indices of vectors corresponding to what were previously dictionary keys.
    """
    # Only possible to use if save_step = 1
    try:
        value = data_dict.get(1)
    except:
        raise ValueError("data_dict parameter must be created with save_step = 1.")

    matrix = np.array(list(data_dict.values()))
    return matrix


def matrix_to_dict(data_matrix: np.ndarray, save_step: int = 1) -> dict:
    """Converts a matrix to a dictionary of plotting values.

    Args:
        data_matrix (np.ndarray): The 2D matrix of data.
        save_step (int, optional): The timestep interval for which to save solution vectors for plotting later. Defaults to 1.

    Returns:
        dict: Dictionary of plotting data with the specified save_step.
    """
    data_dict = {}

    for i in range(0, len(data_matrix), save_step):
        data_dict[i] = data_matrix[i]
    return data_dict

def get_model_filename(
        num_hidden: int,
        hidden_dim: int,
        activation: str,
        iteration: Optional[int] = None,
    ) -> str:
    """
    Get filename of saved model with specified parameters.

    Note: Models trained on a standard feed-forward neural network (FFNN) follow
    the same filename pattern with a leading string of FFNN_. This is not included here.

    Note: This function does not guarantee that the file exists.

    Args:
        num_hidden (int): Number of hidden layers. Defaults to 3.
        hidden_dim (int): Number of neurons per hidden layer. Defaults to 50.
        activation (Type[nn.Module]): The activation function used for all hidden layers.
        Defaults to torch.nn.Tanh.
        iteration (Optional, int): The current iteration if several models are trained with
        these parameters. Defaults to None.

    Returns:
        str: The filename of saved model with specified parameters.
    """
    activation_names = {
        nn.Tanh: "tanh",
        nn.ReLU: "relu",
        nn.LeakyReLU: "leaky_relu",
    }
    if iteration is None:
        iter = ""
    else:
        iter = f"_iter-{iteration + 1}"
    activation_name = activation_names[activation]
    filename = f"pde/models/nhidden-{num_hidden}_dim-{hidden_dim}_activation-{activation_name}{iter}.pt"

    return filename


def make_data_plottable(
        x: torch.Tensor,
        t: torch.Tensor,
        output: torch.Tensor,
        dx: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert torch.Tensors of a flattened grid of spatial and temporal data points, as well
    as a non-flattened torch.Tensor grid of the corresponding outputs to
    np.ndarrays suitable for plotting.

    Note: This funciton assumes the temporal data points to go from 0 to 0.3.

    Args:
        x (torch.Tensor): Spatial data points. Expected shape (N, 1).
        t (torch.Tensor): Temporal data points. Expected shape (N, 1).
        output (torch.Tensor): The output of the neural network, representing the 
        solution to the diffusion equation at the given spatial and temporal points.
        dx (float): The lenght between each spatial data point.
        dt (float): The lenght between each temporal data point.

    Returns:
        tuple (np.ndarray, np.ndarray, np.ndarray):
            A tuple of np.ndarrays with spatial, temporal and the corresponding 
            outputs respectively.
    """
    L = 1
    t_max = 0.3
    Nx = int(L / dx)
    Nt = int(t_max / dt)

    output = output.reshape(Nx + 1, Nt + 1)
    output = output.detach().numpy()

    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, t_max, Nt + 1)

    return x, t, output.T


class Grid:
    """
    Class for handeling saved models from a grid search, and printing the
    results in a LaTeX format suitable for a table.
    """
    def __init__(
            self,
            dx: float,
            dt: float,
            iterations: Optional[int] = 3,
        ) -> None:
        """
        Initialize the grid.

        Args:
        dx (float): The lenght between each spatial data point.
        dt (float): The lenght between each temporal data point.
        iterations (Optional, int): The number of models trained with
        the same parameters. Defaults to 3.

        Returns:
            None
        """
        self.dx = dx
        self.dt = dt
        self.x, self.t = load_PINN_data(dx, dt)
        
        self.iterations = iterations

        self.param_dict = config.create_param_dict()
        self.param_grid = config.create_param_grid()

        x_array, t_array = load_numsolver_data(dx, dt)
        self.true = analytical_solution(x_array, t_array)

        self.grid = np.empty(
            (
                len(self.param_dict["activation"]), 
                len(self.param_dict["num_hidden"]), 
                len(self.param_dict["hidden_dim"]), 
                iterations,
            )
        )

        self.indices = {
            "num_hidden": {key: value for value, key in enumerate(self.param_dict["num_hidden"])},
            "hidden_dim": {key: value for value, key in enumerate(self.param_dict["hidden_dim"])},
            "activation": {key: value for value, key in enumerate(self.param_dict["activation"])},
        }

        self._create_grids()

    def _create_grids(self) -> None:
        """
        Create a grid with the mse of each trained model. If several models are trained with
        the same parameters, then the grid consists of an mse of the mses per model.

        Args:
            None
        
        Returns:
            None
        """
        for i in range(self.iterations):
            for params in self.param_grid:
                num_hidden=params["num_hidden"]
                hidden_dim=params["hidden_dim"]
                activation=params["activation"]

                model_file = get_model_filename(num_hidden, hidden_dim, activation, i)

                model = NN(num_hidden, hidden_dim, activation)
                model.load_state_dict(torch.load(model_file, weights_only=True))
                model.eval()

                output = model(self.x, self.t)

                _, _, output = make_data_plottable(self.x, self.t, output, self.dx, self.dt)

                mse = self.mse(output)
                self.grid[
                    self.indices["activation"][activation],
                    self.indices["num_hidden"][num_hidden],
                    self.indices["hidden_dim"][hidden_dim],
                    i,
                ] = mse

        self.grid = np.mean(self.grid, axis=3)

    def mse(self, output):
        """
        Calculate the mean squared error (MSE) of the outputs predicted by a model and the true values.

        Args:
            None

        Returns:
            None
        """
        return np.mean((self.true - output)**2)

    def __str__(self):
        """
        Return the grids in a LaTeX format suitable for tables.

        Args:
            None

        Returns:
            str: The grids in a LaTeX format.
        """
        activation_names = {
            nn.Tanh: "tanh",
            nn.ReLU: "relu",
            nn.LeakyReLU: "leaky_relu",
        }

        grids = ""
        for i, activation in enumerate(self.param_dict["activation"]):
            grids += f"{activation_names[activation]}\n"
            grids += f"{self.latex_table(self.grid[i])}\n\n"
        return grids

    def latex_table(self, table: np.ndarray) -> str:
        """
        Return a single grid in a LaTeX format.

        Args:
            table (np.ndarray): The grid in the form of a np.ndarray.

        Returns: 
            str: The grid in a LaTeX format.
        """
        latex = ""
        for row in table:
            for element in row:
                latex += f"{element:.2e} & "
            latex += "\\\\\n"
        return latex

