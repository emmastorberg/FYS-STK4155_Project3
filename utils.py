import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from PINN.grid_search import config
from PINN import NN
from data.generate_data import make_data_plottable, load_PINN_data, load_numsolver_data


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


def plot_diffusion_eq(
    space_axis: np.ndarray,
    time_axis: np.ndarray,
    solution_grid: np.ndarray,
    title: str = "Heat Diffusion in 1D Rod",
) -> None:
    """Plot the heat at different x-coordinates as a function of time.

    Args:
        space_axis (np.ndarray): x-axis, i.e. location in space
        time_axis (np.ndarray): time axis
        solution_grid (np.ndarray): heat given as a 2D matrix grid.
        title (str, optional): Title to be displayed on plot. Defaults to "Heat Diffusion in 1D Rod".

    """
    plt.imshow(
        solution_grid,
        aspect="auto",
        extent=[space_axis.min(), space_axis.max(), time_axis.max(), time_axis.min()],
        origin="upper",
        cmap="afmhot",
        vmin=0,
        vmax=1,
    )

    plt.xlabel("Position on Rod (x)")
    plt.ylabel("Time (s)")
    plt.colorbar()
    plt.title(title)
    plt.show()


def make_animation(
    data: dict,
    rod_coordinates: np.ndarray,
    dt: float,
    title: str = "Heat Diffusion in 1D Rod",
    filename: str = "diffusion.gif",
) -> None:
    """Make animation of heat diffusion in 1D rod.

    Args:
        data (dict): Dictionary of plotting data
        rod_coordinates (np.ndarray): x-axis, i.e. location in space
        dt (float): timestep length
        title (str, optional): Title to be displayed on animation. Defaults to "Heat Diffusion in 1D Rod".
        filename (str, optional): Filename to save animation as. Defaults to "diffusion.gif".

    Returns:
        _type_: _description_
    """
    from matplotlib import animation

    fig, ax = plt.subplots()

    v = np.array(list(data.values()))
    t = np.array(list(data.keys()))
    save_step = t[1] - t[0]

    im = ax.imshow(
        np.zeros((1, len(rod_coordinates))),
        aspect="auto",
        cmap="afmhot",
        extent=[
            rod_coordinates.min(),
            rod_coordinates.max(),
            -0.05,
            0.05,
        ],  # limited y range for the rod
        origin="lower",
        animated=True,
    )

    ax.set_facecolor("#1F1F1F")
    ax.set_ylim(-1, 1)  # Limit the height of the heat region (short rod)
    ax.yaxis.set_visible(False)
    ax.set_xlabel("Position on Rod (x)")
    ax.set_title(title)

    # Text object for displaying time in seconds
    time_text = ax.text(
        0.05,
        0.95,
        "",
        transform=ax.transAxes,
        color="white",
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
    )

    def update(frame):
        current_time = frame * save_step * dt
        heat_data = np.array([data[frame * save_step]])
        img_data = np.zeros_like(rod_coordinates)
        img_data[:] = heat_data[0, :]
        im.set_array([img_data])
        im.set_clim(v.min(), v.max())

        # Update the time display text
        time_text.set_text(f"t = {current_time:.3f} s")  # Format the time nicely

        return (im, time_text)

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=len(data), blit=True, interval=20
    )
    ani.save(filename, writer="pillow")
    ani.to_jshtml()
    plt.show()


def get_model_filename(num_hidden, hidden_dim, activation, iteration = None):
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
    filename = f"models/nhidden-{num_hidden}_dim-{hidden_dim}_activation-{activation_name}{iter}.pt"

    return filename

class Grid:
    def __init__(self, dx, dt, iterations = 3):
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

        self.create_grids()

    def create_grids(self):
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
        return np.mean((self.true - output)**2)

    def __str__(self):
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

    def latex_table(self, table: np.ndarray):
        latex = ""
        for row in table:
            for element in row:
                latex += f"{element:.2e} & "
            latex += "\\\\\n"
        return latex
    



