import numpy as np
import matplotlib.pyplot as plt


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
