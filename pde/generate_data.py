import torch
import numpy as np


def load_PINN_data(
        dx: float = 0.02,
        dt: float = 0.01, 
        t_max: float = 0.3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data to solve the one-dimensional diffusion equation for a rod of lenght 1
    with a physics-informed neural network (PINN).
    This function creates a flattened grid of spatial and temporal data points of the rod
    from time 0 to `t_max`.

    Args:
        dx (float): The lenght between each generated spatial data point.
        Defaults to 0.1.
        dt (float): The lenght between each generated temporal data point. 
        Defaults to 0.2.
        t_max (float): Generate temporal data points between time 0 and time `t_max`.
        Defautls to 0.3.

    Returns:
        tuple (torch.Tensor, torch.Tensor):
            A tuple containing torch.Tensors with spatial and temporal data points respectivly.
    """
    L = 1
    Nx = int(L / dx)
    Nt = int(t_max / dt)

    x = torch.linspace(0, L, Nx + 1, requires_grad=True)
    t = torch.linspace(0, t_max, Nt + 1, requires_grad=True)
    X, T = torch.meshgrid(x, t, indexing="ij")
    x = X.flatten().reshape(-1, 1)
    t = T.flatten().reshape(-1, 1)

    return x, t


def load_FFNN_data(
        dx: float = 0.02,
        dt: float = 0.01,
        internal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data to solve the one-dimensional diffusion equation for a rod of lenght 1
    with a standard feed-forward neural network (FFNN).
    This function creates a flattened grid of spatial and temporal data points along the
    rod at time 0, and along the boundaries of the rod from time 0 to time 0.3, with the 
    option of generating internal data points (random, but with a set seed).

    Args:
        dx (float): The lenght between each generated spatial data point.
        Defaults to 0.1.
        dt (float): The lenght between each generated temporal data point. 
        Defaults to 0.2.
        internal (bool): Generate internal data points if `True`. Generate no internal
        data points if `False`. Defaults to `True`.

    Returns:
        tuple (torch.Tensor, torch.Tensor):
            A tuple containing torch.Tensors with spatial and temporal data points respectivly.
    """

    torch.manual_seed(2024)

    L = 1
    t_max = 0.3
    Nx = int(L / dx)
    Nt = int(t_max / dt)

    x = torch.linspace(0, 1, Nx, requires_grad=True)
    t = torch.linspace(dt, 0.3, Nt - 1, requires_grad=True)

    x_zeros = torch.zeros_like(t, requires_grad=True)
    x_ones = torch.ones_like(t, requires_grad=True)
    t_zeros = torch.zeros_like(x, requires_grad=True)


    n_random = 30

    if internal:
        x_random = torch.rand((n_random,), requires_grad=True)
        t_random = torch.rand((n_random,), requires_grad=True) * t_max

        x = torch.cat((x, x_zeros, x_ones, x_random), dim=0).reshape(-1, 1)
        t = torch.cat((t_zeros, t, t, t_random), dim=0).reshape(-1, 1)

    else:
        x = torch.cat((x, x_zeros, x_ones), dim=0).reshape(-1, 1)
        t = torch.cat((t_zeros, t, t), dim=0).reshape(-1, 1)

    return x, t


def load_numsolver_data(
        dx: float = 0.1,
        dt: float = 0.5,
        t_max: float = 0.3
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate data to be used in the numerical solver for the one-dimensional diffusion equation.
    This function creates equally spaced spacial and temporal data points of the rod of length 1,
    from time 0 to `t_max`.

    Args:
        dx (float): The lenght between each generated spatial data point.
        Defaults to 0.1.
        dt (float): The lenght between each generated temporal data point. 
        Defaults to 0.2.
        t_max (float): Generate temporal data points between time 0 and time `t_max`.
        Defautls to 0.3.

    Returns:
        tupe (np.ndarray, np.ndarray):
            A tuple containing np.ndarrays with spatial and temporal data points respectivly.

    """
    L = 1
    Nx = int(L / dx)
    Nt = int(t_max / dt)
    
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, t_max, Nt + 1)

    return x, t