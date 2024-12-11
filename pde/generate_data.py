import torch
import numpy as np


def load_PINN_data(dx: float = 0.1, dt: float = 0.0005, t_max: float = 0.3):
    L = 1
    Nx = int(L / dx)
    Nt = int(t_max / dt)

    x = torch.linspace(0, L, Nx + 1, requires_grad=True)
    t = torch.linspace(0, t_max, Nt + 1, requires_grad=True)
    X, T = torch.meshgrid(x, t, indexing="ij")
    x = X.flatten().reshape(-1, 1)
    t = T.flatten().reshape(-1, 1)

    return x, t


def load_FFNN_data(dx, dt, internal: bool = True):
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


def load_numsolver_data(dx: float = 0.1, dt: float = 0.5, t_max: float = 0.3):
    L = 1
    Nx = int(L / dx)
    Nt = int(t_max / dt)
    
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, t_max, Nt + 1)

    return x, t