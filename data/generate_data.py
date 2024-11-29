
import torch
import numpy as np


def load_PINN_data(dx: float = 0.1, dt: float = 0.0005):
    L = 1
    t_max = 0.3
    Nx = int(L / dx)
    Nt = int(t_max / dt)

    x = torch.linspace(0, L, Nx + 1, requires_grad=True)
    t = torch.linspace(0, t_max, Nt + 1, requires_grad=True)
    X, T = torch.meshgrid(x, t, indexing="ij")
    x = X.flatten().reshape(-1, 1)
    t = T.flatten().reshape(-1, 1)

    return x, t


def make_data_plottable(x, t, output, dx, dt):
    L = 1
    t_max = 0.3
    Nx = int(L / dx)
    Nt = int(t_max / dt)

    output = output.reshape(Nx + 1, Nt + 1)
    output = output.detach().numpy()

    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, t_max, Nt + 1)

    return x, t, output.T
    

def load_numsolver_data(dx: float = 0.1, dt: float = 0.5):
    L = 1
    t_max = 0.3
    Nx = int(L / dx)
    Nt = int(t_max / dt)
    
    x = np.linspace(0, L, Nx + 1)
    t = np.linspace(0, t_max, Nt + 1)

    return x, t