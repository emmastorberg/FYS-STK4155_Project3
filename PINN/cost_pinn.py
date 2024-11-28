import torch

from neural_network import NN


def cost_pde_residual(x: torch.Tensor, t: torch.Tensor, nnet: NN) -> torch.Tensor:
    """
    Compute the loss associated with the residual of the PDE.

    Args:
        x (torch.Tensor): Spatial data points. Expected shape (N, 1).
        t (torch.Tensor): Temporal data points. Expected shape (N, 1).
        nnet (NN): The neural network model to compute the solution of the PDE.

    Returns:
        torch.Tensor: The mean squared residual between the second spatial derivative and time derivative.
    """
    u = nnet(x, t)
    u_t, u_x = torch.autograd.grad(
        u, (t, x), grad_outputs=torch.ones_like(u), create_graph=True
    )
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    return torch.mean((u_xx - u_t) ** 2)


def cost_initial_condition(x: torch.Tensor, t: torch.Tensor, nnet: NN) -> torch.Tensor:
    """
    Compute the loss for the initial condition of the PDE.

    This function calculates the difference between the exact initial condition (u(x, t=0) = sin(pi * x)) 
    and the network's prediction at t=0, and returns the mean squared error.

    Args:
        x (torch.Tensor): Spatial data points. Expected shape (N, 1).
        t (torch.Tensor): Temporal data points. Expected shape (N, 1).
        nnet (NN): The neural network model to compute the solution of the PDE.

    Returns:
        torch.Tensor: The mean squared error between the exact initial condition and the network's prediction.
    """
    ic = torch.sin(torch.pi * x)
    t = torch.zeros_like(t)
    u0 = nnet(x, t)
    return torch.mean((ic - u0) ** 2)


def cost_boundary_condition(x: torch.Tensor, t: torch.Tensor, nnet: NN) -> torch.Tensor:
    """
    Compute the loss for the boundary conditions of the PDE.

    This function calculates the network's output at the boundaries (x=0 and x=L) for all time steps 
    and returns the mean squared error.

    Args:
        x (torch.Tensor): Spatial data points. Expected shape (N, 1).
        t (torch.Tensor): Temporal data points. Expected shape (N, 1).
        nnet (NN): The neural network model to compute the solution of the PDE.

    Returns:
        torch.Tensor: The mean squared error at the boundaries (x=0 and x=L).
    """
    L = 1
    x0 = torch.zeros_like(x)
    x1 = torch.ones_like(x) * L
    u_bc0 = nnet(x0, t)
    u_bc1 = nnet(x1, t)
    return torch.mean(u_bc0**2 + u_bc1**2)


def cost_total(x: torch.Tensor, t: torch.Tensor, nnet: NN) -> torch.Tensor:
    """
    Compute the total loss combining the PDE residual, initial condition, and boundary conditions.

    Args:
        x (torch.Tensor): Spatial data points. Expected shape (N, 1).
        t (torch.Tensor): Temporal data points. Expected shape (N, 1).
        nnet (NN): The neural network model to compute the solution of the PDE.

    Returns:
        torch.Tensor: The combined total loss.
    """
    loss_r = 20 * cost_pde_residual(x, t, nnet)
    loss_ic = cost_initial_condition(x, t, nnet)
    loss_bc = 50 * cost_boundary_condition(x, t, nnet)

    return loss_r + loss_ic + loss_bc
