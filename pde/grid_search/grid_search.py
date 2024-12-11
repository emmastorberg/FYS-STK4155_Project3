from typing import Optional

import torch

from pde.neural_network.train import train_PINN
from pde.grid_search.config import create_param_grid
from pde.generate_data import load_PINN_data


def run_grid_search(
        x: torch.Tensor,
        t: torch.Tensor,
        param_grid: list[dict],
        iteration: Optional[int] = None,
    ) -> None:
    """
    Run a grid search with parameters from `param_grid`.
    All models are trained and saved in pde/models.

    Args:
        x (torch.Tensor): Spatial data points. Expected shape (N, 1).
        t (torch.Tensor): Temporal data points. Expected shape (N, 1).
        param_grid (list): A list of dictionaries with parameters
        to be used in model training.
            All dictionaries must have the following keys: 
                `num_hidden`, `hidden_dim` and `activation`.
        iteration (Optional, int): Current iteration if several models are trained with the 
        given parameters. Defualts to None.

    Returns:
        None
    """
    for params in param_grid:
        train_PINN(
            x, 
            t,
            params["num_hidden"], 
            params["hidden_dim"], 
            params["activation"],
            iteration,
        )


def main():
    x, t = load_PINN_data(dx=0.1, dt=0.005)
    param_grid = create_param_grid()
    iterations = 3
    for i in range(iterations):
        run_grid_search(x, t, param_grid, i)


if __name__ == "__main__":
    main()
