import sys
import os

sys.path.append("../FYS-STK4155_project3")

from tqdm import tqdm

from PINN import train_model
from config import create_param_grid
from data.generate_data import load_PINN_data


def run_grid_search(x, t, param_grid, iteration):
    for params in param_grid:
        train_model(
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
