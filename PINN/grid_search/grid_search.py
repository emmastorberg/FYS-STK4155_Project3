from tqdm import tqdm

from train import train_model
from config import create_param_grid
from data.generate_data import load_PINN_data


def run_grid_search(x, t, param_grid):
    for params in tqdm(param_grid, desc="Training Models"):
        train_model(
            x, 
            t,
            num_hidden=params["num_hidden"], 
            hidden_dim=params["hidden_dim"], 
            activation=params["activation"],
        )


def main():
    x, t = load_PINN_data(dx=..., dt=...)
    param_grid = create_param_grid()
    run_grid_search(x, t, param_grid)


if __name__ == "__main__":
    main()
