from pde.neural_network.train import train_PINN
from pde.grid_search.config import create_param_grid
from pde.generate_data import load_PINN_data


def run_grid_search(x, t, param_grid, iteration):
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
