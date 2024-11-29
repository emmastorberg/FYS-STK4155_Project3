import numpy as np
import torch
import torch.nn as nn

from PINN import NN
from Diffusion1D import *
from data.generate_data import load_numsolver_data, load_PINN_data, make_data_plottable
import utils
from PINN.grid_search import config
import pandas as pd


def main():
    
    dx, dt = 0.01, 0.0005
    grid = utils.Grid(dx, dt, iterations=3)
    print(grid)
    

    # param_grid = config.create_param_grid()
    # grids = utils.Grid(config.create_param_dict(), x, t)

    # for params in param_grid:
    #     num_hidden=params["num_hidden"]
    #     hidden_dim=params["hidden_dim"]
    #     activation=params["activation"]

    #     model_file = utils.get_model_filename(num_hidden, hidden_dim, activation)

    #     model = NN(num_hidden, hidden_dim, activation)
    #     model.load_state_dict(torch.load(model_file, weights_only=True))
    #     model.eval()

        

    #     x, t = load_PINN_data(dx, dt)
    #     output = model(x, t)

    #     x, t, output = make_data_plottable(x, t, output, dx, dt)

    #     grids.add(num_hidden, hidden_dim, activation, output)



        # utils.plot_diffusion_eq(x, t, output, title=f"Hidden layers: {num_hidden}, Number of nodes: {hidden_dim}, Activation: {activation_name}")

    # activation_name = "tanh"
    # num_hidden = 2
    # hidden_dim = 10
    # activation = nn.Tanh

    # model_file = f"models/nhidden-{num_hidden}_dim-{hidden_dim}_activation-{activation_name}.pt"

    # model = NN(num_hidden, hidden_dim, activation)
    # model.load_state_dict(torch.load(model_file, weights_only=True))
    # model.eval()

    # dx, dt = 0.01, 0.0005

    # x, t = load_PINN_data(dx, dt)
    # output = model(x, t)

    # x, t, output = make_data_plottable(x, t, output, dx, dt)

    # utils.plot_diffusion_eq(x, t, output, title=f"Hidden layers: {num_hidden}, Number of nodes: {hidden_dim}, Activation: {activation_name}")

        # u = utils.analytical_solution(x, t)
        # utils.plot_diffusion_eq(x, t, u, title="analytical")


    # utils.make_animation(
    #     utils.matrix_to_dict(output, save_step=2),
    #     x,
    #     dt,
    #     title="Heat Diffusion in 1D Rod NN",
    # )

    # # Plot numerical solution
    # solver = Diffusion1D(dx=dx, dt=dt)
    # data = utils.dict_to_matrix(solver(Nt=Nt, save_step=1))

    # utils.plot_diffusion_eq(solver.x, np.linspace(0, t_max, Nt + 1), data)

    # # Make animation of numerical solution
    # Nt = 7000
    # movie = Diffusion1D(dx=0.01)
    # movie_data = movie(Nt, save_step=20)

    # movie.animation(movie_data)

    # #from neural network. Not supposed to be here. So that we dont forget how to use it

    # # output = nnet(x, t)
    # # output = output.reshape(N + 1, Nt + 1)
    # # output = output.detach().numpy()

    # # x = np.linspace(0, L, N + 1)
    # # t = np.linspace(0, t_max, Nt + 1)

    # # utils.plot_diffusion_eq(x, t, output.T)


if __name__ == "__main__":
    main()
