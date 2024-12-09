import numpy as np
import torch
import torch.nn as nn

import plot
from PINN import NN
from Diffusion1D import *
from data.generate_data import load_numsolver_data, load_PINN_data, make_data_plottable
import utils
from PINN.grid_search import config
import pandas as pd


def main():
    # Helper variables
    dx_1 = 0.1
    dx_2 = 0.01
    dt_2 = 0.00005
    Nt_1 = 60
    Nt_2 = 6000

    x, t = load_numsolver_data(dx_2, dt_2)

    # Making plot 1
    solver_1 = Diffusion1D(dx_1)
    data_1 = utils.dict_to_matrix(solver_1(Nt=Nt_1, save_step=1))

    solver_2 = Diffusion1D(dx_2)
    data_2 = utils.dict_to_matrix(solver_2(Nt=Nt_2, save_step=1))

    u = utils.analytical_solution(x, t)

    # plot.three_subplots(
    #     x,
    #     t,
    #     data_1,
    #     data_2,
    #     u,
    #     title1=r"Numerical ($\Delta x = 0.1$)",
    #     title2=r"Numerical ($\Delta x = 0.01$)",
    #     title3="Analytical",
    # )

    # Making error plot for PDE
    #plot.diffusion_eq(x, t, u-data_2[:-1], title=r"Error of Numerical Method with $\Delta x = 0.01$", normalize=False)
    plot.diffusion_eq(x, t, abs(u-data_2[:-1]), title=r"Absolute Error of Solver with $\Delta x = 0.01$", normalize=False)
    #plot.diffusion_eq(x, t, (u-data_2[:-1])**2, title=r"Squared Error of Numerical Method with $\Delta x = 0.01$", normalize=False)

    # Making plot 3
    num_hidden = 8
    hidden_dim = 30
    activation = nn.Tanh

    x, t = load_PINN_data(dx_2, dt_2)

    model_file = utils.get_model_filename(
        num_hidden, hidden_dim, activation, iteration=1
    )

    model = NN(num_hidden, hidden_dim, activation)
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()

    output = model(x, t)

    x, t, output = make_data_plottable(x, t, output, dx_2, dt_2)

    # plot.diffusion_eq(x, t, output, "Heat Diffusion in 1D Rod Using PINN")

    # Making error plot for PINN
    #plot.diffusion_eq(x, t, u-output, title=r"Error of PINN", normalize=False)
    plot.diffusion_eq(x, t, abs(u-output), title=r"Absolute Error of PINN", normalize=False)
    #plot.diffusion_eq(x, t, (u-output)**2, title=r"Squared Error of PINN", normalize=False)


    # # Making plot 5
    # time_indices = [100, 1000, 2700, 4500]
    # time_indices = [100, 1500, 3200, 5500]

    # plot.three_subplots(x, t, u, data_2, output, time_indices=time_indices)

    # # Making plot 6
    # exact_data = utils.matrix_to_dict(u)
    # numerical_data = utils.matrix_to_dict(data_2)
    # pinn_data = utils.matrix_to_dict(output)

    # plot.gigaplot(exact_data, numerical_data, pinn_data, x, dt_2, time_indices)


if __name__ == "__main__":
    main()
