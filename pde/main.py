import numpy as np
import torch
import torch.nn as nn

import pde.plotutils as plot
from pde.neural_network import NN
from pde.neural_network.train import train_FFNN, train_PINN
from pde.diffusion1d import Diffusion1D
from pde.generate_data import load_numsolver_data, load_PINN_data, load_FFNN_data
from pde import utils


def train():
    dx, dt = 0.2, 0.01

    x, t = load_PINN_data(dx, dt)
    train_PINN(x, t, num_hidden, hidden_dim, activation)


def main():
    dx, dt = 0.01, 0.0005

    # load data
    x_PINN, t_PINN = load_PINN_data(dx, dt)
    x, t = load_numsolver_data(dx, dt)


    # load trained neural network model, and get outputs
    num_hidden = 8
    hidden_dim = 50
    activation = nn.Tanh

    filename = utils.get_model_filename(num_hidden, hidden_dim, activation)

    model = NN(num_hidden, hidden_dim, activation)
    model.load_state_dict(torch.load(filename, weights_only=True))
    model.eval()

    output_PINN = model(x_PINN, t_PINN)

    # x and t values here are the same as generated in `load_numsolver_data`
    # transform data as type torch.tensor to numpy.ndarray
    x, t, output_PINN = utils.make_data_plottable(x, t, output_PINN, dx, dt)


    # numerical solver
    Nt = 6000
    solver = Diffusion1D(dx)
    output_solver_dict = solver(Nt, save_step=1)
    output_solver = utils.dict_to_matrix(output_solver_dict)


    # compute analytical solution
    analytical_solution = utils.analytical_solution(x, t)


    # plot results
    plot.diffusion_eq(x, t, output_PINN)
    plot.diffusion_eq(x, t, output_solver)
    plot.diffusion_eq(x, t, analytical_solution)


if __name__ == "__main__":
    num_hidden = 8
    hidden_dim = 50
    activation = nn.Tanh

    # train()
    main()
