import numpy as np
import torch
import torch.nn as nn

import plot
from PINN import NN
from Diffusion1D import *
from data.generate_data import load_numsolver_data, load_PINN_data, make_data_plottable, load_FFNN_data
import utils
from PINN.grid_search import config
import pandas as pd


def main():
    dx, dt = 0.01, 0.0005

    num_hidden = 8
    hidden_dim = 50
    activation = nn.Tanh

    filename = utils.get_model_filename(num_hidden, hidden_dim, activation, iteration=0)
    # filenameFFNN = utils.get_model_filename(num_hidden, hidden_dim, activation)
    # folder, filenameFFNN = filenameFFNN.split("/")
    # filenameFFNN = f"{folder}/FFNN_{filenameFFNN}"

    model = NN(num_hidden, hidden_dim, activation)
    model.load_state_dict(torch.load(filename, weights_only=True))
    model.eval()

    # FFNN = NN(num_hidden, hidden_dim, activation)
    # FFNN.load_state_dict(torch.load(filenameFFNN, weights_only=True))
    # FFNN.eval()

    x, t = load_PINN_data(dx, dt)
    # outputFFNN = FFNN(x, t)
    # outputPINN = PINN(x, t)
    output = model(x, t)

    # _, _, outputFFNN = make_data_plottable(x, t, outputFFNN, dx, dt)
    # x, t, outputPINN = make_data_plottable(x, t, outputPINN, dx, dt)
    x, t, output = make_data_plottable(x, t, output, dx, dt)

    # x_rand, t_rand = load_FFNN_data(dx, dt)  
    # x_rand = x_rand.detach().numpy() 
    # t_rand = t_rand.detach().numpy()

    # solution = utils.analytical_solution(x, t)

    # plot.two_subplots(x, t, outputFFNN, outputPINN)
    
    plot.diffusion_eq(x, t, abs(output), normalize=True, title="PINN with Unscaled Cost Function")
    


if __name__ == "__main__":
    main()
