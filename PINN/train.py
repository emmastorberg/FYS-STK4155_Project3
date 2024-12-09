import sys
import os

sys.path.append("../FYS-STK4155_project3")

import torch
import torch.nn as nn
from tqdm import tqdm

from PINN import NN
from PINN.cost_pinn import cost_total, cost_FFNN
# from cost_pinn import cost_total
import utils

from data.generate_data import load_PINN_data, load_numsolver_data, load_FFNN_data


def train_model(x, t, num_hidden, hidden_dim, activation, iteration = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nnet = NN(num_hidden, hidden_dim, activation)
    nnet = nnet.to(device)
    optimizer = torch.optim.Adam(nnet.parameters())

    epochs = 3000
    pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        cost = cost_total(x, t, nnet)
        cost.backward()
        optimizer.step()
        pbar.set_postfix(loss=cost.item())
        pbar.update()

    filename = utils.get_model_filename(num_hidden, hidden_dim, activation, iteration)

    torch.save(nnet.state_dict(), filename)


def train_FFNN(x, t, num_hidden, hidden_dim, activation):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nnet = NN(num_hidden, hidden_dim, activation)
    nnet = nnet.to(device)
    optimizer = torch.optim.Adam(nnet.parameters())

    epochs = 5000
    pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        cost = cost_FFNN(x, t, nnet)
        cost.backward(retain_graph=True)
        optimizer.step()
        pbar.update()

    filename = utils.get_model_filename(num_hidden, hidden_dim, activation)
    folder, filename = filename.split("/")
    filename = f"{folder}/FFNN_{filename}"
    # print(filename)

    torch.save(nnet.state_dict(), filename)


if __name__ == "__main__":
    dx, dt, t_max = 0.02, 0.01, 0.01
    x, t = load_FFNN_data(dx, dt, internal=True)
    num_hidden = 8
    hidden_dim = 50
    activation = nn.Tanh

    # x_array, t_array = load_numsolver_data(dx, dt, t_max)
    # true = utils.analytical_solution(x_array, t_array)
    # print(true.shape)

    train_FFNN(x, t, num_hidden, hidden_dim, activation)
