import sys
import os

sys.path.append("../FYS-STK4155_project3")

import torch
import torch.nn as nn
from tqdm import tqdm

from PINN import NN
from PINN.cost_pinn import cost_total
# from cost_pinn import cost_total
import utils

from data.generate_data import load_PINN_data


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

if __name__ == "__main__":
    x, t = load_PINN_data(dx=0.1, dt=0.005)
    num_hidden = 2
    hidden_dim = 10
    activation = nn.Tanh
    train_model(x, t, num_hidden, hidden_dim, activation)
