import sys
import os

sys.path.append("../FYS-STK4155_project3")

import torch
import torch.nn as nn
from tqdm import tqdm

from neural_network import NN
from cost_pinn import cost_total

from data.generate_data import load_PINN_data


def train_model(x, t, num_hidden, hidden_dim, activation):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nnet = NN(num_hidden, hidden_dim, activation)
    nnet = nnet.to(device)
    optimizer = torch.optim.Adam(nnet.parameters())

    activation_names = {
        nn.Tanh: "tanh",
        nn.ReLU: "relu",
        nn.LeakyReLU: "leaky_relu",
    }

    epochs = 5000
    pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        cost = cost_total(x, t, nnet)
        cost.backward()
        optimizer.step()
        pbar.set_postfix(loss=cost.item())
        pbar.update()

    filename = (
        f"nhidden-{num_hidden}_"
        f"dim-{hidden_dim}_"
        f"activation-{activation_names.get(activation, "unknown")}"
    )

    torch.save(nnet.state_dict(), f"models/{filename}.pt")

if __name__ == "__main__":
    x, t = load_PINN_data(dx=0.1, dt=0.005)
    num_hidden = 5
    hidden_dim = 100
    activation = nn.Tanh
    train_model(x, t, num_hidden, hidden_dim, activation)
