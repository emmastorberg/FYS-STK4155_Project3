from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from pde.neural_network import NN
from pde.neural_network.cost_functions import cost_FFNN, cost_PINN
from pde.generate_data import load_PINN_data, load_numsolver_data, load_FFNN_data
from pde import utils


def train_PINN(
        x: torch.Tensor,
        t: torch.Tensor,
        num_hidden: int,
        hidden_dim: int,
        activation: str,
        iteration: Optional[int] = None,
    ) -> None:
    """
    Train a physics-informed neural network (PINN) with specified parameters.
    The model is saved in pde/models.

    Args:
        x (torch.Tensor): Spatial data points. Expected shape (N, 1).
        t (torch.Tensor): Temporal data points. Expected shape (N, 1).
        num_hidden (int): Number of hidden layers.
        hidden_dim (int): Number of nodes per hidden layer.
        activation (str): Name of activation function. Possible values are:
            'tanh', 'relu', 'leaky_relu'.
        iteration (Optional, int): The current iteration if several models are trained with
        these parameters. Defaults to None.

    Returns
        None
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nnet = NN(num_hidden, hidden_dim, activation)
    nnet = nnet.to(device)
    optimizer = torch.optim.Adam(nnet.parameters())

    epochs = 3000
    pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        cost = cost_PINN(x, t, nnet)
        cost.backward()
        optimizer.step()
        pbar.set_postfix(loss=cost.item())
        pbar.update()

    filename = utils.get_model_filename(num_hidden, hidden_dim, activation, iteration)

    torch.save(nnet.state_dict(), filename)


def train_FFNN(
        x: torch.Tensor,
        t: torch.Tensor,
        num_hidden: int,
        hidden_dim: int,
        activation: str,
    ) -> None:
    """
    Train a feed-forward neural network with specified parameters.
    The model is saved in pde/models.

    Args:
        x (torch.Tensor): Spatial data points. Expected shape (N, 1).
        t (torch.Tensor): Temporal data points. Expected shape (N, 1).
        num_hidden (int): Number of hidden layers.
        hidden_dim (int): Number of nodes per hidden layer.
        activation (str): Name of activation function. Possible values are:
            'tanh', 'relu', 'leaky_relu'.

    Returns:
        None
    """
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

    torch.save(nnet.state_dict(), filename)
