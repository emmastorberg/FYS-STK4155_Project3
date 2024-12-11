import torch
import torch.nn as nn
from tqdm import tqdm

from pde.neural_network import NN
from pde.neural_network.cost_functions import cost_FFNN, cost_PINN
from pde.generate_data import load_PINN_data, load_numsolver_data, load_FFNN_data
from pde import utils


def train_PINN(x, t, num_hidden, hidden_dim, activation, iteration = None):
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

    torch.save(nnet.state_dict(), filename)


if __name__ == "__main__":
    dx, dt, t_max = 0.02, 0.01, 0.03
    x, t = load_PINN_data(dx, dt, t_max)

    num_hidden = 8
    hidden_dim = 50
    activation = nn.Tanh

    train_PINN(x, t, num_hidden, hidden_dim, activation)
