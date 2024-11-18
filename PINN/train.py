import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from neural_network import NN
from cost_pinn import cost_total


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    L = 1
    dx = 0.1
    N = int(L / dx)
    dt = 0.005
    Nt = 100
    t_max = dt * Nt

    x = torch.linspace(0, L, N + 1, requires_grad=True)
    t = torch.linspace(0, t_max, Nt + 1, requires_grad=True)
    X, T = torch.meshgrid(x, t, indexing="ij")
    x = X.flatten().reshape(-1, 1)
    t = T.flatten().reshape(-1, 1)

    nnet = NN()
    nnet = nnet.to(device)
    optimizer = torch.optim.Adam(nnet.parameters())

    epochs = 50000
    pbar = tqdm(total=epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        cost = cost_total(x, t, nnet)
        cost.backward()
        optimizer.step()
        pbar.set_postfix(loss=cost.item())
        pbar.update()

    filename = ""

    torch.save(nnet.state_dict(), f"models/{filename}.pt")


if __name__ == "__main__":
    main()