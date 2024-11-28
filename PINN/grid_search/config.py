from itertools import product

import torch.nn as nn


def create_param_grid():
    tanh = nn.Tanh
    relu = nn.ReLU
    leaky_relu = nn.LeakyReLU

    param_grid = {
        "num_hidden": [3, 4],#[10, 30, 50, 70],
        "hidden_dim": [5, 10],#[2, 4, 6, 8],
        "activation": [tanh, relu],#[tanh, relu, leaky_relu],
    }

    return [dict(zip(param_grid, x)) for x in product(*param_grid.values())]