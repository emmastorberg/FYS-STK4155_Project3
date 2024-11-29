from itertools import product

import torch.nn as nn


def create_param_dict():
    tanh = nn.Tanh
    relu = nn.ReLU
    leaky_relu = nn.LeakyReLU

    param_grid = {
        "num_hidden": [2, 4, 6, 8],
        "hidden_dim": [10, 30, 50, 70],
        "activation": [tanh, relu, leaky_relu],
    }
    return param_grid


def create_param_grid():
    param_grid = create_param_dict()
    return [dict(zip(param_grid, x)) for x in product(*param_grid.values())]