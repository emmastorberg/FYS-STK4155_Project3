from typing import Type

import torch
import torch.nn as nn


class NN(nn.Module):
    """
    Class representing a Physics-Informed Neural Network (PINN) to solve the one-dimensional diffusion equation.
    """
    def __init__(
            self, num_hidden: int = 3, 
            hidden_dim: int = 50, 
            activation: Type[nn.Module] = nn.Tanh
        ) -> None:
        """
        Initialize the neural network with a specified number of hidden layers, 
        neurons per hidden layer, and an activation function for the hidden layers.

        The network consists of an input layer, hidden layers, and an output layer. 
        The number of neurons in the input and output layers is fixed (2 and 1, respectively).

        Args:
            num_hidden (int): Number of hidden layers. Defaults to 3.
            hidden_dim (int): Number of neurons per hidden layer. Defaults to 50.
            activation (Type[nn.Module]): The activation function used for all hidden layers.
            Defaults to torch.nn.Tanh.
        """
        super().__init__()

        input_dim = 2
        hidden_dim = 50
        num_hidden = 3
        output_dim = 1

        hiddens = []
        for _ in range(num_hidden):
            hiddens.extend(
                [nn.Linear(hidden_dim, hidden_dim),
                activation()]
            )

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation(),
            *hiddens,
            nn.Linear(hidden_dim, output_dim),
        )

        def init_weights(layer: nn.Linear):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        self.model.apply(init_weights)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): Spatial data points. Expected shape (N, 1).
            t (torch.Tensor): Temporal data points. Expected shape (N, 1).

        Returns:
            torch.Tensor: The output of the neural network, representing the solution to 
            the diffusion equation at the given spatial and temporal points.
        """
        inputs = torch.cat((x, t), dim=1)
        outputs = self.model(inputs)
        return outputs
