import torch
import torch.nn as nn


class NN(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = 2
        hidden_dim = 50
        num_hidden = 3
        output_dim = 1

        hiddens = []
        for _ in range(num_hidden):
            hiddens.extend(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
            )

        self.model = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            *hiddens,
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        def init_weights(layer: nn.Linear):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

        self.model.apply(init_weights)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat((x, t), dim=1)
        outputs = self.model(inputs)
        return outputs
