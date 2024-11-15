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


def loss_pde_residual(x: torch.Tensor, t: torch.Tensor, nnet: NN):
    u = nnet(x, t)
    u_t, u_x = torch.autograd.grad(
        u, (t, x), grad_outputs=torch.ones_like(u), create_graph=True
    )
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    return torch.mean((u_xx - u_t) ** 2)


def loss_initial_condition(x: torch.Tensor, t: torch.Tensor, nnet: NN):
    ic = torch.sin(torch.pi * x)
    t = torch.zeros_like(t)
    u0 = nnet(x, t)
    return torch.mean((ic - u0) ** 2)


def loss_boundary_condition(x: torch.Tensor, t: torch.Tensor, nnet: NN):
    L = 1
    x0 = torch.zeros_like(x)
    x1 = torch.ones_like(x) * L
    u_bc0 = nnet(x0, t)
    u_bc1 = nnet(x1, t)
    return torch.mean(u_bc0**2 + u_bc1**2)


def loss(x: torch.Tensor, t: torch.Tensor, nnet: NN) -> float:
    """
    Total loss = loss(PDE residual) + loss(initial condition) + loss(boundary condition)
    """
    loss_r = 200 * loss_pde_residual(x, t, nnet)
    loss_ic = loss_initial_condition(x, t, nnet)
    loss_bc = loss_boundary_condition(x, t, nnet)

    return loss_r + loss_ic + loss_bc
