import numpy as np
import utils
import torch

from PINN import NN
from Diffusion1D import *


def main():
    # model = NN()
    # model.load_state_dict(torch.load("models/dx01_dt00005_Nt1000_optimAdam_epoch5000.pt", weights_only=True))
    # model.eval()

    # Helper variables
    L = 1
    dx = 0.1
    N = int(L / dx)
    dt = 0.0005
    Nt = 1000
    t_max = dt * Nt

    # Plot analytical solution
    # x = torch.linspace(0, L, N + 1, requires_grad=True)
    # t = torch.linspace(0, t_max, Nt + 1, requires_grad=True)
    # X, T = torch.meshgrid(x, t, indexing="ij")
    # x = X.flatten().reshape(-1, 1)
    # t = T.flatten().reshape(-1, 1)

    # output = model(x, t)
    # output = output.reshape(N + 1, Nt + 1)
    # output = output.detach().numpy()

    x = np.linspace(0, L, N + 1)
    t = np.linspace(0, t_max, Nt + 1)

    # utils.plot_diffusion_eq(x, t, output.T)
    u = utils.analytical_solution(x, t)
    utils.plot_diffusion_eq(x, t, u)

    # # Make animation of analytical solution:
    # utils.make_animation(
    #     utils.matrix_to_dict(output.T, save_step=2),
    #     x,
    #     dt,
    #     title="Heat Diffusion in 1D Rod NN",
    # )

    # # Plot numerical solution
    # solver = Diffusion1D(dx=dx, dt=dt)
    # data = utils.dict_to_matrix(solver(Nt=Nt, save_step=1))

    # utils.plot_diffusion_eq(solver.x, np.linspace(0, t_max, Nt + 1), data)

    # # Make animation of numerical solution
    # Nt = 7000
    # movie = Diffusion1D(dx=0.01)
    # movie_data = movie(Nt, save_step=20)

    # movie.animation(movie_data)

    # #from neural network. Not supposed to be here. So that we dont forget how to use it

    # # output = nnet(x, t)
    # # output = output.reshape(N + 1, Nt + 1)
    # # output = output.detach().numpy()

    # # x = np.linspace(0, L, N + 1)
    # # t = np.linspace(0, t_max, Nt + 1)

    # # utils.plot_diffusion_eq(x, t, output.T)


if __name__ == "__main__":
    main()
