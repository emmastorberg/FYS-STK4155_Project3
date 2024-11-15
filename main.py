import numpy as np
import utils
from Diffusion1D import *


def main():
    # Helper variables
    L = 1
    dx = 0.01
    N = int(L / dx)
    dt = 0.00005
    Nt = 10000
    t_max = dt * Nt

    # Plot analytical solution
    x = np.linspace(0, L, N + 1)
    t = np.linspace(0, t_max, Nt + 1)
    u = utils.analytical_solution(x, t)

    utils.plot_diffusion_eq(x, t, u)

    # Make animation of analytical solution:
    utils.make_animation(
        utils.matrix_to_dict(u, save_step=30),
        x,
        dt,
        title="Heat Diffusion in 1D Rod Determined Analytically",
        filename="diffusion_analytical.gif",
    )

    # Plot numerical solution
    solver = Diffusion1D(dx=dx, dt=dt)
    data = utils.dict_to_matrix(solver(Nt=Nt, save_step=1))

    utils.plot_diffusion_eq(solver.x, np.linspace(0, t_max, Nt + 1), data)

    # Make animation of numerical solution
    Nt = 7000
    movie = Diffusion1D(dx=0.01)
    movie_data = movie(Nt, save_step=20)

    movie.animation(movie_data)


if __name__ == "__main__":
    main()
