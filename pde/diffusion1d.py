from typing import Callable

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from pde import utils


class Diffusion1D:
    """Class for solving the 1D diffusion equation in a 1D rod."""

    def __init__(
        self,
        dx: float,
        dt: float | None = None,
        L: float = 1,
        u0: Callable = utils.initial_condition,
    ) -> None:
        """Initialization of Diffusion1D numerical solver.

        Args:
            dx (float): The step length in space (x-direction).
            dt (float | None, optional): The timestep length. Can be chosen explicitly
                or left empty, in which case a suitable value of dt ensuring stability will be chosen.
                Defaults to None.
            L (float, optional): The length of the rod (maximum value in x-direction). Defaults to 1.
            u0 (Callable, optional): Function describing initial condition, i.e. expression for
                x-values where t = 0. Defaults to utils.initial_condition.

        Raises:
            ValueError: If the choice of dx causes a rounding error
            ValueError: If the choice of dt in combination with dx causes numerical instability.
        """
        self.dx = dx

        # Testing appropriate dx:
        try:
            assert abs(L / dx - int(L / dx)) < 1e-9
        except:
            raise ValueError(
                f"Rounding error encountered in finding N from choice of dx = {dx}."
            )

        self.N = int(L / dx)
        N = self.N
        self.L = L
        self.x = np.linspace(0, L, N + 1)
        self.ujp1 = np.zeros(N + 1)
        self.uj = np.zeros(N + 1)
        self.u0 = u0

        if dt is None:
            dt = utils.find_dt_for_stability(dx)

        # Testing stability with choice of dt:
        try:
            assert dt / (dx) ** 2 <= 0.5
        except:
            raise ValueError(f"Unstable scheme with dt = {dt}.")

        self.dt = dt
        self.cfl = dt / (dx) ** 2

    def D2(self) -> np.ndarray:
        """Create D2 matrix containing scheme for computing the solution at the next timestep.

        Returns:
            np.ndarray: The D2 matrix.
        """
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), "lil")
        return D

    def __call__(self, Nt: int, cfl: float | None = None, save_step: int = 1) -> dict:
        """Solve the diffusion equation with the chosen parameters.

        Args:
            Nt (int): Number of timesteps to march forward.
            cfl (float | None, optional): CFL number (numerical stability constant). 
                Can be specified to use a different number than the one found with dx 
                and dt, but the value input will NOT be tested for stability. If nothing 
                is specified, own CFL based on dx and dt attributes will be used. 
                Defaults to None.
            save_step (int, optional): The timestep interval for which to save solution 
                vectors for plotting later. Defaults to 1.

        Returns:
            dict: Dictionary of solution vectors at specific timesteps. Vectors are spaced 
                at intervals of save_step.
        """
        D = self.D2()
        self.cfl = C = self.cfl if cfl is None else cfl
        self.uj[:] = self.u0(self.x)
        plotdata = {0: self.uj.copy()}

        for n in range(1, Nt + 1):
            self.ujp1[:] = self.uj + C * (D @ self.uj)
            self.ujp1[0] = 0
            self.ujp1[-1] = 0
            self.uj[:] = self.ujp1
            if n % save_step == 0:  # save every save_step timestep
                plotdata[n] = self.ujp1.copy()

        return plotdata

    def animation(self, data: dict) -> None:
        """Animate heat diffusion in a 1D rod. 

        Args:
            data (dict): A dictionary containing all points to plot with timestep indices as keys.
        """
        from matplotlib import animation

        fig, ax = plt.subplots()

        v = np.array(list(data.values()))
        t = np.array(list(data.keys()))
        save_step = t[1] - t[0]

        im = ax.imshow(
            np.zeros((1, len(self.x))),
            aspect="auto",
            cmap="afmhot",
            interpolation=None,
            extent=[
                self.x.min(),
                self.x.max(),
                -0.05,
                0.05,
            ],  # limited y range for the rod
            origin="lower",
            animated=True,
        )

        ax.set_facecolor("#1F1F1F")
        ax.set_ylim(-1, 1)
        ax.yaxis.set_visible(False)
        ax.set_xlabel("Position on Rod (x)")
        ax.set_title("Heat Diffusion in 1D Rod Determined Numerically")

        time_text = ax.text(
            0.05,
            0.95,
            "",
            transform=ax.transAxes,
            color="white",
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="left",
        )

        def update(frame):
            current_time = frame * save_step * self.dt
            heat_data = np.array([data[frame * save_step]])
            img_data = np.zeros_like(self.x)
            img_data[:] = heat_data[0, :]
            im.set_array([img_data])
            im.set_clim(v.min(), v.max())

            time_text.set_text(f"t = {current_time:.3f} s")

            return (im, time_text)

        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=len(data), blit=True, interval=20
        )
        ani.save("numerical_diffusion.gif", writer="pillow")
        ani.to_jshtml()
        plt.show()
