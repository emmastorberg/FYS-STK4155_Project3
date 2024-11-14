import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from typing import Callable
import utils


class Diffusion1D:
    """Class for solving the 1D diffusion equation."""

    def __init__(
        self,
        dx: float,
        dt: float | None = None,
        L: float = 1,
        u0: Callable = utils.initial_condition,
    ) -> None:
        self.dx = dx

        # Testing appropriate dx:
        try:
            assert abs(L / dx - int(L / dx)) < 1e-9
        except:
            raise ValueError(
                f"Rounding error encountered in finding N from choice of dx = {dx}."
            )

        self.N = int(L / dx); N = self.N
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
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), "lil")
        return D

    def __call__(self, Nt: int, cfl: float | None = None, save_step: int = 1) -> dict:
        D = self.D2()
        self.cfl = C = self.cfl if cfl is None else cfl
        self.uj[:] = self.u0(self.x)
        plotdata = {0: self.uj.copy()}

        for n in range(1, Nt + 1):
            self.ujp1[:] = self.uj + C * (D @ self.uj)
            self.ujp1[0] = 0; self.ujp1[-1] = 0
            self.uj[:] = self.ujp1
            if n % save_step == 0:  # save every save_step timestep
                plotdata[n] = self.ujp1.copy()

        return plotdata

    def animation(self, data):
        from matplotlib import animation
        fig, ax = plt.subplots()

        v = np.array(list(data.values()))
        t = np.array(list(data.keys()))
        save_step = t[1] - t[0]

        im = ax.imshow(np.zeros((1, len(self.x))), aspect="auto", cmap="gist_heat", 
                    extent=[self.x.min(), self.x.max(), -0.05, 0.05],  # limited y range for the rod
                    origin="lower", animated=True)

        ax.set_facecolor("#1F1F1F")
        ax.set_ylim(-1, 1)
        ax.yaxis.set_visible(False)
        ax.set_xlabel("Position on Rod (x)")
        ax.set_title("Diffusion of Heat in 1D Rod")

        time_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, color="white", fontsize=12,
                        verticalalignment="top", horizontalalignment="left")

        def update(frame):
            current_time = frame * save_step * self.dt
            heat_data = np.array([data[frame * save_step]])
            img_data = np.zeros_like(self.x)
            img_data[:] = heat_data[0, :]
            im.set_array([img_data])
            im.set_clim(v.min(), v.max())

            time_text.set_text(f"t = {current_time:.3f} s")

            return (im, time_text)

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data), blit=True)
        ani.save("diffusionmovie.apng", writer="pillow", fps=5)
        ani.to_jshtml()
        plt.show()

