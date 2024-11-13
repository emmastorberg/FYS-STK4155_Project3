import numpy as np
import sympy as sp
from scipy import sparse
import matplotlib.pyplot as plt

x, t = sp.symbols("x", "t")


class Diffusion:
    """Class for solving the diffusion equation."""

    def __init__(
        self, N: int, cfl: float = 1, L: float = 1, u0: sp.Function = sp.sin(sp.pi * x)
    ) -> None:
        self.N = N
        self.cfl = cfl
        self.L = L
        self.x = np.linspace(0, L, N + 1)
        self.dx = L / N
        self.u0 = u0
        self.ujp1 = np.zeros(N + 1)
        self.uj = np.zeros(N + 1)

    def D2(self) -> np.ndarray:
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (self.N + 1, self.N + 1), "lil")
        return D

    # @property
    # def dt(self) -> float:
    #     raise NotImplementedError
    #     return self.cfl * self.dx / self.c

    def __call__(self, Nt: int, cfl: float | None = None, save_step: int = 1) -> dict:
        D = self.D2()
        self.cfl = C = self.cfl if cfl is None else cfl
        # dt = self.dt
        u0 = sp.lambdify(x, self.u0.subs({t: 0}))
        self.uj[:] = u0(self.x)
        plotdata = {0: self.uj.copy()}

        for n in range(1, Nt + 1):
            self.ujp1[:] = self.uj + (1 / C) * (D @ self.uj)
            self.uj[:] = self.ujp1
            if n % save_step == 0:  # save every save_step timestep
                plotdata[n] = self.ujp1.copy()

        return plotdata

    # def plot_with_offset(self, data):
    #     Nd = len(data)
    #     v = np.array(list(data.values()))
    #     t = np.array(list(data.keys()))
    #     dt = t[1] - t[0]
    #     v0 = abs(v).max()
    #     fig = plt.figure(facecolor="k")
    #     ax = fig.add_subplot(111, facecolor="k")
    #     for i, u in data.items():
    #         ax.plot(self.x, u + i * v0 / dt, "w", lw=2, zorder=i)
    #         ax.fill_between(
    #             self.x, u + i * v0 / dt, i * v0 / dt, facecolor="k", lw=0, zorder=i - 1
    #         )
    #     plt.show()

    # def animation(self, data):
    #     from matplotlib import animation

    #     fig, ax = plt.subplots()
    #     v = np.array(list(data.values()))
    #     t = np.array(list(data.keys()))
    #     save_step = t[1] - t[0]
    #     (line,) = ax.plot(self.x, data[0])
    #     ax.set_ylim(v.min(), v.max())

    #     def update(frame):
    #         line.set_ydata(data[frame * save_step])
    #         return (line,)

    #     ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data), blit=True)
    #     ani.save(
    #         "wavemovie.apng", writer="pillow", fps=5
    #     )  # This animated png opens in a browser
    #     ani.to_jshtml()
    #     plt.show()

# CHANGE TEST!!!!!! WILL NOT PASS!!!!!!
def test_pulse_bcs():
    sol = Diffusion(100, cfl=1, L0=2, c0=1)
    data = sol(100, bc=0, ic=0, save_step=100)
    assert np.linalg.norm(data[0] + data[100]) < 1e-12
    data = sol(100, bc=0, ic=1, save_step=100)
    assert np.linalg.norm(data[0] + data[100]) < 1e-12
    data = sol(100, bc=1, ic=0, save_step=100)
    assert np.linalg.norm(data[0] - data[100]) < 1e-12
    data = sol(100, bc=1, ic=1, save_step=100)
    assert np.linalg.norm(data[0] - data[100]) < 1e-12
    data = sol(100, bc=2, ic=0, save_step=100)
    assert np.linalg.norm(data[100]) < 1e-12
    data = sol(100, bc=2, ic=1, save_step=100)
    assert np.linalg.norm(data[100]) < 1e-12
    data = sol(100, bc=3, ic=0, save_step=100)
    assert np.linalg.norm(data[0] - data[100]) < 1e-12
    data = sol(100, bc=3, ic=1, save_step=100)
    assert np.linalg.norm(data[0] - data[100]) < 1e-12

if __name__ == "__main__":
    # sol = Diffusion(100, cfl=1, L=1)
    # data = sol(100, save_step=1)
    # sol.animation(data)
    # test_pulse_bcs() # test needs to be changed
    # data = sol(200, save_step=100)
    ...