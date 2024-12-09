import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, ScalarFormatter


def diffusion_eq(
    space_axis: np.ndarray,
    time_axis: np.ndarray,
    solution_grid: np.ndarray,
    title: str = "Heat Diffusion in 1D Rod",
    points: tuple[np.ndarray] | None = None,
    normalize: bool = True,
) -> None:
    """Plot the heat at different x-coordinates as a function of time.

    Args:
        space_axis (np.ndarray): x-axis, i.e. location in space
        time_axis (np.ndarray): time axis
        solution_grid (np.ndarray): heat given as a 2D matrix grid.
        title (str, optional): Title to be displayed on plot. Defaults to "Heat Diffusion in 1D Rod".
        points (tuple[np.ndarray], optional): Points to overlay on the plot with x-values first,
        time values second. Defaults to None.
        normalize (bool): Determines whether to normalize color bar from 0 to 1.

    """
    if normalize:
        plt.imshow(
            solution_grid,
            aspect="auto",
            extent=[
                space_axis.min(),
                space_axis.max(),
                time_axis.max(),
                time_axis.min(),
            ],
            origin="upper",
            cmap="afmhot",
            vmin=0,
            vmax=1,
        )

        plt.colorbar()

    else:
        img = plt.imshow(
            solution_grid,
            aspect="auto",
            extent=[
                space_axis.min(),
                space_axis.max(),
                time_axis.max(),
                time_axis.min(),
            ],
            origin="upper",
            cmap="cividis",
        )

        colorbar = plt.colorbar(img)

        formatter = ScalarFormatter()
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        colorbar.ax.yaxis.set_major_formatter(formatter)

    if points is not None:
        x, t = points
        plt.scatter(x, t, marker="x", c="lightskyblue")
    plt.xlabel("Position on Rod (x)")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.show()


def make_animation(
    data: dict,
    rod_coordinates: np.ndarray,
    dt: float,
    title: str = "Heat Diffusion in 1D Rod",
    filename: str = "diffusion.gif",
) -> None:
    """Make animation of heat diffusion in 1D rod.

    Args:
        data (dict): Dictionary of plotting data
        rod_coordinates (np.ndarray): x-axis, i.e. location in space
        dt (float): timestep length
        title (str, optional): Title to be displayed on animation. Defaults to "Heat Diffusion in 1D Rod".
        filename (str, optional): Filename to save animation as. Defaults to "diffusion.gif".

    Returns:
        _type_: _description_
    """
    from matplotlib import animation

    fig, ax = plt.subplots()

    v = np.array(list(data.values()))
    t = np.array(list(data.keys()))
    save_step = t[1] - t[0]

    im = ax.imshow(
        np.zeros((1, len(rod_coordinates))),
        aspect="auto",
        cmap="afmhot",
        extent=[
            rod_coordinates.min(),
            rod_coordinates.max(),
            -0.05,
            0.05,
        ],  # limited y range for the rod
        origin="lower",
        animated=True,
    )

    ax.set_facecolor("#1F1F1F")
    ax.set_ylim(-1, 1)  # Limit the height of the heat region (short rod)
    ax.yaxis.set_visible(False)
    ax.set_xlabel("Position on Rod (x)")
    ax.set_title(title)

    # Text object for displaying time in seconds
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
        current_time = frame * save_step * dt
        heat_data = np.array([data[frame * save_step]])
        img_data = np.zeros_like(rod_coordinates)
        img_data[:] = heat_data[0, :]
        im.set_array([img_data])
        im.set_clim(v.min(), v.max())

        # Update the time display text
        time_text.set_text(f"t = {current_time:.3f} s")  # Format the time nicely

        return (im, time_text)

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=len(data), blit=True, interval=20
    )
    ani.save(filename, writer="pillow")
    ani.to_jshtml()
    plt.show()


def two_subplots(
    space_axis: np.ndarray,
    time_axis: np.ndarray,
    solution_grid1: np.ndarray,
    solution_grid2: np.ndarray,
    suptitle: str | None = None,
    title1: str | None = "FFNN",
    title2: str | None = "PINN",
) -> None:
    """Plot the heat at different x-coordinates as a function of time in two subplots.

    Args:
        space_axis (np.ndarray): x-axis, i.e. location in space
        time_axis (np.ndarray): time axis
        solution_grid1 (np.ndarray): heat given as a 2D matrix grid for first subplot
        solution_grid2 (np.ndarray): heat given as a 2D matrix grid for second subplot
        suptitle (str | None, optional): Title to be displayed over both subplots. Defaults to None.
        title1 (str | None, optional): Title to be displayed over first subplot. Defaults to "FFNN".
        title2 (str | None, optional): Title to be displayed over second subplot. Defaults to "PINN".
    """

    fig, ax = plt.subplots(1, 2, sharey=True)  # include figsize arg here if necessary

    vmin = min(solution_grid1.min(), solution_grid2.min())
    vmax = max(solution_grid1.max(), solution_grid2.max())

    # Create a Normalize object to scale the colorbar between the min and max values
    norm = Normalize(vmin=vmin, vmax=vmax)

    cax1 = ax[0].imshow(
        solution_grid1,
        aspect="auto",
        extent=[space_axis.min(), space_axis.max(), time_axis.max(), time_axis.min()],
        origin="upper",
        cmap="afmhot",
        norm=norm,
    )

    ax[0].set_xlabel("Position on Rod (x)")
    ax[0].set_ylabel("Time (s)")

    if title1 is not None:
        ax[0].set_title(title1)

    cax2 = ax[1].imshow(
        solution_grid2,
        aspect="auto",
        extent=[space_axis.min(), space_axis.max(), time_axis.max(), time_axis.min()],
        origin="upper",
        cmap="afmhot",
        norm=norm,
    )

    ax[1].set_xlabel("Position on Rod (x)")
    ax[1].tick_params(axis="y", which="both", left=False, right=False)

    if title2 is not None:
        ax[1].set_title(title2)

    fig.colorbar(cax1, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)

    if suptitle is not None:
        fig.title(suptitle)

    # Show the plot
    plt.show()


def three_subplots(
    space_axis: np.ndarray,
    time_axis: np.ndarray,
    solution_grid1: np.ndarray,
    solution_grid2: np.ndarray,
    solution_grid3: np.ndarray,
    suptitle: str | None = None,
    title1: str | None = "Analytical",
    title2: str | None = "Numerical",
    title3: str | None = "PINN",
    time_indices: list[int] | None = None,
) -> None:
    """Plot the heat at different x-coordinates as a function of time in three subplots.

    Args:
        space_axis (np.ndarray): x-axis, i.e. location in space
        time_axis (np.ndarray): time axis
        solution_grid1 (np.ndarray): heat given as a 2D matrix grid for first subplot
        solution_grid2 (np.ndarray): heat given as a 2D matrix grid for second subplot
        solution_grid2 (np.ndarray): heat given as a 2D matrix grid for third subplot
        suptitle (str | None, optional): Title to be displayed over both subplots. Defaults to None.
        title1 (str | None, optional): Title to be displayed over first subplot. Defaults to "Exact".
        title2 (str | None, optional): Title to be displayed over second subplot. Defaults to "Numerical".
        title2 (str | None, optional): Title to be displayed over third subplot. Defaults to "PINN".
        time_indices (list[int] | None): List containing indices of timesteps, which will add red lines over the plots. Defaults to None.
    """
    fig, ax = plt.subplots(1, 3, sharey=True)  # include figsize arg here if necessary

    vmin = min(solution_grid1.min(), solution_grid2.min(), solution_grid3.min())
    vmax = max(solution_grid1.max(), solution_grid2.max(), solution_grid3.max())

    # Create a Normalize object to scale the colorbar between the min and max values
    norm = Normalize(vmin=vmin, vmax=vmax)

    cax1 = ax[0].imshow(
        solution_grid1,
        aspect="auto",
        extent=[space_axis.min(), space_axis.max(), time_axis.max(), time_axis.min()],
        origin="upper",
        cmap="afmhot",
        norm=norm,
    )

    cax2 = ax[1].imshow(
        solution_grid2,
        aspect="auto",
        extent=[space_axis.min(), space_axis.max(), time_axis.max(), time_axis.min()],
        origin="upper",
        cmap="afmhot",
        norm=norm,
    )

    cax3 = ax[2].imshow(
        solution_grid2,
        aspect="auto",
        extent=[space_axis.min(), space_axis.max(), time_axis.max(), time_axis.min()],
        origin="upper",
        cmap="afmhot",
        norm=norm,
    )

    if time_indices is not None:
        time_values = time_axis[time_indices]

        for time in time_values:
            ax[0].axhline(y=time, color="r", linewidth=0.5)
            ax[1].axhline(y=time, color="r", linewidth=0.5)
            ax[2].axhline(y=time, color="r", linewidth=0.5)

    ax[0].set_xlabel("Position on Rod (x)")
    ax[0].set_ylabel("Time (s)")

    ax[1].set_xlabel("Position on Rod (x)")
    ax[1].tick_params(axis="y", which="both", left=False, right=False)

    ax[2].set_xlabel("Position on Rod (x)")
    ax[2].tick_params(axis="y", which="both", left=False, right=False)

    if title1 is not None:
        ax[0].set_title(title1)

    if title2 is not None:
        ax[1].set_title(title2)

    if title3 is not None:
        ax[2].set_title(title3)

    fig.colorbar(cax1, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)

    if suptitle is not None:
        fig.title(suptitle)

    # Show the plot
    plt.show()


def gigaplot(
    exact_data: dict,
    numerical_data: dict,
    pinn_data: dict,
    rod_coordinates: np.ndarray,
    dt: float,
    time_indices: list[int],
) -> None:
    """Make gigaplot of animation frames at different timesteps.

    Args:
        exact_data (dict): Dictionary of plotting data from exact solution.
        numerical_data (dict): Dictionary of plotting data from numerical solver.
        pinn_data (dict): Dictionary of plotting data from PINN.
        rod_coordinates (np.ndarray): x-axis, i.e. location in space
        dt (float): timestep length
        time_indices (list[int]): List containing indices of timesteps to create freeze frames of. Length must be at least 2 for code to run.
    """

    fig, ax = plt.subplots(len(time_indices), 4, figsize=(12, 16))
    plt.subplots_adjust(wspace=0.1)

    for row in range(len(time_indices)):
        pos_1 = ax[row, 0].get_position()
        pos_2 = ax[row, 1].get_position()
        pos_3 = ax[row, 2].get_position()
        pos_4 = ax[row, 3].get_position()

        ax[row, 0].set_position([pos_1.x0, pos_1.y0, pos_1.width, pos_1.height])
        ax[row, 1].set_position([pos_2.x0, pos_2.y0, pos_2.width, pos_2.height])
        ax[row, 2].set_position([pos_3.x0, pos_3.y0, pos_3.width, pos_3.height])

        # increase gap between third and fourth columns
        ax[row, 3].set_position([pos_4.x0 + 0.05, pos_4.y0, pos_4.width, pos_4.height])

    def freeze_frame(data, row, column, time_index, xaxislabel=False):
        v = np.array(list(data.values()))
        t = np.array(list(data.keys()))
        save_step = t[1] - t[0]

        im = ax[row, column].imshow(
            np.zeros((1, len(rod_coordinates))),
            aspect="auto",
            cmap="afmhot",
            extent=[
                rod_coordinates.min(),
                rod_coordinates.max(),
                -0.05,
                0.05,
            ],  # limited y range for the rod
            origin="lower",
            animated=True,
        )

        ax[row, column].set_facecolor("#1F1F1F")
        ax[row, column].set_ylim(-1, 1)
        ax[row, column].yaxis.set_visible(False)

        if xaxislabel:
            ax[row, column].set_xlabel("Position on Rod (x)")

        def custom_tick_formatter(x, pos):
            if x == 0 or x == 1:
                return f"{int(x)}"
            else:
                return f"{x:.1f}"

        ax[row, column].xaxis.set_major_formatter(FuncFormatter(custom_tick_formatter))

        # text object for displaying time in seconds
        time_text = ax[row, column].text(
            0.05,
            0.95,
            "",
            transform=ax[row, column].transAxes,
            color="white",
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="left",
        )

        current_time = time_index * save_step * dt
        heat_data = np.array([exact_data[time_index * save_step]])
        img_data = np.zeros_like(rod_coordinates)
        img_data[:] = heat_data[0, :]
        im.set_array([img_data])
        im.set_clim(v.min(), v.max())

        time_text.set_text(f"t = {current_time:.3f} s")

    for row, index in enumerate(time_indices):
        last_index = False

        if time_indices[row] == time_indices[-1]:
            last_index = True

        freeze_frame(exact_data, row, 0, index, last_index)
        freeze_frame(numerical_data, row, 1, index, last_index)
        freeze_frame(pinn_data, row, 2, index, last_index)

        ax[row, 3].plot(
            rod_coordinates,
            exact_data[index],
            color="r",
            label="Analytical",
            linewidth=3,
        )
        ax[row, 3].plot(
            rod_coordinates,
            numerical_data[index],
            color="limegreen",
            label="Numerical",
            linestyle="--",
            linewidth=3,
        )
        ax[row, 3].plot(
            rod_coordinates,
            pinn_data[index],
            color="b",
            label="PINN",
            linestyle=":",
            linewidth=3,
        )
        ax[row, 3].yaxis.set_visible(True)
        ax[row, 3].set_yticks(np.linspace(0, 1, 5))
        ax[row, 3].set_yticklabels([f"{tick:.2f}" for tick in np.linspace(0, 1, 5)])
        ax[row, 3].set_ylabel(r"$u(x, t)$")
        ax[row, 3].grid(True)

        if row == 0:
            ax[row, 0].set_title("Analytical")
            ax[row, 1].set_title("Numerical")
            ax[row, 2].set_title("PINN")
            ax[row, 3].set_title("Heat at Various Timesteps")
            ax[row, 3].legend()

        if last_index:
            ax[row, 3].set_xlabel("Position on Rod (x)")

    plt.savefig("selected_timesteps.png")
    plt.show()
