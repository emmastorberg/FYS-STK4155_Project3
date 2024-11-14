import numpy as np
import matplotlib.pyplot as plt


def initial_condition(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def find_dt_for_stability(dx: float) -> float:
    # Ensure dx is positive
    if dx <= 0:
        raise ValueError("dx must be positive")

    # Calculate the maximum allowable dt
    dt = 0.5 * dx**2

    return dt


def analytical_solution(x: np.ndarray, t: np.ndarray):
    xx, tt = np.meshgrid(x, t, sparse=True)
    return np.sin(np.pi * xx) * np.exp(-np.pi**2 * tt)


def dict_to_matrix(data_dict: dict):
    # Only possible to use if save_step = 1
    matrix = np.array(list(data_dict.values()))
    return matrix


def matrix_to_dict(data_matrix: np.ndarray, save_step: int = 1) -> dict:
    data_dict = {}

    for i in range(0, len(data_matrix), save_step):
        data_dict[i] = data_matrix[i]
    return data_dict


def plot_diffusion_eq(
    space_axis: np.ndarray, time_axis: np.ndarray, solution_grid: np.ndarray
) -> None:
    h = plt.contourf(space_axis, time_axis, solution_grid)
    plt.axis("scaled")
    plt.xlabel("Position on Rod (x)")
    plt.ylabel("Time (s)")
    plt.colorbar()
    plt.show()


def make_animation(data: dict, rod_coordinates: np.ndarray, dt: float) -> None:
    from matplotlib import animation

    fig, ax = plt.subplots()

    v = np.array(list(data.values()))
    t = np.array(list(data.keys()))
    save_step = t[1] - t[0]

    im = ax.imshow(
        np.zeros((1, len(rod_coordinates))),
        aspect="auto",
        cmap="cubehelix",
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
    ax.set_title("Diffusion of Heat in 1D Rod")

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

    ani = animation.FuncAnimation(fig=fig, func=update, frames=len(data), blit=True)
    ani.save("diffusionmovie.apng", writer="pillow", fps=5)
    ani.to_jshtml()
    plt.show()


def generate_dataset_for_NN(): 
    # Use analytical solution somehow to create dataset
    ...
