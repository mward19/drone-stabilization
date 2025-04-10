import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from paper_drone_utils import rotation_matrix


def plot_trajectory(y, save_path=None):
    X, Y, Z = y[:3, :]  # Shape: (N, 3)

    # Plot the rotor control
    # Plot the trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(X, Y, Z)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Drone Trajectory")
    plt.show()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

def plot_angles(t, y, save_path=None):
    phi, theta, psi = y[3:6, :]

    # Plot the angles
    fig = plt.figure()
    plt.subplot(311)
    plt.plot(t, phi)
    plt.ylabel(r"$\phi$")
    plt.subplot(312)
    plt.plot(t, theta)
    plt.ylabel(r"$\theta$")
    plt.subplot(313)
    plt.plot(t, psi)
    plt.xlabel("Time (seconds)")
    plt.ylabel(r"$\psi$")
    plt.tight_layout()
    plt.suptitle("Drone Angles")
    plt.subplots_adjust(top=0.88)
    plt.show()

    fig.savefig("drone_angles.pdf", bbox_inches="tight")


def plot_rotor_control(t, rotor_ctrl, save_path=None):
    fig = plt.figure()
    plt.plot(t, rotor_ctrl[:, 0], label="Rotor 1")
    plt.plot(t, rotor_ctrl[:, 1], label="Rotor 2")
    plt.plot(t, rotor_ctrl[:, 2], label="Rotor 3")
    plt.plot(t, rotor_ctrl[:, 3], label="Rotor 4")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Rotor Control (rad./sec)")
    plt.legend()
    plt.title("Rotor Control Signals")
    plt.show()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


def set_limits(ax, positions):
    """
    Set equal limits for x, y, and z axes based on the positions of the quadcopter
    throughout the trajectory.
    """
    max_positions = np.max(positions, axis=0)
    min_positions = np.min(positions, axis=0)
    mid = (max_positions + min_positions) / 2
    max_range = np.max(max_positions - min_positions) / 2

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    return max_range


def animate_trajectory(tf, y, save_path="quadcopter_animation.mp4"):
    """
    Animate the trajectory of a quadcopter in 3D space and save the as a video file.
    """
    positions = y[:3, :].T  # Shape: (N, 3)
    angles = y[3:6, :].T  # Shape: (N, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_size = set_limits(ax, positions)

    # Represent the quadcopter as a rectangular prism
    quad = [ax.plot([], [], [], "b-")[0] for _ in range(12)]  # 12 edges of a cube

    def update(frame):
        x, y, z = positions[frame]
        roll, pitch, yaw = angles[frame]

        arm_length = 0.1 * plot_size
        rotor_positions = np.array(
            [
                [arm_length, arm_length, 0],  # front-right
                [-arm_length, arm_length, 0],  # front-left
                [-arm_length, -arm_length, 0],  # back-left
                [arm_length, -arm_length, 0],  # back-right
            ]
        )

        # Include the center point
        body_center = np.array([[0, 0, 0]])
        vertices = np.vstack([body_center, rotor_positions])  # shape: (5, 3)

        # Apply rotation and translation
        rotated = vertices @ rotation_matrix(roll, pitch, yaw) + [x, y, z]

        # Define edges from center to each rotor
        edges = [
            (0, 1),  # center to front-right
            (0, 2),  # center to front-left
            (0, 3),  # center to back-left
            (0, 4),  # center to back-right
        ]

        # Update lines representing arms
        for i, (start, end) in enumerate(edges):
            quad[i].set_data(
                [rotated[start][0], rotated[end][0]],
                [rotated[start][1], rotated[end][1]],
            )
            quad[i].set_3d_properties([rotated[start][2], rotated[end][2]])

    interval = np.ceil((1000 * tf) / (len(positions) - 1))
    fps = int(1000 / interval)
    ani = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval=interval,
    )

    # Save
    ani.save(save_path, writer="ffmpeg", fps=fps)
    plt.close(fig)
    return save_path
