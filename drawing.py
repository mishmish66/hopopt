from physics import foot_pos_func

import numpy as np

from einops import einsum

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

import mediapy as media


def animate_robot(qs, interval):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1)

    (bl,) = ax.plot([], [], "salmon", lw=5, alpha=0.75)
    (bd,) = ax.plot([], [], "salmon", lw=5, alpha=0.75)
    (br,) = ax.plot([], [], "salmon", lw=5, alpha=0.75)
    (bu,) = ax.plot([], [], "salmon", lw=5, alpha=0.75)
    (leg,) = ax.plot([], [], "salmon", lw=5, alpha=0.75)
    (foot_marker,) = ax.plot([], [], marker="x", markersize=5, color="k", alpha=0.75)

    body_lines = np.array(
        [
            [[-0.5, -0.5], [0.5, -0.5]],
            [[0.5, -0.5], [0.5, 0.5]],
            [[0.5, 0.5], [-0.5, 0.5]],
            [[-0.5, 0.5], [-0.5, -0.5]],
        ]
    )
    body_artists = [bl, bd, br, bu]
    ground = np.array(
        [
            [-10, 0],
            [10, 0],
        ]
    )
    ax.plot(ground[:, 0], ground[:, 1], "brown")

    # set it to be a square with no axis ticks
    ax.axis("square")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()

    # Get rid of the frame
    for spine in ax.spines.values():
        spine.set_visible(False)

    def draw_frame(i):
        q = qs[i]

        com_vec = q[1:3]
        theta = -q[0]

        rot_mat = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        body_lines_i = (
            einsum(
                body_lines,
                rot_mat,
                "l i j, j k -> l i k",
            )
            + com_vec
        )

        for line_artist, line in zip(body_artists, body_lines_i):
            line_artist.set_data(line[:, 0], line[:, 1])

        leg_line = np.array(
            [
                [0.5, 0.0],
                [q[3] + 1.0, 0.0],
            ]
        )

        leg_line = (
            einsum(
                leg_line,
                rot_mat,
                "i j, j k -> i k",
            )
            + com_vec
        )

        leg.set_data(leg_line[:, 0], leg_line[:, 1])

        foot_pos = foot_pos_func(q)
        foot_marker.set_data([foot_pos[0]], [foot_pos[1]])

        # Set the limits
        ax.set_xlim(-5, 5)
        ax.set_ylim(-3, 7)
        # filter_q = np.nan_to_num(q)
        # ax.set_xlim(-5 + filter_q[1], 5 + filter_q[1])
        # ax.set_ylim(-5 + q[2], 5 + q[2])

        return body_artists + [leg, foot_marker]

    anim = FuncAnimation(
        fig,
        draw_frame,
        frames=range(len(qs)),
        interval=interval,
        blit=True,
    )

    plt.close()
    return anim
