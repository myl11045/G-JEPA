"""Plot one or more saved simulator rollouts as frames, error curves, and GIFs."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np
from PIL import Image, ImageDraw

INK, BLUE, GREEN = "#172B4D", "#3478B5", "#389B71"
METHODS = {
    "cem": "Gaussian\nCEM",
    "direct": "Direct prior\nexecution",
    "mixed": "Prior-guided\nplanning",
}
COLORS = {"cem": "#8293A2", "direct": "#3478B5", "mixed": "#E89629"}
TEE = np.array(
    [[-60, 0], [60, 0], [60, 30], [15, 30], [15, 120], [-15, 120], [-15, 30], [-60, 30]]
)


def draw_pusht(ax, state, goal, path=None):
    """Draw the recorded physical states with the actual task-specific goal."""
    ax.set(xlim=(0, 512), ylim=(512, 0), aspect="equal", xticks=[], yticks=[])
    ax.set_facecolor("#FBFCFE")
    for spine in ax.spines.values():
        spine.set(color="#DAE2E9", linewidth=0.7)
    for pose, fill, edge in [(goal, "#E8F4EA", GREEN), (state, "#8293A2", "#5F7285")]:
        cosine, sine = np.cos(pose[4]), np.sin(pose[4])
        points = TEE @ np.array([[cosine, sine], [-sine, cosine]]) + pose[2:4]
        ax.add_patch(Polygon(points, facecolor=fill, edgecolor=edge, linewidth=0.8))
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color=BLUE, alpha=0.5, linewidth=0.7)
    ax.add_patch(
        Circle(state[:2], 15, facecolor=BLUE, edgecolor="white", linewidth=0.7)
    )


def load_rollouts(paths):
    result = []
    for path in paths:
        path = path / "rollout.npz" if path.is_dir() else path
        with np.load(path, allow_pickle=False) as archive:
            item = {key: archive[key].copy() for key in archive.files}
        item["metadata"] = json.loads(item["metadata"].item())
        if result:
            reference = result[0]
            if item["metadata"]["domain"] != reference["metadata"]["domain"]:
                raise ValueError("Compare rollouts from the same domain in one figure")
            for key in ["goal_image", "goal_state"]:
                if not np.array_equal(item[key], reference[key]):
                    raise ValueError("Compared rollouts must have the same goal")
            if not np.allclose(
                item["states"][0], reference["states"][0], rtol=0, atol=1e-6
            ):
                raise ValueError("Compared rollouts must have the same initial state")
        result.append(item)
    return result


def plot_frames(rollouts, output, snapshots, rgb):
    snapshots = min(snapshots, min(len(item["frames"]) for item in rollouts))
    rows, columns = len(rollouts), snapshots + 1
    fig = plt.figure(
        figsize=(2 + 1.65 * columns, 0.65 + 2.15 * rows), facecolor="white"
    )
    grid = fig.add_gridspec(
        rows,
        columns,
        left=0.16,
        right=0.995,
        top=0.90,
        bottom=0.08,
        wspace=0.065,
        hspace=0.30,
    )
    for row, item in enumerate(rollouts):
        meta = item["metadata"]
        indices = np.linspace(
            0, len(item["frames"]) - 1, min(snapshots, len(item["frames"])), dtype=int
        )
        left = None
        for col, index in enumerate([*indices, None]):
            ax = fig.add_subplot(grid[row, col])
            if left is None:
                left = ax.get_position()
            if meta["domain"] == "pusht" and not rgb:
                state = item["goal_state"] if index is None else item["states"][index]
                path = None if index is None else item["states"][: index + 1, :2]
                draw_pusht(ax, state, item["goal_state"], path)
            else:
                ax.imshow(
                    item["goal_image"] if index is None else item["frames"][index]
                )
                ax.set_axis_off()
            title = "Goal" if index is None else f"t = {int(item['steps'][index])}"
            if col == snapshots - 1:
                title = "Final: " + title
            ax.set_title(title, fontsize=10, color=INK, pad=6)
        y = (left.y0 + left.y1) / 2
        height = fig.get_figheight()
        fig.text(
            0.018,
            y + 0.22 / height,
            METHODS[meta["mode"]],
            fontsize=11,
            weight="bold",
            color=INK,
            va="center",
        )
        reached = bool(item["success"].any())
        fig.text(
            0.018,
            y - 0.19 / height,
            "Goal reached" if reached else "Goal not reached",
            fontsize=9,
            color=GREEN if reached else "#986277",
            weight="bold",
        )
        error = f"Final: {item['position_error'][-1]:.1f} {meta['position_unit']}"
        if meta["domain"] == "pusht":
            error += f" | {item['angle_error_deg'][-1]:.1f}°"
        fig.text(0.018, y - 0.39 / height, error, fontsize=8, color="#54677B")
    domain = rollouts[0]["metadata"]["domain"]
    fig.suptitle(
        "PushT" if domain == "pusht" else "OGBench Cube",
        x=0.018,
        ha="left",
        fontsize=14,
        weight="bold",
        color=INK,
    )
    if domain == "pusht":
        note = (
            "Original observations: the green background marker is fixed; the task goal is shown in the last column."
            if rgb
            else "Recorded simulator states. Green outline: task goal. Blue trace: executed pusher path."
        )
        fig.text(0.16, 0.02, note, fontsize=8, color="#54677B")
    for extension in ["png", "svg"]:
        fig.savefig(
            output / f"trajectories.{extension}",
            dpi=200,
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)


def plot_errors(rollouts, output):
    pusht = rollouts[0]["metadata"]["domain"] == "pusht"
    fig, axes = plt.subplots(
        1, 2 if pusht else 1, figsize=(10 if pusht else 5, 3.2), squeeze=False
    )
    for item in rollouts:
        meta = item["metadata"]
        for col, key in enumerate(
            ["position_error", "angle_error_deg"] if pusht else ["position_error"]
        ):
            ax = axes[0, col]
            ax.plot(
                item["steps"],
                item[key],
                color=COLORS[meta["mode"]],
                label=METHODS[meta["mode"]].replace("\n", " "),
                linewidth=1.7,
            )
            ax.set_xlabel("Executed environment steps")
            ax.set_ylabel(
                f"Block position error ({meta['position_unit']})"
                if col == 0
                else "Block angle error (degrees)"
            )
            ax.spines[["right", "top"]].set_visible(False)
    for col, ax in enumerate(axes[0]):
        ax.axhline(
            20 if pusht else 4,
            color="#AAB4BF",
            linestyle="--",
            linewidth=0.8,
            label="Goal threshold",
        )
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for extension in ["png", "svg"]:
        fig.savefig(output / f"goal_error.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_gifs(rollouts, output):
    for index, item in enumerate(rollouts):
        goal = Image.fromarray(item["goal_image"].astype(np.uint8)).convert("RGB")
        frames = []
        for step, array in zip(item["steps"], item["frames"]):
            frame = Image.fromarray(array.astype(np.uint8)).convert("RGB")
            canvas = Image.new(
                "RGB",
                (frame.width + goal.width, max(frame.height, goal.height) + 26),
                "white",
            )
            canvas.paste(frame, (0, 26))
            canvas.paste(goal, (frame.width, 26))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 7), f"Observation: t = {step}", fill=INK)
            draw.text((frame.width + 6, 7), "Goal image", fill=INK)
            frames.append(canvas)
        frames[0].save(
            output / f"rollout_{index}_{item['metadata']['mode']}.gif",
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("runs/figures"))
    parser.add_argument("--snapshots", type=int, default=5)
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="use raw PushT observations instead of physical-state drawings",
    )
    parser.add_argument(
        "--gif", action="store_true", help="also export observation/goal animations"
    )
    args = parser.parse_args()
    if args.snapshots < 2:
        parser.error("--snapshots must be at least 2")
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    rollouts = load_rollouts(args.rollouts)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_frames(rollouts, args.output, args.snapshots, args.rgb)
    plot_errors(rollouts, args.output)
    if args.gif:
        save_gifs(rollouts, args.output)
    print(f"Saved trajectory figures to {args.output}")


if __name__ == "__main__":
    main()
