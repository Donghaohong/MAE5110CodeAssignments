"""Plot saved RoA classifications without rerunning the dynamics.

Run: uv run python -m HW1.plot_rimless_wheel_roa
Read rimless_wheel_roa_results_event.npz by default and save a PNG alongside it.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Input and output options; this plotting script runs directly from top to bottom.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--input",
    type=Path,
    default=Path(__file__).with_name("rimless_wheel_roa_results_event.npz"),
)
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).with_name("rimless_wheel_roa_event.png"),
)
parser.add_argument("--no-show", action="store_true")
args = parser.parse_args()

# Load the sampled initial states and labels; do not rerun or interpolate simulations.
with np.load(args.input) as data:
    # Keep stored angles in radians; convert to degrees only for display.
    angles = np.rad2deg(data["initial_angles"])
    velocities = data["initial_velocities"]
    labels = data["roa_results"]
    slope = float(np.rad2deg(data["slope_angle"]))
    spokes = int(data["num_spokes"])

# Match the label matrix: rows index velocity, columns index angle.
angle_grid, velocity_grid = np.meshgrid(angles, velocities)
fig, ax = plt.subplots(figsize=(9, 6.5))
colors = ["#2478b5", "#ed922c", "#999999"]
names = ["Standing", "Rolling", "Unclassified"]
# Shrink markers on dense grids to reduce overlap that resembles a filled region.
size = max(3, min(65, 15000 / labels.size))
for code, color in enumerate(colors):
    # Select coordinates belonging to this class with a Boolean mask.
    mask = labels == code
    ax.scatter(angle_grid[mask], velocity_grid[mask], s=size, color=color, zorder=2)
# Keep every class and its count in the legend, even when its count is zero.
handles = [
    Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        color=color,
        label=f"{name} ({np.count_nonzero(labels == code)})",
    )
    for code, (color, name) in enumerate(zip(colors, names))
]

ax.set(
    xlim=(angles[0], angles[-1]),
    ylim=(velocities[0], velocities[-1]),
    xlabel="Initial angle (deg)",
    ylabel="Initial angular velocity (rad/s)",
    title=f"Rimless wheel RoA samples | slope {slope:g}°, {spokes} spokes\n"
    f"{len(angles)} × {len(velocities)} initial states",
)
ax.grid(alpha=0.15)
ax.set_axisbelow(True)
ax.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    frameon=False,
)
fig.tight_layout()
fig.savefig(args.output, dpi=200, bbox_inches="tight")
print(f"Saved: {args.output}")
if not args.no_show:
    plt.show()
plt.close(fig)
