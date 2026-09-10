"""HW1 slope and spoke-count effects on basins and local step-to-step convergence.

Reuse scan_grid, compute_return, and estimate_floquet with the same physical model.
Run: uv run python -m HW1.rimless_wheel_parameter_sweep
Replot saved results: uv run python -m HW1.rimless_wheel_parameter_sweep --plot-only
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import brentq

from .rimless_wheel_floquet import estimate_floquet
from .rimless_wheel_return_map import compute_return
from .rimless_wheel_roa import DEFAULT_PARAMS, scan_grid

DIRECTORY = Path(__file__).resolve().parent
DATA_PATH = DIRECTORY / "rimless_wheel_parameter_sweep_results.npz"
# Hold N=8 for the slope sweep and slope=5 degrees for the spoke-count sweep.
SLOPES_DEG = [0.0, 2.0, 4.0, 5.0, 8.0, 12.0]
SPOKE_COUNTS = list(range(6, 13))
EPSILON = 1e-4  # rad/s; use the same perturbation for every existing rolling cycle.
COLORS = ["#2478b5", "#ed922c", "#999999"]
LABELS = ["Standing", "Rolling", "Unclassified"]


def analyze_rolling_cycle(params):
    """Return a rolling fixed point, multiplier, and cycle status for one parameter set.

    Energy balance supplies the existence check and root bracket only. Compute
    the reported root from the numerical map and the multiplier from perturbed
    simulations. Missing results remain NaN.
    """
    alpha = np.pi / params["num_spokes"]
    slope = params["slope_angle"]
    if not 0 <= slope < alpha:
        raise ValueError("This sweep covers 0 <= slope < alpha only")
    gravity_length = params["gravity"] / params["length"]
    cosine_squared = np.cos(2 * alpha) ** 2
    # Speed must strictly exceed the upright-crossing threshold for a finite return.
    threshold = np.sqrt(2 * gravity_length * (1 - np.cos(slope - alpha)))
    gain = 4 * gravity_length * np.sin(alpha) * np.sin(slope)
    # An algebraic energy-map root may lie outside the physical forward-return domain.
    candidate = np.sqrt(cosine_squared * gain / (1 - cosine_squared))
    result = {
        "fixed_point": np.nan,
        "multiplier": np.nan,
        "step_duration": np.nan,
        "residual": np.nan,
        "threshold": threshold,
        "cycle_status": "no_rolling_cycle",
    }
    if candidate <= threshold:
        return result

    # Keep both endpoints in the valid domain and locate the numerical-map root.
    left = (threshold + candidate) / 2
    right = max(2.0, 2 * candidate)

    def residual(velocity):
        """Subtract input speed from numerical return speed; zero means a repeated step."""
        return compute_return(velocity, params, max_time=80.0) - velocity

    fixed_point = brentq(residual, left, right, xtol=1e-10)
    result.update(
        fixed_point=fixed_point,
        residual=residual(fixed_point),
        step_duration=compute_return(
            fixed_point, params, max_time=80.0, return_details=True
        )["contact_time"],
    )
    # An existing cycle may still have a perturbation outside the valid return domain.
    if fixed_point - EPSILON <= threshold:
        result["cycle_status"] = "cycle_exists_epsilon_outside_domain"
        return result
    multiplier = estimate_floquet(fixed_point, EPSILON, params, max_time=80.0)
    result.update(multiplier=multiplier, cycle_status="rolling_cycle")
    # Use the ideal model's analytic multiplier only to check the numerical estimate.
    if abs(multiplier - cosine_squared) > 1e-5:
        raise RuntimeError("Numerical multiplier disagrees with energy-map derivative")
    return result


def plot_basins(data, kind):
    """Save RoA subplots for one sweep and return the Figure for the caller to close.

    kind is slope or spokes. Each case uses its own physical stance interval.
    Colors show simulated initial-state labels without interpolation between points.
    """
    indices = np.flatnonzero(data["kind"] == kind)
    rows = (len(indices) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(12, 3.1 * rows + 1.1), squeeze=False)
    for ax, index in zip(axes.flat, indices):
        angles = np.rad2deg(data["initial_angles"][index])
        velocities = data["initial_velocities"][index]
        labels = data["roa_results"][index]
        x, y = np.meshgrid(angles, velocities)
        for code, color in enumerate(COLORS):
            mask = labels == code
            ax.scatter(x[mask], y[mask], s=4, c=color, linewidths=0)
        # This is a finite-grid sample fraction, not a global rolling probability.
        fraction = data["counts"][index, 1] / labels.size
        ax.set(
            xlim=(angles[0], angles[-1]),
            ylim=(-2, 2),
            xlabel="Initial angle (deg)",
            ylabel="Initial velocity (rad/s)",
            title=f"Slope {data['slope_degrees'][index]:g}°, N={data['num_spokes'][index]}"
            f"\nRolling samples: {100 * fraction:.1f}%",
        )
        ax.grid(alpha=0.12)
        ax.set_axisbelow(True)
    # Hide unused axes when the case count does not fill the final row.
    for ax in list(axes.flat)[len(indices) :]:
        ax.set_visible(False)
    handles = [
        Line2D([], [], marker="o", ls="", c=c, label=name)
        for c, name in zip(COLORS, LABELS)
    ]
    title = (
        "Slope sweep (8 spokes)" if kind == "slope" else "Spoke-count sweep (5° slope)"
    )
    grid = data["roa_results"].shape[-1]
    fig.suptitle(f"{title} — {grid} × {grid} sampled states per case", fontsize=14)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    path = DIRECTORY / f"rimless_wheel_{kind}_sweep_roa.png"
    fig.savefig(path, dpi=200)
    print(f"Saved: {path}")
    return fig


# Experiment settings and execution; this script is not imported by other analyses.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--grid", type=int, default=31)
parser.add_argument("--plot-only", action="store_true")
parser.add_argument("--no-show", action="store_true")
args = parser.parse_args()
grid_size = args.grid
if grid_size < 2:
    parser.error("--grid must be at least 2")
# --plot-only uses the saved grid; --grid does not change its resolution in this mode.
if args.plot_only:
    with np.load(DATA_PATH) as saved:
        data = {key: saved[key] for key in saved.files}
else:
    cases = [("slope", slope, 8) for slope in SLOPES_DEG]
    cases += [("spokes", 5.0, spokes) for spokes in SPOKE_COUNTS]
    records = []
    # Reuse the shared (5-degree slope, 8-spoke) case instead of simulating it twice.
    cache = {}
    for index, (kind, slope, spokes) in enumerate(cases):
        print(
            f"[{index + 1}/{len(cases)}] {kind}: slope={slope:g} deg, N={spokes}",
            flush=True,
        )
        key = (slope, spokes)
        if key not in cache:
            # Merge into a new dictionary without modifying shared DEFAULT_PARAMS.
            params = DEFAULT_PARAMS | {
                "slope_angle": np.deg2rad(slope),
                "num_spokes": spokes,
            }
            cycle = analyze_rolling_cycle(params)
            scan = scan_grid(
                params,
                grid_size=grid_size,
                velocity_limit=2.0,
                sim_time=20.0,
                max_time=80.0,
                verbose=False,
            )
            counts = np.bincount(scan["roa_results"].ravel(), minlength=3)
            # Check consistency with cycle existence; do not assign grid labels analytically.
            if cycle["cycle_status"] == "no_rolling_cycle" and counts[1] > 0:
                raise RuntimeError("Rolling labels found where no rolling cycle exists")
            cache[key] = scan | cycle | {"counts": counts}
        result = cache[key]
        print(
            f"  standing/rolling/unknown={result['counts'].tolist()}, "
            f"omega*={result['fixed_point']:.9f}, lambda={result['multiplier']:.9f}, "
            f"scan={result['elapsed_seconds']:.2f}s",
            flush=True,
        )
        records.append(
            {"kind": kind, "slope_degrees": slope, "num_spokes": spokes} | result
        )

    # Collect each field across cases and save settings for replotting and verification.
    data = {key: np.array([row[key] for row in records]) for key in records[0]}
    np.savez(
        DATA_PATH,
        **data,
        grid_size=grid_size,
        velocity_limit=2.0,
        epsilon=EPSILON,
        gravity=9.81,
        length=1.0,
        mass=0.2,
        initial_horizon=20.0,
        maximum_horizon=80.0,
        roa_rtol=1e-8,
        roa_atol=1e-10,
        map_rtol=1e-10,
        map_atol=1e-12,
        max_step=0.05,
        stop_velocity=1e-3,
        required_impacts=6,
        rolling_atol=1e-4,
        rolling_rtol=1e-3,
    )
    print(f"Saved: {DATA_PATH}")

# The basin plot is reused for both sweeps; the one-off summary is drawn below.
figures = [plot_basins(data, "slope"), plot_basins(data, "spokes")]

# Compare finite-grid fractions above and per-step multipliers below.
# Stance intervals vary across cases; fractions are not areas in one common rectangle.
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for column, kind in enumerate(["slope", "spokes"]):
    mask = data["kind"] == kind
    values = (
        data["slope_degrees"][mask] if kind == "slope" else data["num_spokes"][mask]
    )
    # Normalize each case by its sample count so the three fractions sum to one.
    fractions = data["counts"][mask] / data["counts"][mask].sum(
        axis=1, keepdims=True
    )
    label = "Slope (deg), N=8" if kind == "slope" else "Number of spokes, slope=5°"
    for code, (name, color) in enumerate(zip(LABELS, COLORS)):
        axes[0, column].plot(values, fractions[:, code], "o-", c=color, label=name)
    axes[0, column].set(
        ylabel="Fraction of sampled initial states", ylim=(-0.03, 1.05)
    )
    multipliers = data["multiplier"][mask]
    axes[1, column].plot(values, multipliers, "o-", c="#8556a5")
    # Leave gaps at NaN; N/A is not a zero multiplier. Default missing cases
    # have no rolling cycle; check cycle_status if extending the parameter range.
    for value, multiplier in zip(values, multipliers):
        if np.isnan(multiplier):
            axes[1, column].text(value, 0.06, "N/A", ha="center", color="0.4")
    axes[1, column].set(ylabel="Floquet multiplier (per step)", ylim=(0, 1))
    axes[1, column].text(
        0.03,
        0.94,
        "N/A: no rolling cycle in these cases",
        transform=axes[1, column].transAxes,
        va="top",
        fontsize=9,
    )
    for ax in axes[:, column]:
        ax.set_xlabel(label)
        ax.set_xticks(values)
        ax.grid(alpha=0.2)
axes[0, 0].legend(frameon=False)
fig.suptitle("Rimless wheel — sampled basins and local convergence", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
path = DIRECTORY / "rimless_wheel_sweep_summary.png"
fig.savefig(path, dpi=200)
print(f"Saved: {path}")
figures.append(fig)
if not args.no_show:
    plt.show()
for figure in figures:
    plt.close(figure)
