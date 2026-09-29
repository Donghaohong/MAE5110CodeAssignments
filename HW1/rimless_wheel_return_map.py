"""HW1 return map from one downhill post-impact velocity to the next.

Run: uv run python -m HW1.rimless_wheel_return_map
Save map data and a plot; the Floquet analysis reuses the saved fixed point.
One step means a change of support spoke, not an integration timestep.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from models import rimless_wheel as model

from .rimless_wheel_roa import DEFAULT_PARAMS, advance_to_impact


def compute_return(angular_velocity, params, max_time=20.0, *, return_details=False):
    """Map post-impact speed omega to the next post-impact speed, both in rad/s.

    The initial angle is fixed at gamma - alpha, leaving one state coordinate.
    Require the next contact to be downhill; reverse contact or timeout returns
    NaN, not zero. return_details=True returns a dictionary including status,
    contact time, and pre-/post-impact states.
    """
    alpha = np.pi / params["num_spokes"]
    slope = params["slope_angle"]
    if not 0 <= slope < alpha or not 6 <= params["num_spokes"] <= 12:
        raise ValueError(
            "This forward-step map supports 0 <= slope < alpha, 6-12 spokes"
        )
    if not np.isfinite(angular_velocity) or max_time <= 0:
        raise ValueError("Use a finite velocity and a positive time horizon")

    # Keep invalid returns as NaN rather than plotting a missing step as zero speed.
    result = {
        "next_velocity": np.nan,
        "contact_time": np.nan,
        "pre_impact_state": np.full(2, np.nan),
        "post_impact_state": np.full(2, np.nan),
        "status": "nonpositive_initial_velocity",
    }
    if angular_velocity > 0:
        state = np.array([slope - alpha, angular_velocity])
        # At the energy threshold, upright is approached only at infinite time.
        # Exclude roundoff-sized equality to avoid a spurious finite forward return.
        barrier = 2 * params["gravity"] / params["length"] * (1 - np.cos(state[0]))
        if np.isclose(angular_velocity**2, barrier, rtol=1e-12, atol=1e-14):
            result["status"] = "separatrix_no_finite_return"
        else:
            time, before, hit = advance_to_impact(
                state,
                0.0,
                max_time,
                params,
                rtol=1e-10,
                atol=1e-12,
                max_step=0.05,
            )
            if not hit:
                result["status"] = "no_contact_before_timeout"
            else:
                result["contact_time"] = time
                result["pre_impact_state"] = before
                # Do not skip reverse contact and call a later downhill contact one step.
                if before[1] <= 0 or abs(before[0] - (slope + alpha)) > 1e-8:
                    result["status"] = "reverse_contact_not_a_forward_step"
                else:
                    # Use the same convention on both axes: post-impact velocity.
                    after = model.reset_state(time, before, params)
                    result["post_impact_state"] = after
                    result["next_velocity"] = float(after[1])
                    result["status"] = "forward_return"
    return result if return_details else result["next_velocity"]


def find_fixed_points(velocities, next_velocities, params, max_time=20.0):
    """Find P(omega) = omega within sampled valid intervals and return the roots.

    Bracket residual sign changes between adjacent samples and refine roots
    using the numerical map. This does not guarantee finding roots outside
    the samples or every root without a sign change.
    """
    roots = []

    def residual(velocity):
        """Return the vertical difference from the identity line; a root repeats speed."""
        return compute_return(velocity, params, max_time) - velocity

    for index in range(len(velocities) - 1):
        left, right = velocities[index : index + 2]
        mapped_left, mapped_right = next_velocities[index : index + 2]
        # Do not bracket a root across samples without valid returns.
        if not np.isfinite(mapped_left) or not np.isfinite(mapped_right):
            continue
        left_residual, right_residual = mapped_left - left, mapped_right - right
        if abs(left_residual) < 1e-10:
            root = left
        elif left_residual * right_residual < 0:
            root = brentq(residual, left, right, xtol=1e-10)
        else:
            continue
        # Adjacent intervals can identify the same root; store it only once.
        if not roots or abs(root - roots[-1]) > 1e-7:
            roots.append(root)
    # The loop checks left endpoints; check the final sample separately.
    if (
        np.isfinite(next_velocities[-1])
        and abs(next_velocities[-1] - velocities[-1]) < 1e-10
        and (not roots or abs(velocities[-1] - roots[-1]) > 1e-7)
    ):
        roots.append(velocities[-1])
    return np.array(roots)


# Floquet and parameter-sweep code import the map without running this experiment.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=201)
    parser.add_argument("--max-velocity", type=float, default=2.0)
    parser.add_argument("--max-time", type=float, default=20.0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.samples < 2 or args.max_velocity <= 0 or args.max_time <= 0:
        parser.error("Use at least two samples and positive limits")
    # Sample the post-impact section and locate intersections with the identity line.
    params = DEFAULT_PARAMS.copy()
    velocities = np.linspace(0.0, args.max_velocity, args.samples)
    records = [
        compute_return(v, params, args.max_time, return_details=True)
        for v in velocities
    ]
    next_velocities = np.array([r["next_velocity"] for r in records])
    statuses = np.array([r["status"] for r in records])
    roots = find_fixed_points(velocities, next_velocities, params, args.max_time)

    # Independently check flight energy balance and reset, without replacing simulation.
    # energy_gain is an increment in speed squared, in (rad/s)**2, not joules.
    alpha = np.pi / params["num_spokes"]
    slope = params["slope_angle"]
    gravity_length = params["gravity"] / params["length"]
    threshold = np.sqrt(2 * gravity_length * (1 - np.cos(slope - alpha)))
    cosine = np.cos(2 * alpha)
    energy_gain = 4 * gravity_length * np.sin(alpha) * np.sin(slope)
    valid = np.isfinite(next_velocities)
    reference = cosine * np.sqrt(velocities[valid] ** 2 + energy_gain)
    max_error = (
        float(np.max(np.abs(reference - next_velocities[valid])))
        if np.any(valid)
        else np.nan
    )

    print(
        f"Section: theta = {np.rad2deg(slope - alpha):.6f} deg, AFTER downhill impact"
    )
    print(f"Forward-step threshold: omega > {threshold:.9f} rad/s")
    print(f"Valid forward returns: {np.count_nonzero(valid)}/{args.samples}")
    for status, count in zip(*np.unique(statuses, return_counts=True)):
        print(f"  {status}: {count}")
    if not roots.size:
        print("No fixed point found within the sampled valid domain.")
    for root in roots:
        details = compute_return(root, params, args.max_time, return_details=True)
        print(f"Fixed point: {root:.9f} rad/s")
        print(f"P(omega*) - omega*: {details['next_velocity'] - root:.3e} rad/s")
        print(f"One-step duration at fixed point: {details['contact_time']:.9f} s")
    print(f"Maximum error vs energy-balance formula: {max_error:.3e} rad/s")

    directory = Path(__file__).resolve().parent
    data_path = directory / "rimless_wheel_return_map_results.npz"
    figure_path = directory / "rimless_wheel_return_map.png"
    # Save unrounded roots and matching parameters for the later perturbation analysis.
    np.savez(
        data_path,
        initial_velocities=velocities,
        next_velocities=next_velocities,
        statuses=statuses,
        contact_times=np.array([r["contact_time"] for r in records]),
        pre_impact_states=np.array([r["pre_impact_state"] for r in records]),
        post_impact_states=np.array([r["post_impact_state"] for r in records]),
        fixed_points=roots,
        forward_threshold=threshold,
        max_formula_error=max_error,
        max_time=args.max_time,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.05,
        section="immediately after downhill impact",
        **params,
    )
    # Plot the sampled map, identity line, and fixed point in the same experiment.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(
        velocities,
        next_velocities,
        color="tab:blue",
        lw=2,
        label=r"Numerical return map $P(\omega_k)$",
    )
    ax.plot(
        velocities,
        velocities,
        "--",
        color="black",
        lw=1.3,
        label=r"Identity line $\omega_{k+1}=\omega_k$",
    )
    # Shading denotes no finite forward step, not a zero-valued portion of the map.
    ax.axvspan(0, threshold, color="0.9", label="No finite forward-step return")
    ax.axvline(threshold, color="0.55", ls=":", lw=1)
    # The intersection (omega*, omega*) represents periodic rolling, not rest.
    if roots.size:
        ax.scatter(roots, roots, color="tab:red", s=65, zorder=5, label="Fixed point")
        for root in roots:
            ax.annotate(
                f"Fixed point: ({root:.6f}, {root:.6f})",
                xy=(root, root),
                xytext=(0.40, 0.88),
                textcoords="axes fraction",
                fontsize=10,
                arrowprops={"arrowstyle": "->", "color": "tab:red"},
            )
    slope = np.rad2deg(params["slope_angle"])
    ax.set(
        xlabel=r"Post-impact velocity $\omega_k$ (rad/s)",
        ylabel=r"Next post-impact velocity $\omega_{k+1}$ (rad/s)",
        title=f"Rimless wheel: one-step return map\nSlope {slope:g}°, {params['num_spokes']} spokes",
        xlim=(velocities[0], velocities[-1]),
        ylim=(0, velocities[-1]),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    if not args.no_show:
        plt.show()
    plt.close(fig)
    print(f"Saved data: {data_path}")
    print(f"Saved plot: {figure_path}")
