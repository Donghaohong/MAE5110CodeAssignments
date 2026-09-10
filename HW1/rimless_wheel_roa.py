"""HW1 RoA estimation by independently simulating and classifying a state grid.

Read simulate for one experiment, scan_grid for repetition, then the script block.
integrate_to_impact handles adaptive integration between contacts using the model.
Run: uv run python -m HW1.rimless_wheel_roa; plot with plot_rimless_wheel_roa.py.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from models import rimless_wheel as model

# Numeric labels support storage and plotting; unclassified is not a third attractor.
RESULT_CODES = {"standing": 0, "rolling": 1, "unclassified": 2}
DEFAULT_PARAMS = {
    "gravity": 9.81,
    "length": 1.0,
    "mass": 0.2,
    "slope_angle": np.deg2rad(5.0),
    "num_spokes": 8,
}


def integrate_to_impact(state, start_time, end_time, params, *, rtol=1e-8, atol=1e-10, max_step=0.05):
    """Integrate until the next contact or end_time and return the solver result.

    State is [theta, omega] in rad and rad/s; times are in seconds. rtol/atol
    control local integration error, while max_step is a ceiling, not a fixed
    timestep. No reset is applied here: a contact endpoint is still pre-impact.
    """
    alpha = np.pi / params["num_spokes"]
    slope = params["slope_angle"]

    def downhill_event(t, state):
        """Return the signed angle offset, zero at downhill contact gamma + alpha."""
        return state[0] - (slope + alpha)

    def uphill_event(t, state):
        """Return the signed angle offset, zero at uphill contact gamma - alpha."""
        return state[0] - (slope - alpha)

    # Accept only outward crossings; terminal stops integration at the event.
    downhill_event.direction = 1
    downhill_event.terminal = True
    uphill_event.direction = -1
    uphill_event.terminal = True

    def continuous_dynamics(t, state):
        """Bind params to match the solver's (t, state) function signature."""
        return model.dynamics(t, state, params)

    # Tiny rocking motions can return soon after departure; cap the restart step.
    # Speed divided by maximum acceleration gives a scale, not an exact turning time.
    departing_contact = (
        abs(state[0] - (slope - alpha)) < 1e-12
        and state[1] > 0
    ) or (
        abs(state[0] - (slope + alpha)) < 1e-12
        and state[1] < 0
    )
    first_step = None
    if departing_contact:
        first_step = min(
            max_step,
            end_time - start_time,
            0.1 * abs(state[1]) / (params["gravity"] / params["length"]),
        )

    # Adapt steps and locate contact; restart from the reset state after each impact.
    solution = solve_ivp(
        continuous_dynamics,
        (start_time, end_time),
        state,
        method="RK45",
        events=[downhill_event, uphill_event],
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        first_step=first_step,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution


def advance_to_impact(state, start_time, end_time, params, **solver_options):
    """Return (final time, final state, contact detected) for return-map callers.

    Without contact, the state belongs to end_time; otherwise it is pre-impact.
    Forward solver_options, such as tolerances, to integrate_to_impact.
    """
    solution = integrate_to_impact(
        state, start_time, end_time, params, **solver_options
    )
    return solution.t[-1], solution.y[:, -1], solution.status == 1


def simulate(
    initial_state,
    params,
    sim_time=20.0,
    *,
    stop_velocity=1e-3,
    required_impacts=6,
    rolling_atol=1e-4,
    rolling_rtol=1e-3,
    rtol=1e-8,
    atol=1e-10,
    max_step=0.05,
    return_details=False,
):
    """Classify one initial state as standing, rolling, or unclassified.

    This is the event-driven counterpart of the experiment in assignment_1.py.
    Return a label by default; return_details=True adds final state, impacts,
    reason, and evaluation count. Angles are in rad; speeds and absolute speed
    tolerances are in rad/s. Rest can terminate early; rolling is checked after
    sim_time using the latest post-impact speeds.
    """
    alpha = np.pi / params["num_spokes"]
    slope = params["slope_angle"]
    lower, upper = slope - alpha, slope + alpha
    # Keep each experiment independent without modifying the caller's initial state.
    state = np.array(initial_state, dtype=float, copy=True)
    if state.shape != (2,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain a finite angle and velocity")
    if not lower - 1e-12 <= state[0] <= upper + 1e-12:
        raise ValueError("Initial angle must lie in the stance interval")
    if sim_time <= 0 or max_step <= 0 or required_impacts < 2:
        raise ValueError("Use positive times and at least two required impacts")
    # Clip only the roundoff-sized excursions allowed by the preceding bounds check.
    state[0] = np.clip(state[0], lower, upper)
    can_stand = abs(slope) < alpha
    current_time = 0.0
    impact_times, impact_velocities = [], []
    function_evaluations = 0

    def finish(label, reason):
        """Package a consistent result for grid statistics and diagnostic checks."""
        details = {
            "classification": label,
            "reason": reason,
            "final_time": current_time,
            "final_state": state.copy(),
            "impact_times": np.array(impact_times),
            "impact_velocities": np.array(impact_velocities),
            "nfev": function_evaluations,
        }
        return details if return_details else label

    # Recognize initial two-contact rest without introducing an artificial impact.
    at_contact = min(abs(state[0] - lower), abs(state[0] - upper)) < 1e-12
    if can_stand and at_contact and state[1] == 0.0:
        return finish("standing", "Initially at rest with two contacts")

    while current_time < sim_time:
        # An outward initial velocity at a boundary requires an immediate reset.
        immediate_impact = (abs(state[0] - upper) < 1e-12 and state[1] > 0) or (
            abs(state[0] - lower) < 1e-12 and state[1] < 0
        )
        if immediate_impact:
            before_impact = state.copy()
        else:
            solution = integrate_to_impact(
                state,
                current_time,
                sim_time,
                params,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            function_evaluations += solution.nfev
            # t contains segment times; each column of y is a [theta, omega] state.
            current_time = float(solution.t[-1])
            before_impact = solution.y[:, -1].copy()
            # Status 1 means contact; reaching the time limit needs no impact reset.
            if solution.status != 1:
                state = before_impact
                break

        # Switch support and consistently record post-impact, not pre-impact, speeds.
        state = model.reset_state(current_time, before_impact, params)
        impact_times.append(current_time)
        impact_velocities.append(state[1])

        # Approximate rest only at contact and with insufficient energy to pass upright.
        # barrier is the required speed squared, with the same units as state[1]**2.
        barrier = 2 * params["gravity"] / params["length"] * (1 - np.cos(state[0]))
        if can_stand and abs(state[1]) < stop_velocity and state[1] ** 2 < barrier:
            state[1] = 0.0
            return finish("standing", "Post-impact speed below rest tolerance")
        # Bound the cost of tiny repeated impacts without labeling a timeout as rest.
        if len(impact_times) >= 10000:
            return finish("unclassified", "Impact limit reached")

    # np.ptp measures the range: test step-to-step repetition, not constant flight speed.
    if len(impact_velocities) >= required_impacts:
        recent = np.array(impact_velocities[-required_impacts:])
        tolerance = rolling_atol + rolling_rtol * abs(np.mean(recent))
        if np.all(recent > 0) and np.ptp(recent) < tolerance:
            return finish("rolling", "Repeated downhill post-impact speeds")
    return finish("unclassified", "No convergence within the time horizon")


def scan_grid(
    params, grid_size=31, velocity_limit=2.0, sim_time=20.0, max_time=80.0, verbose=True
):
    """Simulate every angle-velocity pair; return labels and statistics, not a plot.

    results[row, column] corresponds to velocities[row] and angles[column].
    Retry unresolved states from the same initial condition with longer horizons,
    up to max_time; retain label 2 if unresolved. The returned arrays can be
    saved with model parameters in an NPZ file.
    """
    alpha = np.pi / params["num_spokes"]
    angles = np.linspace(
        params["slope_angle"] - alpha, params["slope_angle"] + alpha, grid_size
    )
    # Preserve the exact unstable upright equilibrium rather than seeding roundoff noise.
    angles[np.abs(angles) < 1e-14] = 0.0
    velocities = np.linspace(-velocity_limit, velocity_limit, grid_size)
    # Untested points must not look like confirmed standing states.
    results = np.full((grid_size, grid_size), 2, dtype=int)
    horizons = np.zeros_like(results, dtype=float)
    impacts = np.zeros_like(results)
    total_evaluations = 0
    started = perf_counter()
    # The nested loops form all initial states; simulate performs one experiment.
    for row, velocity in enumerate(velocities):
        for column, angle in enumerate(angles):
            horizon = sim_time
            while True:
                details = simulate(
                    [angle, velocity], params, horizon, return_details=True
                )
                total_evaluations += details["nfev"]
                if details["classification"] != "unclassified" or horizon >= max_time:
                    break
                # Restart with a longer horizon: 20, 40, then 80 seconds by default.
                horizon = min(2 * horizon, max_time)
            results[row, column] = RESULT_CODES[details["classification"]]
            horizons[row, column] = horizon
            impacts[row, column] = len(details["impact_times"])
        if verbose:
            print(f"Rows completed: {row + 1}/{grid_size}", flush=True)
    return {
        "initial_angles": angles,
        "initial_velocities": velocities,
        "roa_results": results,
        "time_horizons": horizons,
        "impact_counts": impacts,
        "elapsed_seconds": perf_counter() - started,
        "function_evaluations": total_evaluations,
    }


# Keep batch execution separate from the functions imported by other analyses.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=61)
    parser.add_argument("--sim-time", type=float, default=20.0)
    parser.add_argument("--max-time", type=float, default=80.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("rimless_wheel_roa_results_event.npz"),
    )
    parser.add_argument(
        "--baseline", type=Path, help="Optional old NPZ on exactly the same grid"
    )
    args = parser.parse_args()
    if args.grid < 2 or args.sim_time <= 0 or args.max_time < args.sim_time:
        parser.error("grid >= 2 and 0 < sim-time <= max-time are required")
    # Run the grid using the selected limits; solver and model functions stay reusable.
    params = DEFAULT_PARAMS.copy()
    output = scan_grid(
        params, args.grid, sim_time=args.sim_time, max_time=args.max_time
    )
    for name, code in RESULT_CODES.items():
        print(f"{name}: {np.count_nonzero(output['roa_results'] == code)}")
    print(f"Elapsed scan time: {output['elapsed_seconds']:.3f} s")
    # Optional baseline labels are comparable only on an identical grid.
    if args.baseline:
        with np.load(args.baseline) as baseline:
            for key in ("initial_angles", "initial_velocities"):
                if baseline[key].shape != output[key].shape or not np.allclose(
                    baseline[key], output[key]
                ):
                    raise ValueError("Baseline comparison requires identical grids")
            changed = output["roa_results"] != baseline["roa_results"]
            print(
                f"Changed classifications: {np.count_nonzero(changed)}/{changed.size}"
            )
            output["baseline_changed_count"] = np.count_nonzero(changed)

    # Store solver and classification settings so the plotted results are traceable.
    np.savez(
        args.output,
        **output,
        **params,
        method="RK45 with impact events",
        rtol=1e-8,
        atol=1e-10,
        max_step=0.05,
        stop_velocity=1e-3,
        required_impacts=6,
        rolling_atol=1e-4,
        rolling_rtol=1e-3,
        sim_time=args.sim_time,
        max_time=args.max_time,
    )
    print(f"Saved: {args.output}")
