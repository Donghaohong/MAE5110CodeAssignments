"""HW1 local stability from small perturbations on both sides of the fixed point.

First save the fixed point with rimless_wheel_return_map.py, then run:
uv run python -m HW1.rimless_wheel_floquet
The multiplier scales a small speed error per step; it is not the wheel's speed.
"""

from pathlib import Path

import numpy as np

from .rimless_wheel_return_map import compute_return


def estimate_floquet(fixed_point, epsilon, params, max_time=20.0):
    """Estimate P'(omega*) by central differences; return a dimensionless multiplier.

    fixed_point and epsilon are in rad/s. Both trials use the same post-impact
    angle and must complete a forward step for the difference to be meaningful.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    # Simulate one step per side; the input speeds are separated by 2 * epsilon.
    return_minus = compute_return(fixed_point - epsilon, params, max_time)
    return_plus = compute_return(fixed_point + epsilon, params, max_time)
    if not np.all(np.isfinite([return_minus, return_plus])):
        raise ValueError("Both perturbed states must complete a forward step")
    return (return_plus - return_minus) / (2 * epsilon)


# Parameter sweeps import estimate_floquet without running this standalone experiment.
if __name__ == "__main__":
    # Use the unrounded root and matching parameters to center on the correct cycle.
    path = Path(__file__).with_name("rimless_wheel_return_map_results.npz")
    with np.load(path) as data:
        if data["fixed_points"].size != 1:
            raise ValueError("This script expects one saved rolling fixed point")
        fixed_point = float(data["fixed_points"][0])
        params = {
            key: data[key].item()
            for key in ("gravity", "length", "mass", "slope_angle", "num_spokes")
        }
        max_time = float(data["max_time"])

    epsilon = 1e-4
    multiplier = estimate_floquet(fixed_point, epsilon, params, max_time)
    print(f"Fixed point: {fixed_point:.9f} rad/s")
    print(f"epsilon: {epsilon:.1e} rad/s")
    print(f"Floquet multiplier: {multiplier:.9f}")
