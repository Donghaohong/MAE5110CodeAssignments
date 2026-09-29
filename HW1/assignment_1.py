"""HW1 single-state experiment using fixed-step rimless-wheel simulation.

Run: uv run python -m HW1.assignment_1
Change initial_state to test another state; rimless_wheel_roa.py runs grid scans.
This baseline checks contact at step endpoints, without locating impacts within a step.
"""

import numpy as np
import matplotlib.pyplot as plt

from models import rimless_wheel as model
from integrators import explicit_euler as integrator

# Use SI units for gravity, length, and mass; convert the slope to radians.
params = {
    "gravity": 9.81,
    "length": 1.0,
    "mass": 0.2,
    "slope_angle": np.deg2rad(5.0),
    "num_spokes": 8,
}

# alpha is half the spoke spacing; the stance interval is gamma +/- alpha.
alpha = np.pi / params["num_spokes"]

# State is [theta, omega] in rad and rad/s; start at rest on a contact boundary.
initial_state = np.array([params["slope_angle"] - alpha, 0.0])

# Approximate rest using a post-impact speed tolerance, not exact numerical zero.
stop_velocity = 1e-3
has_stopped = False

can_stand = abs(params["slope_angle"]) < alpha

impact_velocities = []

has_converged_to_rolling = False

# Require several similar downhill post-impact speeds before labeling rolling.
required_impacts = 6
rolling_atol = 1e-4  
rolling_rtol = 1e-3  

timestep = 1e-5
sim_time = 20.0

n_timesteps = int(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep

# Each column is a state at one time: angle in row 0, angular velocity in row 1.
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state

# The final time stores a result; do not integrate beyond it.
for step, t in enumerate(time_traj[:-1]):
    next_state = integrator(model.dynamics, t, state_traj[:, step], timestep, params) 

    # Check the endpoint for contact, then switch support and reset velocity.
    if model.detect_event(t, next_state, params):
        next_state = model.reset_state(t, next_state, params)

        impact_velocities.append(next_state[1])

        if can_stand and abs(next_state[1]) < stop_velocity:
            next_state[1] = 0.0
            has_stopped = True

            # Fill the remaining columns with the resting state, not unused zeros.
            state_traj[:, step + 1:] = next_state[:, None]
            break

        has_converged_to_rolling = False

        if len(impact_velocities) >= required_impacts:
            recent_velocities = np.array(impact_velocities[-required_impacts:])

            all_downhill = np.all(recent_velocities > 0)

            velocity_spread = (
                np.max(recent_velocities)
                - np.min(recent_velocities)
            )

            # Combine a low-speed absolute tolerance with a speed-scaled tolerance.
            tolerance = (
                rolling_atol
                + rolling_rtol * abs(np.mean(recent_velocities))
            )

            has_converged_to_rolling = (
                all_downhill and velocity_spread < tolerance
            )


    state_traj[:, step + 1] = next_state

# Failing both criteria means unresolved behavior, not a third stable attractor.
if has_stopped:
    print("Approximately standing still.")
elif has_converged_to_rolling:
    print("Approximately steady downhill rolling.")
else:
    print("Not yet classified: try a longer simulation.")

print(f"Number of impacts: {len(impact_velocities)}")

print(
    "Last 6 post-impact velocities:",
    np.array(impact_velocities[-6:])
)
