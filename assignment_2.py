# Assignment 2, Anna Schaldenbrand
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import time
from pathlib import Path

from models import inverted_pendulum_walker as model
from integrators_models import rk4 as integrator
print("Assignment 2 script initiated...")

## Functions ##

# 1. Controller Logic Functions: 

def compute_ankle_torque(state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]
    damping = params["damping"]
    stiffness = params["stiffness"]
    lower_bound = params["lower_torque_bound"]
    upper_bound = params["upper_torque_bound"]
    
    theta = state[0]
    theta_dot = state[1]

    gravity_cancelling_torque = -mass*gravity*length*np.sin(theta)
    damping_torque = -damping*theta_dot # derivative control
    restoring_torque = -stiffness*theta # proportional control

    ankle_torque = gravity_cancelling_torque + damping_torque + restoring_torque

    ankle_torque = np.clip(ankle_torque, lower_bound, upper_bound) # return

    return ankle_torque


def simulate_ankle_controller(initial_state, params, timestep, sim_time, theta_tolerance, theta_dot_tolerance, theta_giveup):
    params = dict(params)
    n_timesteps = round(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    for step in range(n_timesteps - 1):
        t = step * timestep
        state = state_traj[:, step]
        params["ankle_torque"] = compute_ankle_torque(state, params)
        state_traj[:, step + 1] = integrator.integrate(t, state, params, timestep, model)

        theta = state_traj[0, step + 1]
        theta_dot = state_traj[1, step + 1]

        clearly_converged = abs(theta) < theta_tolerance and abs(theta_dot) < theta_dot_tolerance
        clearly_diverged = abs(theta) > theta_giveup

        if clearly_converged or clearly_diverged:
            break

    time_traj = time_traj[: step + 2]
    state_traj = state_traj[:, : step + 2]
    return state_traj, time_traj


# 2. ROA Characterization Functions:

def check_converged(state_traj, theta_tolerance, theta_dot_tolerance):
    final_theta = state_traj[0, -1]
    final_theta_dot = state_traj[1, -1]

    converged = abs(final_theta) < theta_tolerance and abs(final_theta_dot) < theta_dot_tolerance

    return converged


def determine_roa(params, timestep, sim_time, theta_range, theta_dot_range, theta_tolerance, theta_dot_tolerance, theta_giveup):
    grid = np.zeros((len(theta_dot_range), len(theta_range)))

    start_time = time.time()
    for i, theta_dot0 in enumerate(theta_dot_range):
        for j, theta0 in enumerate(theta_range):
            initial_state = np.array([theta0, theta_dot0])
            state_traj, _  = simulate_ankle_controller(initial_state, params, timestep, sim_time, theta_tolerance, theta_dot_tolerance, theta_giveup)
            grid[i, j] = check_converged(state_traj, theta_tolerance, theta_dot_tolerance)
    elapsed = time.time() - start_time
    print(f"Controller RoA grid took {elapsed:.1f} seconds")

    return grid


def plot_controller_roa(theta_range, theta_dot_range, grid):
    plt.figure()
    plt.pcolormesh(theta_range, theta_dot_range, grid, vmin=0, vmax=1, shading="auto")
    cbar = plt.colorbar(label="0=does not converge, 1=converges")
    cbar.set_ticks([0, 1])
    plt.xlabel("Angle θ (rad)")
    plt.ylabel("Angular Velocity θ̇ (rad/s)")
    plt.title("Controller Region of Attraction")
    plt.tight_layout()


def reached_roa(state, theta_range, theta_dot_range, roa_grid):
    theta = state[0]
    theta_dot = state[1]

    theta_distances = np.abs(theta_range - theta)
    theta_dot_distances = np.abs(theta_dot_range - theta_dot)

    theta_index = np.argmin(theta_distances) # grabs index of smallest distance -> closest point on roa grid
    theta_dot_index = np.argmin(theta_dot_distances)

    return bool(roa_grid[theta_dot_index, theta_index])

# 3. Main Walking Loop Functions:

def simulate_walker(initial_state, params, timestep, sim_time, theta_range, theta_dot_range, roa_grid):
    params = dict(params)
    n_timesteps = round(sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state
    completed_steps = 0

    for step, t in enumerate(time_traj[:-1]):
        state = state_traj[:, step]

        if reached_roa(state, theta_range, theta_dot_range, roa_grid):
            params["ankle_torque"] = compute_ankle_torque(state, params)
        else:
            params["ankle_torque"] = 0

        next_state = integrator.integrate(t, state, params, timestep, model)

        if model.event_guard(state, next_state, params):
            next_state = model.event_dynamics(next_state, params)
            completed_steps = completed_steps +1

        state_traj[:, step + 1] = next_state
        if completed_steps == desired_number_of_steps:
                    break

    time_traj = time_traj[: step + 2]
    state_traj = state_traj[:, : step + 2]

    return state_traj, time_traj, completed_steps


def simulate_footstep(theta_dot_initial, alpha, params, timestep, max_sim_time=5.0):
    # Simulates one cycle of the Poincare map at theta=0 by starting at theta=0 with theta_dot_initial, 
    # applies footstep control alpha for a single footsetp
    params = dict(params)
    params["angle_of_attack"] = alpha
    params["ankle_torque"] = 0

    n_timesteps = round(max_sim_time / timestep) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = [0.0, theta_dot_initial]
    touched_down = False

    for step in range(n_timesteps - 1):
        t = step * timestep
        state = state_traj[:, step]
        next_state = integrator.integrate(t, state, params, timestep, model)

        if not touched_down and model.event_guard(state, next_state, params):
            next_state = model.event_dynamics(next_state, params)
            touched_down = True

        elif touched_down and state[0] < 0 and next_state[0] >= 0:
            state_traj[:, step + 1] = next_state
            return state_traj[:, : step + 2], time_traj[: step + 2]
        
        elif touched_down and next_state[0] < 0 and next_state[1] <= 0:
            return None  # Check for if it fell back, never reaches theta=0

        state_traj[:, step + 1] = next_state

    raise RuntimeError(
        f"Step neither returned to theta=0 nor clearly fell back within {max_sim_time}s "
        f"(theta_dot_initial={theta_dot_initial}, alpha={alpha}), check timestep/max_sim_time"
    )


def find_nearest_index(value, grid):
    # Find the index of whichever entry in the grid is closest to the value by finding the index of the minimum distance
    return np.argmin(np.abs(grid - value))


def find_speeds_already_in_roa(theta_dot_sweep, theta_range, theta_dot_range, roa_grid):
    # determine which speeds are already in the region of attraction and replace them with 0 steps
    n_theta_dot = len(theta_dot_sweep)  # how many speeds are in grid 
    steps_to_standstill = np.full(n_theta_dot, np.inf) # fill entries with inf by default

    for i, theta_dot_initial in enumerate(theta_dot_sweep): # Iterate through every speed and find the ones in the ROA already

        state = np.array([0.0, theta_dot_initial]) # checking at theta = 0 (Poincare section) and each speed

        if reached_roa(state, theta_range, theta_dot_range, roa_grid): # check if in ROA
            # If yes, this speed needs 0 footsteps
            steps_to_standstill[i] = 0

    return steps_to_standstill


def check_speed_for_better_path(i, theta_dot_sweep, alpha_sweep, next_theta_dot_table, steps_to_standstill, best_alpha):
    # from one speed do any of the alphas result in footsteps or alphas you can take and if so, is that better than
    # what is currently saved for this speed?

    best_steps = steps_to_standstill[i] # only overwrite these if better steps are found
    best_alpha_for_state = best_alpha[i]

    # Try every footstep option, one at a time.
    # j = the index of this alpha in alpha_sweep; alpha = the actual value.
    for j, alpha in enumerate(alpha_sweep):

        theta_dot_next = next_theta_dot_table[i, j] # grab speed at index
    
        if np.isnan(theta_dot_next):
            continue

        next_index = find_nearest_index(theta_dot_next, theta_dot_sweep)
        possible_steps = steps_to_standstill[next_index] + 1

        if possible_steps < best_steps:
            best_steps = possible_steps
            best_alpha_for_state = alpha

    improved = best_steps < steps_to_standstill[i]

    steps_to_standstill[i] = best_steps
    best_alpha[i] = best_alpha_for_state

    return improved   # Can identify whether anything changed.


def find_steps_to_standstill(theta_dot_sweep, alpha_sweep, next_theta_dot_table, theta_range, theta_dot_range, roa_grid):
    # Repeat the step process until there are no new better paths to find
    steps_to_standstill = find_speeds_already_in_roa(theta_dot_sweep, theta_range, theta_dot_range, roa_grid)  # First, figure out which speeds need zero footsteps

    best_alpha = np.full(len(theta_dot_sweep), np.nan) # Start as NaN for every alpha and will later get filled in

    still_improving = True # Initialize while loop

    while still_improving:
  
        still_improving = False # assume no better steps will be found until proven wrong

        for i in range(len(theta_dot_sweep)): # check every speed

            if check_speed_for_better_path(i, theta_dot_sweep, alpha_sweep, next_theta_dot_table, steps_to_standstill, best_alpha):
                # if improved, steps_to_standstill, best_alpha are automatically updated in above function
                still_improving = True

        # If it goes through every state and none of them improved, still_improving is still False so the while loop stops here -> thus, converged.

    return steps_to_standstill, best_alpha


def walk_forward_with_alpha_control(theta_dot_initial, theta_dot_sweep, best_alpha, theta_range, theta_dot_range, roa_grid, params, timestep, max_steps=50):
    # main function for walking forward with alpha actively computed based on the known state space
    theta_dot_current = theta_dot_initial
    state_traj = None
    time_traj = None
    steps_taken = 0

    for step_number in range(max_steps):
        state = np.array([0, theta_dot_current])

        if reached_roa(state, theta_range, theta_dot_range, roa_grid):
            break  # ankle controller should take over

        nearest_index = find_nearest_index(theta_dot_current, theta_dot_sweep)
        theta_dot_current = theta_dot_sweep[nearest_index]
        alpha = best_alpha[nearest_index]
        if np.isnan(alpha):
        # This snapped grid speed was already counted as converged 
            break

        step_state_traj, step_time_traj = simulate_footstep(theta_dot_current, alpha, params, timestep)

        if state_traj is None: # identify first step
            state_traj = step_state_traj
            time_traj = step_time_traj
        else: # otherwise continue building time and state vectors
            step_time_traj = step_time_traj+time_traj[-1]
            state_traj = np.concatenate([state_traj, step_state_traj], axis=1)
            time_traj = np.concatenate([time_traj, step_time_traj])

        theta_dot_current = step_state_traj[1, -1]  # end speed becomes speed of the nextnext step
        steps_taken = steps_taken + 1

    if state_traj is None: # in RoA from start
        state_traj = np.array([[0], [theta_dot_initial]])
        time_traj = np.array([0])

    return state_traj, time_traj, steps_taken


def determine_steps_to_standstill_for_a_resolution(n_theta_dot, n_alpha, params, timestep, theta_range, theta_dot_range, roa_grid):
    # shorter version of walk forward with alpha control function above but specifically for testing resolutions
    theta_dot_sweep = np.linspace(0, np.sqrt(2*params["gravity"]/params["length"]), n_theta_dot)
    alpha_sweep = np.linspace(np.pi/8, np.pi/7, n_alpha)

   # simulate 
    next_theta_dot_table = np.full((n_theta_dot, n_alpha), np.nan)
    for i, theta_dot_initial in enumerate(theta_dot_sweep):
        if reached_roa(np.array([0.0, theta_dot_initial]), theta_range, theta_dot_range, roa_grid):
            continue  
        for j, alpha in enumerate(alpha_sweep):
            result = simulate_footstep(theta_dot_initial, alpha, params, timestep)
            if result is not None:
                footstep_state_traj, _ = result
                next_theta_dot_table[i, j] = footstep_state_traj[1, -1]

    steps_to_standstill, _ = find_steps_to_standstill(theta_dot_sweep, alpha_sweep, next_theta_dot_table,theta_range, theta_dot_range, roa_grid)

    return theta_dot_sweep, steps_to_standstill


## Main Code ##

# Initialize Parameters
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
    "damping": 2, # Kd N*m*s/rad
    "stiffness": 1, # Kp
}
params["lower_torque_bound"] = -0.1 * params["mass"] * params["gravity"] * params["length"]
params["upper_torque_bound"] = 0.05 * params["mass"] * params["gravity"] * params["length"]

timestep = 1e-4
sim_time = 4
desired_number_of_steps = 6


# A. Find and Plot ROA
lower_torque_bound = params["lower_torque_bound"]
upper_torque_bound = params["upper_torque_bound"]
mass = params["mass"]
gravity = params["gravity"]
length = params["length"]
roa_sim_time = 5 # need a bit longer for ROA

theta_lower_bound = np.arcsin(-upper_torque_bound/(mass*gravity*length))
theta_upper_bound = np.arcsin(-lower_torque_bound/(mass*gravity*length))
theta_range = np.linspace(10*theta_lower_bound, 5*theta_upper_bound, 40)
theta_dot_range = np.linspace(10*(-0.1), 10*(0.1), 40) 

grid = determine_roa(params, timestep, roa_sim_time, theta_range, theta_dot_range, theta_tolerance=0.01,  theta_dot_tolerance=0.01, theta_giveup=0.6)
plot_controller_roa(theta_range, theta_dot_range, grid)
Path("output/assignment_2").mkdir(parents=True, exist_ok=True)
with open("output/assignment_2/roa.png", "wb") as f:
    plt.savefig(f)
print("ROA complete, please check output folder for roa.png")


# B. Check grid resolution: 
# test whether the needed # of steps changes as the resolution changes and checking at several different initial speeds
benchmark_theta_dots = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
resolutions_to_test = [16, 17, 18, 19, 20, 21, 22]

steps_at_each_resolution = np.zeros((len(resolutions_to_test), len(benchmark_theta_dots))) # rows=resolutions, columns=benchmark speeds

for i, n in enumerate(resolutions_to_test):
    resolution_theta_dot_sweep, resolution_steps_to_standstill = determine_steps_to_standstill_for_a_resolution(
        n_theta_dot=n, n_alpha=n, params=params, timestep=timestep,
        theta_range=theta_range, theta_dot_range=theta_dot_range, roa_grid=grid
    )

    for j, benchmark_theta_dot in enumerate(benchmark_theta_dots):
        nearest_index = find_nearest_index(benchmark_theta_dot, resolution_theta_dot_sweep)
        steps_at_each_resolution[i, j] = resolution_steps_to_standstill[nearest_index]

# print table used in deliverable
print("\nSteps to Standstill Needed, by Grid Resolution")
header = "resolution".ljust(12)
for benchmark_theta_dot in benchmark_theta_dots:
    header += f"theta_dot={benchmark_theta_dot}".rjust(16)
print(header)

for i, n in enumerate(resolutions_to_test):
    row = f"{n}x{n}".ljust(12)
    for steps in steps_at_each_resolution[i]:
        row += f"{steps}".rjust(16)
    print(row)


# C. Create Lookup Table: 
# for any initial condition, find the sequence of footsteps that brings it into the RoA of the standing controller
theta_dot_bound = np.sqrt(2*params["gravity"]/params["length"]) # given
n_theta_dot = 21  # Determined resolution from above
n_alpha = 21 # Determined resolution from above

theta_dot_sweep = np.linspace(0, theta_dot_bound, n_theta_dot) # only testing positive initial forward angular velocities
alpha_sweep = np.linspace(np.pi/8, np.pi/7, n_alpha)

next_theta_dot_table = np.full((n_theta_dot, n_alpha), np.nan)
for i, theta_dot_initial in enumerate(theta_dot_sweep):
    if reached_roa(np.array([0.0, theta_dot_initial]), theta_range, theta_dot_range, grid):
        continue  # already converged
    for j, alpha in enumerate(alpha_sweep):
        result = simulate_footstep(theta_dot_initial, alpha, params, timestep)
        if result is not None:
            footstep_state_traj, _ = result
            next_theta_dot_table[i, j] = footstep_state_traj[1, -1]

plt.figure()
plt.pcolormesh(alpha_sweep, theta_dot_sweep, next_theta_dot_table, shading="auto")
plt.colorbar(label="theta_dot_next")
plt.xlabel("alpha (rad)")
plt.ylabel("theta_dot_initial (rad/s)")
plt.title("Footstep Lookup Table")
with open("output/assignment_2/lookup_table.png", "wb") as f:
    plt.savefig(f)
print("Lookup Table Complete, please check output folder for lookup_table.png")


# D. Plot number of steps to standstill for range of angular velocities
steps_to_standstill, best_alpha = find_steps_to_standstill(theta_dot_sweep, alpha_sweep, next_theta_dot_table, theta_range, theta_dot_range, grid)
plt.figure()
plt.plot(theta_dot_sweep, steps_to_standstill, "o-")
plt.xlabel("Initial Theta_dot (rad/s)")
plt.ylabel("Steps to Standstill")
plt.title("Steps to Standstill vs. Initial Angular Velocity")
with open("output/assignment_2/steps_to_standstill.png", "wb") as f:
    plt.savefig(f)
print("Steps to Standstill for various, please check output folder for steps_to_standstill.png")


# E. Plot the trajectory for a walker needing at least 3 steps and the max
# Get trajectory of walker needing at least 3 footsteps:
index_needing_3_steps = np.where(steps_to_standstill >= 3)[0][0]  # find every index where steps>=3 is true, take the first one
theta_dot_for_3_steps = theta_dot_sweep[index_needing_3_steps]
state_traj_3_steps, time_traj_3_steps, steps_taken_3 = walk_forward_with_alpha_control(theta_dot_for_3_steps, theta_dot_sweep, best_alpha, theta_range, theta_dot_range, grid, params, timestep)

# Get trajectory of walker needing the maximum footsteps:
max_steps_needed = np.max(steps_to_standstill)
index_needing_max_steps = np.where(steps_to_standstill == max_steps_needed)[0][0]
theta_dot_for_max_steps = theta_dot_sweep[index_needing_max_steps]

state_traj_max_steps, time_traj_max_steps, steps_taken_max = walk_forward_with_alpha_control(theta_dot_for_max_steps, theta_dot_sweep, best_alpha, theta_range, theta_dot_range, grid, params, timestep)

plt.figure()
plt.plot(time_traj_3_steps, state_traj_3_steps[0], color="tab:blue", linestyle="-", label="theta (3 steps)")
plt.plot(time_traj_3_steps, state_traj_3_steps[1], color="tab:blue", linestyle="--", label="theta_dot (3 steps)")
plt.plot(time_traj_max_steps, state_traj_max_steps[0], color="tab:orange", linestyle="-", label="theta (max steps)")
plt.plot(time_traj_max_steps, state_traj_max_steps[1], color="tab:orange", linestyle="--", label="theta_dot (max steps)")
plt.axhline(0, color="gray", linewidth=0.5)
plt.xlabel("Time (s)")
plt.legend()
plt.title(f"Comparison: {steps_taken_3}-step walk (theta_dot={theta_dot_for_3_steps:.2f}) vs. {steps_taken_max}-step walk (theta_dot={theta_dot_for_max_steps:.2f})")
with open("output/assignment_2/trajectory_comparison.png", "wb") as f:
    plt.savefig(f)

# 6. Optional: Animate Sample Walker
initial_state = np.array([0.2, 0.0])
state_traj, time_traj, completed_steps = simulate_walker(initial_state, params, timestep, sim_time, theta_range, theta_dot_range, grid)
print(f"Walker completed {completed_steps} steps")

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")

# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
import shutil
import tempfile

with tempfile.TemporaryDirectory() as tmp_dir:
    temp_gif_path = Path(tmp_dir) / "walker.gif"
    animation.save(temp_gif_path, writer=PillowWriter(fps=fps))
    shutil.copy(temp_gif_path, output / "walker.gif")

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
# plt.show()



print("Assignment 2 script complete!")