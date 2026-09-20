import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from models import inverted_pendulum_walker as model

params = model.generate_params()

argument_parser = argparse.ArgumentParser(description="Analyze and control the inverted-pendulum walker.")
argument_parser.add_argument("--no-show", action="store_true", help="Save the animation without opening a Matplotlib window.")
arguments = argument_parser.parse_args()

angle_of_attack_min = np.pi / 8
angle_of_attack_max = np.pi / 7

ankle_torque_min = -0.1 * params["mass"] * params["gravity"] * params["length"]
ankle_torque_max = 0.05 * params["mass"] * params["gravity"] * params["length"]


def compute_ankle_torque(state, params, torque_min, torque_max):
    """Cancel gravity and apply bounded PD feedback about the upright state."""
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    proportional_gain = 4.0
    derivative_gain = 4.0

    desired_angular_acceleration = -proportional_gain * angle - derivative_gain * angular_velocity
    gravity_angular_acceleration = gravity / length * np.sin(angle)
    ankle_torque = mass * length**2 * (desired_angular_acceleration - gravity_angular_acceleration)

    return np.clip(ankle_torque, torque_min, torque_max)


def reaches_upright_equilibrium(initial_state, params, torque_min, torque_max, timestep=1e-3, sim_time=10.0):
    """Return whether the bounded ankle controller stabilizes one stance leg."""
    simulation_params = params.copy()
    state = np.array(initial_state, dtype=float)

    n_timesteps = round(sim_time / timestep)

    for step in range(n_timesteps):
        t = step * timestep

        simulation_params["ankle_torque"] = compute_ankle_torque(state, simulation_params, torque_min, torque_max)

        next_state = state + timestep * model.dynamics(t, state, simulation_params)

        if abs(next_state[0]) >= np.pi / 2:
            return False

        state = next_state

    angle_tolerance = 1e-3
    velocity_tolerance = 1e-3

    return abs(state[0]) < angle_tolerance and abs(state[1]) < velocity_tolerance


def is_inside_standing_roa(state, angle_grid, angular_velocity_grid, roa_grid):
    """Conservatively classify a state using its surrounding RoA grid cells."""
    angle = state[0]
    angular_velocity = state[1]

    if angle < angle_grid[0] or angle > angle_grid[-1]:
        return False

    if angular_velocity < angular_velocity_grid[0] or angular_velocity > angular_velocity_grid[-1]:
        return False

    upper_angle_index = int(np.searchsorted(angle_grid, angle, side="right"))
    upper_velocity_index = int(np.searchsorted(angular_velocity_grid, angular_velocity, side="right"))
    lower_angle_index = max(0, upper_angle_index - 1)
    lower_velocity_index = max(0, upper_velocity_index - 1)
    upper_angle_index = min(len(angle_grid) - 1, upper_angle_index)
    upper_velocity_index = min(len(angular_velocity_grid) - 1, upper_velocity_index)

    surrounding_values = roa_grid[lower_velocity_index : upper_velocity_index + 1, lower_angle_index : upper_angle_index + 1]

    return bool(np.all(surrounding_values))


def entered_standing_roa(previous_state, next_state, angle_grid, angular_velocity_grid, roa_grid):
    was_inside = is_inside_standing_roa(previous_state, angle_grid, angular_velocity_grid, roa_grid)
    is_inside = is_inside_standing_roa(next_state, angle_grid, angular_velocity_grid, roa_grid)

    return not was_inside and is_inside


def crossed_poincare_section(previous_state, next_state):
    """Detect a downhill crossing of the section theta = 0."""
    crossed_zero_angle = previous_state[0] < 0.0 and next_state[0] >= 0.0
    moving_downhill = next_state[1] > 0.0

    return crossed_zero_angle and moving_downhill


def interpolate_state_at_angle(previous_state, next_state, target_angle):
    """Linearly localize a guard crossing between two integration states."""
    angle_change = next_state[0] - previous_state[0]
    crossing_fraction = (target_angle - previous_state[0]) / angle_change
    return previous_state + crossing_fraction * (next_state - previous_state)


def simulate_one_step(initial_angular_velocity, angle_of_attack, params, angle_grid, angular_velocity_grid, roa_grid, timestep=1e-4, maximum_step_time=5.0):
    """Simulate one uncontrolled walking step from the Poincare section."""
    step_params = params.copy()

    step_params["ankle_torque"] = 0.0
    step_params["angle_of_attack"] = angle_of_attack

    if initial_angular_velocity <= 0.0:
        return None, False

    state = np.array([0.0, initial_angular_velocity])

    has_impact_occurred = False
    maximum_timesteps = round(maximum_step_time / timestep)

    for step in range(maximum_timesteps):
        t = step * timestep

        next_state = state + timestep * model.dynamics(t, state, step_params)

        if not has_impact_occurred and model.event_guard(state, next_state, step_params):
            impact_angle = step_params["incline"] + step_params["angle_of_attack"]

            impact_state = interpolate_state_at_angle(state, next_state, impact_angle)

            if entered_standing_roa(state, impact_state, angle_grid, angular_velocity_grid, roa_grid):
                return impact_state, True

            state = model.event_dynamics(impact_state, step_params)

            if is_inside_standing_roa(state, angle_grid, angular_velocity_grid, roa_grid):
                return state, True

            has_impact_occurred = True
            continue

        if has_impact_occurred and crossed_poincare_section(state, next_state):
            poincare_state = interpolate_state_at_angle(state, next_state, 0.0)

            reached_roa = entered_standing_roa(state, poincare_state, angle_grid, angular_velocity_grid, roa_grid)

            return poincare_state, reached_roa

        if entered_standing_roa(state, next_state, angle_grid, angular_velocity_grid, roa_grid):
            return next_state, True

        if has_impact_occurred and next_state[0] < 0.0 and next_state[1] <= 0.0:
            return None, False

        if abs(next_state[0]) >= np.pi / 2:
            return None, False

        state = next_state

    return None, False


def calculate_new_stance_position(impact_state, stance_position, params):
    """Return the new foot position at a plastic swing-foot collision."""
    length = params["length"]
    angle_of_attack = params["angle_of_attack"]
    impact_angle = impact_state[0]
    new_stance_angle = impact_angle - 2.0 * angle_of_attack
    hub_position = stance_position + length * np.array([np.sin(impact_angle), np.cos(impact_angle)])
    return hub_position - length * np.array([np.sin(new_stance_angle), np.cos(new_stance_angle)])


def simulate_hybrid_policy(
    initial_angular_velocity, angle_sequence, params, angle_grid, angular_velocity_grid, roa_grid, torque_min, torque_max, timestep=1e-4, maximum_time=20.0
):
    """Simulate walking actions followed by bounded upright stabilization."""
    if len(angle_sequence) == 0:
        raise ValueError("angle_sequence must contain at least one action.")

    simulation_params = params.copy()
    state = np.array([0.0, initial_angular_velocity])
    stance_position = np.zeros(2)
    action_index = 0
    has_impact_occurred = False
    completed_steps = 0
    mode = "walking"
    status = "time_limit"
    roa_entry_index = None
    settled_timesteps = 0
    required_settled_timesteps = round(0.1 / timestep)
    maximum_timesteps = round(maximum_time / timestep)

    simulation_params["angle_of_attack"] = float(angle_sequence[action_index])
    simulation_params["ankle_torque"] = 0.0

    time_history = [0.0]
    state_history = [state.copy()]
    torque_history = [0.0]
    angle_history = [simulation_params["angle_of_attack"]]
    mode_history = [mode]
    stance_position_history = [stance_position.copy()]
    impact_indices = []

    for step in range(maximum_timesteps):
        t = step * timestep

        if mode == "walking":
            simulation_params["ankle_torque"] = 0.0
        else:
            simulation_params["ankle_torque"] = compute_ankle_torque(state, simulation_params, torque_min, torque_max)

        next_state = state + timestep * model.dynamics(t, state, simulation_params)

        if mode == "walking":
            if not has_impact_occurred and model.event_guard(state, next_state, simulation_params):
                impact_angle = simulation_params["incline"] + simulation_params["angle_of_attack"]
                impact_state = interpolate_state_at_angle(state, next_state, impact_angle)

                if is_inside_standing_roa(impact_state, angle_grid, angular_velocity_grid, roa_grid) and reaches_upright_equilibrium(
                    impact_state, simulation_params, torque_min, torque_max
                ):
                    next_state = impact_state
                    mode = "balancing"
                    roa_entry_index = len(state_history)
                else:
                    stance_position = calculate_new_stance_position(impact_state, stance_position, simulation_params)
                    next_state = model.event_dynamics(impact_state, simulation_params)
                    completed_steps += 1
                    impact_indices.append(len(state_history))
                    has_impact_occurred = True

                    if is_inside_standing_roa(next_state, angle_grid, angular_velocity_grid, roa_grid) and reaches_upright_equilibrium(
                        next_state, simulation_params, torque_min, torque_max
                    ):
                        mode = "balancing"
                        roa_entry_index = len(state_history)

            elif has_impact_occurred and crossed_poincare_section(state, next_state):
                next_state = interpolate_state_at_angle(state, next_state, 0.0)

                if reaches_upright_equilibrium(next_state, simulation_params, torque_min, torque_max):
                    mode = "balancing"
                    roa_entry_index = len(state_history)
                else:
                    action_index += 1

                    if action_index >= len(angle_sequence):
                        status = "action_sequence_exhausted"
                    else:
                        simulation_params["angle_of_attack"] = float(angle_sequence[action_index])
                        has_impact_occurred = False

            elif is_inside_standing_roa(next_state, angle_grid, angular_velocity_grid, roa_grid) and reaches_upright_equilibrium(
                next_state, simulation_params, torque_min, torque_max
            ):
                mode = "balancing"
                roa_entry_index = len(state_history)
            elif has_impact_occurred and next_state[0] < 0.0 and next_state[1] <= 0.0:
                status = "walking_reversal"
            elif abs(next_state[0]) >= np.pi / 2:
                status = "fallen"
        else:
            if abs(next_state[0]) >= np.pi / 2:
                status = "balancing_failed"
            elif abs(next_state[0]) < 1e-3 and abs(next_state[1]) < 1e-3:
                settled_timesteps += 1
            else:
                settled_timesteps = 0

            if settled_timesteps >= required_settled_timesteps:
                status = "stabilized"

        state = next_state
        time_history.append((step + 1) * timestep)
        state_history.append(state.copy())
        torque_history.append(simulation_params["ankle_torque"])
        angle_history.append(simulation_params["angle_of_attack"])
        mode_history.append(mode)
        stance_position_history.append(stance_position.copy())

        if status != "time_limit":
            break

    return {
        "time": np.array(time_history),
        "state": np.array(state_history).T,
        "ankle_torque": np.array(torque_history),
        "angle_of_attack": np.array(angle_history),
        "mode": np.array(mode_history),
        "stance_position": np.array(stance_position_history).T,
        "status": status,
        "completed_steps": completed_steps,
        "roa_entry_index": roa_entry_index,
        "impact_indices": np.array(impact_indices, dtype=int),
    }


def save_walker_animation(hybrid_result, params, output_path, frames_per_second=25, show_animation=True):
    """Render a hybrid trajectory with its moving stance-foot position."""
    time_history = hybrid_result["time"]
    state_history = hybrid_result["state"]
    stance_position_history = hybrid_result["stance_position"]
    frame_times = np.arange(0.0, time_history[-1], 1.0 / frames_per_second)
    frame_indices = np.searchsorted(time_history, frame_times)
    frame_indices = np.unique(np.append(frame_indices, len(time_history) - 1))

    length = params["length"]
    hub_x = stance_position_history[0] + length * np.sin(state_history[0])
    hub_y = stance_position_history[1] + length * np.cos(state_history[0])
    all_x = np.concatenate((stance_position_history[0], hub_x))
    all_y = np.concatenate((stance_position_history[1], hub_y))
    view_limits = (np.min(all_x) - 1.2 * length, np.max(all_x) + 1.2 * length, np.min(all_y) - 0.8 * length, np.max(all_y) + 0.8 * length)
    figure, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    def draw_frame(frame_index):
        frame_params = params.copy()
        frame_params["angle_of_attack"] = hybrid_result["angle_of_attack"][frame_index]
        frame_params["ankle_torque"] = hybrid_result["ankle_torque"][frame_index]
        show_swing_leg = hybrid_result["mode"][frame_index] == "walking"
        model.visualize(
            state_history[:, frame_index],
            frame_params,
            ax=ax,
            show_swing=show_swing_leg,
            stance_position=stance_position_history[:, frame_index],
            view_limits=view_limits,
        )
        ax.set_title(f"Hybrid walker control, t = {time_history[frame_index]:.2f} s")

    animation = FuncAnimation(figure, draw_frame, frames=frame_indices, interval=1000 / frames_per_second, repeat=False)
    animation.save(output_path, writer=PillowWriter(fps=frames_per_second))

    if show_animation:
        plt.show()

    plt.close(figure)


def estimate_standing_roa(angle_grid, angular_velocity_grid, params, torque_min, torque_max):
    """Classify a state grid under the bounded standing controller."""
    roa_grid = np.zeros((len(angular_velocity_grid), len(angle_grid)), dtype=bool)
    total_initial_states = roa_grid.size
    completed_initial_states = 0

    for velocity_index, initial_angular_velocity in enumerate(angular_velocity_grid):
        for angle_index, initial_angle in enumerate(angle_grid):
            initial_state = np.array([initial_angle, initial_angular_velocity])
            roa_grid[velocity_index, angle_index] = reaches_upright_equilibrium(initial_state, params, torque_min, torque_max)
            completed_initial_states += 1

            if completed_initial_states % 100 == 0 or completed_initial_states == total_initial_states:
                print(f"\rTesting initial states: {completed_initial_states}/{total_initial_states}", end="")

    print()
    return roa_grid


def plot_hybrid_trajectory(ax, hybrid_result, angle_grid, angular_velocity_grid, roa_grid, title):
    state_history = hybrid_result["state"]
    roa_entry_index = hybrid_result["roa_entry_index"]
    impact_indices = hybrid_result["impact_indices"]

    ax.contourf(np.rad2deg(angle_grid), angular_velocity_grid, roa_grid.astype(float), levels=[0.5, 1.5], colors=["#d8efd3"], alpha=0.8)
    ax.fill([], [], color="#d8efd3", label="Standing-controller RoA")
    ax.axvline(0.0, color="0.45", linestyle="--", linewidth=1.2, label=r"Poincaré section $\theta=0$")

    if roa_entry_index is None:
        ax.plot(np.rad2deg(state_history[0]), state_history[1], color="#2468a2", linewidth=2.0, label="Walking")
    else:
        ax.plot(np.rad2deg(state_history[0, : roa_entry_index + 1]), state_history[1, : roa_entry_index + 1], color="#2468a2", linewidth=2.0, label="Walking")
        ax.plot(np.rad2deg(state_history[0, roa_entry_index:]), state_history[1, roa_entry_index:], color="#d47616", linewidth=2.0, label="Balancing")
        ax.plot(
            np.rad2deg(state_history[0, roa_entry_index]),
            state_history[1, roa_entry_index],
            marker="*",
            color="#b22222",
            markersize=13,
            linestyle="none",
            label="RoA entry",
        )

    if impact_indices.size > 0:
        ax.plot(
            np.rad2deg(state_history[0, impact_indices]),
            state_history[1, impact_indices],
            marker="x",
            color="black",
            markersize=7,
            linestyle="none",
            label="Post-impact state",
        )

    ax.plot(np.rad2deg(state_history[0, 0]), state_history[1, 0], marker="o", color="#2468a2", markersize=7, linestyle="none", label="Initial state")
    ax.plot(np.rad2deg(state_history[0, -1]), state_history[1, -1], marker="s", color="#d47616", markersize=6, linestyle="none", label="Final state")
    ax.set_title(title)
    ax.set_xlabel(r"Angle $\theta$ (deg)")
    ax.grid(alpha=0.25)


def save_hybrid_trajectory_figure(minimum_result, maximum_result, angle_grid, angular_velocity_grid, roa_grid, output_path):
    """Save the minimum- and maximum-step state-space trajectories."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True, layout="constrained")
    plot_hybrid_trajectory(
        axes[0], minimum_result, angle_grid, angular_velocity_grid, roa_grid, f"Minimum-step policy ({minimum_result['completed_steps']} steps)"
    )
    plot_hybrid_trajectory(
        axes[1], maximum_result, angle_grid, angular_velocity_grid, roa_grid, f"Maximum-step policy ({maximum_result['completed_steps']} steps)"
    )
    axes[0].set_ylabel(r"Angular velocity $\dot{\theta}$ (rad/s)")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_steps_to_stand_figure(poincare_velocity_grid, steps_to_stand, output_path):
    """Save the minimum required step count over the Poincare section."""
    maximum_required_steps = int(np.max(steps_to_stand))
    figure, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    step_colors = plt.get_cmap("viridis", maximum_required_steps + 1)
    ax.step(poincare_velocity_grid, steps_to_stand, where="mid", color="0.35", linewidth=1.5)
    step_points = ax.scatter(
        poincare_velocity_grid, steps_to_stand, c=steps_to_stand, cmap=step_colors, vmin=-0.5, vmax=maximum_required_steps + 0.5, s=42, zorder=3
    )
    colorbar = figure.colorbar(step_points, ax=ax, ticks=np.arange(maximum_required_steps + 1))
    colorbar.set_label("Minimum number of steps")
    ax.set_xlabel(r"Initial angular velocity $\dot{\theta}_k$ (rad/s)")
    ax.set_ylabel("Minimum number of steps")
    ax.set_yticks(np.arange(maximum_required_steps + 1))
    ax.set_xlim(poincare_velocity_grid[0], poincare_velocity_grid[-1])
    ax.set_ylim(-0.25, maximum_required_steps + 0.35)
    ax.set_title("Minimum steps required to reach the standing-controller RoA")
    ax.grid(alpha=0.25)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_standing_roa_figure(angle_grid, angular_velocity_grid, roa_grid, output_path):
    """Save the standing-controller region-of-attraction map."""
    figure, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
    roa_map = ax.contourf(np.rad2deg(angle_grid), angular_velocity_grid, roa_grid.astype(int), levels=[-0.5, 0.5, 1.5], colors=["#eeeeee", "#70b77e"])
    ax.contour(np.rad2deg(angle_grid), angular_velocity_grid, roa_grid.astype(int), levels=[0.5], colors=["#245a32"], linewidths=1.5)
    ax.plot(0.0, 0.0, marker="*", color="#b22222", markersize=14, linestyle="none", label="Upright equilibrium")
    colorbar = figure.colorbar(roa_map, ax=ax, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["Outside RoA", "Inside RoA"])
    ax.set_xlabel(r"Initial angle $\theta_0$ (deg)")
    ax.set_ylabel(r"Initial angular velocity $\dot{\theta}_0$ (rad/s)")
    ax.set_title("Estimated region of attraction of the standing controller")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def build_lookup_policy(
    number_of_velocity_samples,
    number_of_action_samples,
    maximum_angular_velocity,
    angle_of_attack_min,
    angle_of_attack_max,
    params,
    angle_grid,
    angular_velocity_grid,
    roa_grid,
    torque_min,
    torque_max,
    show_progress=False,
):
    poincare_velocity_grid = np.linspace(0.0, maximum_angular_velocity, number_of_velocity_samples)
    angle_of_attack_grid = np.linspace(angle_of_attack_min, angle_of_attack_max, number_of_action_samples)
    zero_step_states = np.zeros(number_of_velocity_samples, dtype=bool)

    for velocity_index, angular_velocity in enumerate(poincare_velocity_grid):
        zero_step_states[velocity_index] = reaches_upright_equilibrium(np.array([0.0, angular_velocity]), params, torque_min, torque_max)

    next_velocity_table = np.full((number_of_action_samples, number_of_velocity_samples), np.nan)
    one_step_reaches_roa = np.zeros((number_of_action_samples, number_of_velocity_samples), dtype=bool)
    total_transitions = next_velocity_table.size
    completed_transitions = 0

    for angle_index, angle_of_attack in enumerate(angle_of_attack_grid):
        for velocity_index, initial_angular_velocity in enumerate(poincare_velocity_grid):
            if not zero_step_states[velocity_index]:
                step_result, reached_roa = simulate_one_step(initial_angular_velocity, angle_of_attack, params, angle_grid, angular_velocity_grid, roa_grid)

                if step_result is not None:
                    verified_reached_roa = reaches_upright_equilibrium(step_result, params, torque_min, torque_max)

                    if verified_reached_roa:
                        one_step_reaches_roa[angle_index, velocity_index] = True
                    elif not reached_roa:
                        next_velocity_table[angle_index, velocity_index] = step_result[1]

            completed_transitions += 1

            if show_progress and (completed_transitions % 50 == 0 or completed_transitions == total_transitions):
                print(f"\rBuilding lookup table: {completed_transitions}/{total_transitions}", end="")

    if show_progress:
        print()

    steps_to_stand = np.full(number_of_velocity_samples, -1, dtype=int)
    best_angle_of_attack = np.full(number_of_velocity_samples, np.nan)
    steps_to_stand[zero_step_states] = 0
    newly_reachable_counts = []

    for required_steps in range(1, number_of_velocity_samples + 1):
        newly_reachable_states = 0

        for velocity_index in range(number_of_velocity_samples):
            if steps_to_stand[velocity_index] >= 0:
                continue

            candidate_actions = []

            for angle_index, angle_of_attack in enumerate(angle_of_attack_grid):
                if one_step_reaches_roa[angle_index, velocity_index]:
                    remaining_steps = 0
                    candidate_velocity = 0.0
                else:
                    next_angular_velocity = next_velocity_table[angle_index, velocity_index]

                    if not np.isfinite(next_angular_velocity):
                        continue

                    next_velocity_index = int(np.argmin(np.abs(poincare_velocity_grid - next_angular_velocity)))
                    remaining_steps = steps_to_stand[next_velocity_index]
                    candidate_velocity = next_angular_velocity

                if remaining_steps == required_steps - 1:
                    candidate_actions.append((candidate_velocity, angle_of_attack))

            if candidate_actions:
                if required_steps == 1:
                    selected_candidate = candidate_actions[len(candidate_actions) // 2]
                else:
                    selected_candidate = min(candidate_actions, key=lambda candidate: candidate[0])

                best_angle_of_attack[velocity_index] = selected_candidate[1]
                steps_to_stand[velocity_index] = required_steps
                newly_reachable_states += 1

        newly_reachable_counts.append(newly_reachable_states)

        if newly_reachable_states == 0:
            break

    return {
        "poincare_velocity_grid": poincare_velocity_grid,
        "angle_of_attack_grid": angle_of_attack_grid,
        "zero_step_states": zero_step_states,
        "next_velocity_table": next_velocity_table,
        "one_step_reaches_roa": one_step_reaches_roa,
        "steps_to_stand": steps_to_stand,
        "best_angle_of_attack": best_angle_of_attack,
        "newly_reachable_counts": newly_reachable_counts,
    }


def validate_lookup_policy(policy, validation_velocity_grid, params, angle_grid, angular_velocity_grid, roa_grid, torque_min, torque_max):
    successful_rollouts = 0
    matching_step_counts = 0
    poincare_velocity_grid = policy["poincare_velocity_grid"]
    steps_to_stand = policy["steps_to_stand"]
    best_angle_of_attack = policy["best_angle_of_attack"]

    for initial_angular_velocity in validation_velocity_grid:
        nearest_initial_index = int(np.argmin(np.abs(poincare_velocity_grid - initial_angular_velocity)))
        predicted_steps = steps_to_stand[nearest_initial_index]
        current_angular_velocity = initial_angular_velocity
        actual_steps = 0
        reached_roa = reaches_upright_equilibrium(np.array([0.0, current_angular_velocity]), params, torque_min, torque_max)

        for _ in range(len(poincare_velocity_grid)):
            if reached_roa or predicted_steps < 0:
                break

            nearest_velocity_index = int(np.argmin(np.abs(poincare_velocity_grid - current_angular_velocity)))
            selected_angle = best_angle_of_attack[nearest_velocity_index]

            if not np.isfinite(selected_angle):
                break

            step_result, estimated_reached_roa = simulate_one_step(
                current_angular_velocity, selected_angle, params, angle_grid, angular_velocity_grid, roa_grid
            )

            if step_result is None:
                break

            actual_steps += 1
            reached_roa = reaches_upright_equilibrium(step_result, params, torque_min, torque_max)

            if estimated_reached_roa and not reached_roa:
                break

            current_angular_velocity = step_result[1]

        if reached_roa:
            successful_rollouts += 1

            if actual_steps == predicted_steps:
                matching_step_counts += 1

    return successful_rollouts, matching_step_counts


def find_longest_lookup_path(initial_velocity_index, policy):
    """Return the longest finite action sequence from one lookup-table state."""
    poincare_velocity_grid = policy["poincare_velocity_grid"]
    angle_of_attack_grid = policy["angle_of_attack_grid"]
    next_velocity_table = policy["next_velocity_table"]
    one_step_reaches_roa = policy["one_step_reaches_roa"]
    memoized_paths = {}
    active_search_states = set()

    def search_from_state(velocity_index):
        if velocity_index in active_search_states:
            raise RuntimeError("The lookup graph contains a reachable cycle, so the maximum walking duration is unbounded on this grid.")

        if velocity_index in memoized_paths:
            return memoized_paths[velocity_index]

        active_search_states.add(velocity_index)
        longest_path = None

        for angle_index in range(len(angle_of_attack_grid)):
            if one_step_reaches_roa[angle_index, velocity_index]:
                candidate_path = [angle_index]
            else:
                next_angular_velocity = next_velocity_table[angle_index, velocity_index]

                if not np.isfinite(next_angular_velocity):
                    continue

                next_velocity_index = int(np.argmin(np.abs(poincare_velocity_grid - next_angular_velocity)))
                remaining_path = search_from_state(next_velocity_index)

                if remaining_path is None:
                    continue

                candidate_path = [angle_index, *remaining_path]

            if longest_path is None or len(candidate_path) > len(longest_path):
                longest_path = candidate_path

        active_search_states.remove(velocity_index)
        memoized_paths[velocity_index] = longest_path
        return longest_path

    longest_action_indices = search_from_state(initial_velocity_index)

    if longest_action_indices is None:
        return []

    return [float(angle_of_attack_grid[index]) for index in longest_action_indices]


timestep = 1e-4

minimum_angle = -np.pi / 2
maximum_angle = np.pi / 2

maximum_angular_velocity = np.sqrt(2.0 * params["gravity"] / params["length"])

angle_grid = np.linspace(
    minimum_angle,
    maximum_angle,
    61,
)

angular_velocity_grid = np.linspace(
    -maximum_angular_velocity,
    maximum_angular_velocity,
    61,
)

roa_grid = estimate_standing_roa(angle_grid, angular_velocity_grid, params, ankle_torque_min, ankle_torque_max)
total_initial_states = roa_grid.size
number_inside_roa = np.count_nonzero(roa_grid)

print(
    "States inside estimated RoA:",
    f"{number_inside_roa}/{total_initial_states}",
)

print(
    "Fraction inside estimated RoA:",
    f"{number_inside_roa / total_initial_states:.2%}",
)

number_of_velocity_samples = 61
number_of_action_samples = 25
lookup_policy = build_lookup_policy(
    number_of_velocity_samples,
    number_of_action_samples,
    maximum_angular_velocity,
    angle_of_attack_min,
    angle_of_attack_max,
    params,
    angle_grid,
    angular_velocity_grid,
    roa_grid,
    ankle_torque_min,
    ankle_torque_max,
    show_progress=True,
)
poincare_velocity_grid = lookup_policy["poincare_velocity_grid"]
angle_of_attack_grid = lookup_policy["angle_of_attack_grid"]
zero_step_states = lookup_policy["zero_step_states"]
next_velocity_table = lookup_policy["next_velocity_table"]
one_step_reaches_roa = lookup_policy["one_step_reaches_roa"]
steps_to_stand = lookup_policy["steps_to_stand"]
best_angle_of_attack = lookup_policy["best_angle_of_attack"]
number_of_poincare_states = len(poincare_velocity_grid)
total_transitions = next_velocity_table.size
valid_transitions = np.count_nonzero(np.isfinite(next_velocity_table))
states_with_one_step_solution = np.count_nonzero(np.any(one_step_reaches_roa, axis=0))

print(f"Valid step transitions: {valid_transitions}/{total_transitions}")
print(f"Poincare states with a one-step solution: {states_with_one_step_solution}/{number_of_poincare_states}")
print(f"Poincare states already inside RoA: {np.count_nonzero(zero_step_states)}/{number_of_poincare_states}")

for required_steps, newly_reachable_states in enumerate(lookup_policy["newly_reachable_counts"], start=1):
    print(f"New states requiring {required_steps} step(s): {newly_reachable_states}")

reachable_states = steps_to_stand >= 0
unreachable_states = steps_to_stand < 0

maximum_required_steps = int(np.max(steps_to_stand[reachable_states]))

print(f"Maximum required steps: {maximum_required_steps}")
print(f"Reachable Poincare states: {np.count_nonzero(reachable_states)}/{number_of_poincare_states}")
print(f"Unreachable Poincare states: {np.count_nonzero(unreachable_states)}/{number_of_poincare_states}")

maximum_step_state_indices = np.where(steps_to_stand == maximum_required_steps)[0]
test_velocity_index = int(maximum_step_state_indices[-1])
test_initial_velocity = float(poincare_velocity_grid[test_velocity_index])

current_velocity = test_initial_velocity
policy_velocity_history = [current_velocity]
policy_angle_history = []
policy_reached_roa = False
maximum_policy_steps = maximum_required_steps + 5

print(f"Testing policy from omega = {test_initial_velocity:.6f} rad/s")
print(f"Predicted minimum steps = {steps_to_stand[test_velocity_index]}")

for policy_step in range(maximum_policy_steps):
    current_state = np.array([0.0, current_velocity])

    if reaches_upright_equilibrium(current_state, params, ankle_torque_min, ankle_torque_max):
        policy_reached_roa = True
        break

    nearest_velocity_index = int(np.argmin(np.abs(poincare_velocity_grid - current_velocity)))
    selected_angle = best_angle_of_attack[nearest_velocity_index]

    if not np.isfinite(selected_angle):
        print("Policy failed: no action is available for the nearest grid state.")
        break

    step_result, reached_roa = simulate_one_step(current_velocity, selected_angle, params, angle_grid, angular_velocity_grid, roa_grid)

    if step_result is None:
        print("Policy failed: the selected action did not complete a step.")
        break

    next_velocity = step_result[1]

    print(f"Policy step {policy_step + 1}: omega = {current_velocity:.6f}, alpha = {np.rad2deg(selected_angle):.3f} deg, next omega = {next_velocity:.6f}")

    policy_angle_history.append(selected_angle)
    policy_velocity_history.append(next_velocity)
    current_velocity = next_velocity

    verified_reached_roa = reaches_upright_equilibrium(step_result, params, ankle_torque_min, ankle_torque_max)

    if reached_roa and not verified_reached_roa:
        print("Policy failed: the estimated RoA produced a false-positive entry.")
        break

    if verified_reached_roa:
        policy_reached_roa = True
        break

print(f"Policy reached RoA: {policy_reached_roa}")
print(f"Actual number of steps: {len(policy_angle_history)}")
print("Velocity history:", np.array(policy_velocity_history))
print("Angle history in degrees:", np.rad2deg(np.array(policy_angle_history)))

maximum_action_sequence = find_longest_lookup_path(test_velocity_index, lookup_policy)
maximum_current_velocity = test_initial_velocity
maximum_policy_velocity_history = [maximum_current_velocity]
maximum_policy_angle_history = []
maximum_policy_reached_roa = False

print(f"Testing maximum-step policy from omega = {maximum_current_velocity:.6f} rad/s")
print(f"Predicted maximum successful steps = {len(maximum_action_sequence)}")

for maximum_policy_step, selected_angle in enumerate(maximum_action_sequence):
    step_result, reached_roa = simulate_one_step(maximum_current_velocity, selected_angle, params, angle_grid, angular_velocity_grid, roa_grid)

    if step_result is None:
        print("Maximum-step policy failed: the planned action did not complete a step.")
        break

    next_velocity = step_result[1]
    verified_reached_roa = reaches_upright_equilibrium(step_result, params, ankle_torque_min, ankle_torque_max)

    if reached_roa and not verified_reached_roa:
        print("Maximum-step policy failed: the estimated RoA produced a false-positive entry.")
        break

    print(
        f"Maximum-policy step {maximum_policy_step + 1}: omega = {maximum_current_velocity:.6f}, alpha = {np.rad2deg(selected_angle):.3f} deg, next omega = {next_velocity:.6f}"
    )

    maximum_policy_angle_history.append(selected_angle)
    maximum_policy_velocity_history.append(next_velocity)
    maximum_current_velocity = next_velocity

    if verified_reached_roa:
        maximum_policy_reached_roa = True
        break

print(f"Maximum-step policy reached RoA: {maximum_policy_reached_roa}")
print(f"Actual maximum number of steps: {len(maximum_policy_angle_history)}")
print("Maximum-policy velocity history:", np.array(maximum_policy_velocity_history))
print("Maximum-policy angles in degrees:", np.rad2deg(np.array(maximum_policy_angle_history)))

minimum_hybrid_result = simulate_hybrid_policy(
    test_initial_velocity, policy_angle_history, params, angle_grid, angular_velocity_grid, roa_grid, ankle_torque_min, ankle_torque_max, timestep=timestep
)
maximum_hybrid_result = simulate_hybrid_policy(
    test_initial_velocity,
    maximum_policy_angle_history,
    params,
    angle_grid,
    angular_velocity_grid,
    roa_grid,
    ankle_torque_min,
    ankle_torque_max,
    timestep=timestep,
)

for policy_name, hybrid_result in (("Minimum-step", minimum_hybrid_result), ("Maximum-step", maximum_hybrid_result)):
    roa_entry_index = hybrid_result["roa_entry_index"]
    maximum_walking_torque = (
        np.max(np.abs(hybrid_result["ankle_torque"][: roa_entry_index + 1])) if roa_entry_index is not None else np.max(np.abs(hybrid_result["ankle_torque"]))
    )
    roa_entry_time = hybrid_result["time"][roa_entry_index] if roa_entry_index is not None else np.nan

    print(f"{policy_name} hybrid simulation status: {hybrid_result['status']}")
    print(f"{policy_name} completed walking steps: {hybrid_result['completed_steps']}")
    print(f"{policy_name} RoA entry time: {roa_entry_time:.6f} s")
    print(f"{policy_name} maximum torque before RoA entry: {maximum_walking_torque:.6e} N m")
    print(f"{policy_name} final state: {hybrid_result['state'][:, -1]}")

output_directory = Path("output/assignment_2")
output_directory.mkdir(parents=True, exist_ok=True)
animation_path = output_directory / "walker.gif"
save_walker_animation(minimum_hybrid_result, params, animation_path, show_animation=not arguments.no_show)
print(f"Saved {animation_path}")
trajectory_figure_path = output_directory / "hybrid_policy_trajectories.png"
save_hybrid_trajectory_figure(minimum_hybrid_result, maximum_hybrid_result, angle_grid, angular_velocity_grid, roa_grid, trajectory_figure_path)
print(f"Saved {trajectory_figure_path}")

steps_figure_path = output_directory / "steps_to_stand.png"
save_steps_to_stand_figure(poincare_velocity_grid, steps_to_stand, steps_figure_path)
print(f"Saved {steps_figure_path}")

roa_figure_path = output_directory / "standing_controller_roa.png"
save_standing_roa_figure(angle_grid, angular_velocity_grid, roa_grid, roa_figure_path)
print(f"Saved {roa_figure_path}")

validation_velocity_grid = np.linspace(0.0, maximum_angular_velocity, 101)
lower_resolution_policy = build_lookup_policy(
    51,
    21,
    maximum_angular_velocity,
    angle_of_attack_min,
    angle_of_attack_max,
    params,
    angle_grid,
    angular_velocity_grid,
    roa_grid,
    ankle_torque_min,
    ankle_torque_max,
)

print("Grid-resolution validation on 101 independent initial velocities:")

for resolution_name, candidate_policy in (("51 x 21", lower_resolution_policy), ("61 x 25", lookup_policy)):
    successful_rollouts, matching_step_counts = validate_lookup_policy(
        candidate_policy, validation_velocity_grid, params, angle_grid, angular_velocity_grid, roa_grid, ankle_torque_min, ankle_torque_max
    )
    success_fraction = successful_rollouts / len(validation_velocity_grid)
    agreement_fraction = matching_step_counts / len(validation_velocity_grid)
    resolution_passes = successful_rollouts == len(validation_velocity_grid) and agreement_fraction >= 0.99
    print(
        f"{resolution_name}: successful rollouts = {successful_rollouts}/101 ({success_fraction:.2%}), step-count agreement = {matching_step_counts}/101 ({agreement_fraction:.2%}), passes = {resolution_passes}"
    )
