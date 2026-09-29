"""Sanity checks for the pendulum's energy and control inputs."""

import numpy as np

from integrators import rk4
from models import pendulum


def test_energy_is_conserved_without_damping_or_torque():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0
    state = pendulum.generate_initial_condition()
    state[0] = np.pi / 4  # Start away from equilibrium.
    timestep = 0.01
    energies = [sum(pendulum.calculate_energy(state, params))]

    for step in range(100):
        state = rk4(pendulum.dynamics, step * timestep, state, timestep, params)
        energies.append(sum(pendulum.calculate_energy(state, params)))

    assert np.all(np.isclose(energies, energies[0], rtol=1e-5, atol=1e-8))


def test_torque_changes_angular_acceleration():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0
    state = pendulum.generate_initial_condition()
    acceleration_without_torque = pendulum.dynamics(0.0, state, params)[1]

    params["torque"] = 0.1
    acceleration_with_torque = pendulum.dynamics(0.0, state, params)[1]
    acceleration_change = acceleration_with_torque - acceleration_without_torque
    expected_change = params["torque"] / (params["mass"] * params["length"] ** 2)

    assert np.isclose(acceleration_change, expected_change)


def test_damping_reduces_total_energy():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.2
    params["torque"] = 0.0
    state = pendulum.generate_initial_condition()
    state[0] = np.pi / 4
    state[1] = 0.5
    timestep = 0.01
    initial_energy = sum(pendulum.calculate_energy(state, params))

    for step in range(100):
        state = rk4(pendulum.dynamics, step * timestep, state, timestep, params)

    final_energy = sum(pendulum.calculate_energy(state, params))
    # Require a clear loss, not just tiny numerical drift from the integrator.
    assert final_energy < 0.95 * initial_energy
