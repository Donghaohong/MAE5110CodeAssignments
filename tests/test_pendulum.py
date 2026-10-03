import numpy as np

import integrators
from models import pendulum

def test_total_system_energy():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0

    initial_state = np.array([0.5,0.0])

    potential_energy, kinetic_energy = pendulum.calculate_energy(
        initial_state, params)
    initial_energy = potential_energy + kinetic_energy

    timestep = 1e-4
    state = initial_state.copy()

    for step in range(100):
        time = step * timestep
        state = integrators.explicit_euler(pendulum.dynamics, time,
                                           state, timestep, params)
    
    potential_energy, kinetic_energy = pendulum.calculate_energy(
                state, params)
    final_energy = potential_energy + kinetic_energy

    assert np.isclose(initial_energy, final_energy)

def test_torque():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.0
    params["torque"] = 0.0

    state = np.array([0.0,0.0])

    state_derivative = pendulum.dynamics(0.0, state, params)

    expected_acceleration = (params["torque"] / 
                        (params["mass"] * params["length"] ** 2))

    assert np.isclose(state_derivative[1], expected_acceleration)

def test_damping():
    params = pendulum.generate_params()
    params["damping_coeff"] = 0.1
    params["torque"] = 0.0

    state = np.array([0.0,1.0])

    state_derivative = pendulum.dynamics(0.0, state, params)

    expected_acceleration = ((-params["damping_coeff"] * state[1]) / 
                        (params["mass"] * params["length"] ** 2))

    assert np.isclose(state_derivative[1], expected_acceleration)
