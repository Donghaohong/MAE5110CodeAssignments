import numpy as np

def dynamics(t, state, params):
    gravity = params["gravity"]

    vertical_velocity = state[1]

    vertical_acceleration = -gravity

    state_derivative = np.array([vertical_velocity, vertical_acceleration])
    return state_derivative


def handle_impact(state, params):
    restitution = params["restitution"]

    height = state[0]
    vertical_velocity = state[1]

    if height <= 0 and vertical_velocity < 0:
        height = 0
        vertical_velocity = -restitution * vertical_velocity

    next_state = np.array([height, vertical_velocity])
    return next_state

def calculate_energy(state, params):
    gravity = params["gravity"]
    mass = params["mass"]

    height = state[0]
    vertical_velocity = state[1]

    kinetic_energy = 0.5 * mass * (vertical_velocity) ** 2
    potential_energy = mass * gravity * height
    return potential_energy, kinetic_energy
    
