import numpy as np


def generate_params():
    """Return the physical parameters used in the Assignment 1 example."""
    return {"gravity": 9.81, "length": 0.5, "mass": 0.2}


def generate_initial_condition():
    """Start the example at mid-stance with forward angular velocity."""
    return np.array([0.0, 3.0])


def dynamics(t, state, params):
    gravity = params["gravity"]
    length = params["length"]
    mass = params["mass"]

    angle = state[0]
    angular_velocity = state[1]

    angular_acceleration = (
     mass * gravity * length * np.sin(angle)
    ) / (mass * length**2)

    state_derivative = np.array([angular_velocity, angular_acceleration])
    return state_derivative
