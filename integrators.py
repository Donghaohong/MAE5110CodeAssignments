"""Reusable fixed-step integrators for continuous dynamics models.

dynamics(t, state, params) must return the state derivative. Both functions
share an interface and return the approximate state one timestep later.
Neither function detects impacts or applies resets.
"""


def explicit_euler(dynamics, t, state, timestep, params):
    """Advance one explicit Euler step using the current state derivative."""
    state_derivative = dynamics(t, state, params)
    next_state = state + timestep * state_derivative
    return next_state

def rk4(dynamics, t, state, timestep, params):
    """Advance one classical fourth-order Runge-Kutta step."""
    # k1 through k4 are derivatives, not states; midpoint states are trial values.
    k1 = dynamics(t, state, params)
    k2 = dynamics(t + timestep/2, state + timestep/2 * k1, params)
    k3 = dynamics(t + timestep/2, state + timestep/2 * k2, params)
    k4 = dynamics(t + timestep, state + timestep * k3, params)
    # Weight the derivatives and multiply by the timestep to obtain a state increment.
    next_state = state + timestep/6 * (k1 + 2*k2 + 2*k3 + k4)
    return next_state
