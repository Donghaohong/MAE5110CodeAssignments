"""Time-stepping methods shared by the course examples and tests."""


def explicit_euler(dynamics, t, state, timestep, params):
    """Advance one step using the derivative at the start of the interval."""
    return state + timestep * dynamics(t, state, params)


def rk4(dynamics, t, state, timestep, params):
    """Advance one step with the classical fourth-order Runge-Kutta method."""
    k1 = dynamics(t, state, params)
    k2 = dynamics(t + timestep / 2, state + timestep * k1 / 2, params)
    k3 = dynamics(t + timestep / 2, state + timestep * k2 / 2, params)
    k4 = dynamics(t + timestep, state + timestep * k3, params)
    return state + timestep * (k1 + 2 * k2 + 2 * k3 + k4) / 6
