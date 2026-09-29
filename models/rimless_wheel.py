"""Rimless-wheel continuous dynamics, contact guard, and instantaneous reset.

State is [theta, omega] in rad and rad/s. Angle is measured from upward vertical,
positive downhill; gamma is the slope and alpha = pi/N is half the spoke spacing.
The autonomous model retains t for compatibility with the integration interface.
"""

import numpy as np

def dynamics(t, state, params):
    """Return the single-stance state derivative [omega, domega/dt].

    A pinned, nonslipping stance spoke gives inverted-pendulum dynamics.
    The caller handles impacts separately from this continuous motion.
    """
    gravity = params["gravity"]
    length = params["length"]

    angle = state[0]
    angular_velocity = state[1]

    # Measuring theta from upward vertical gives a positive sin(theta) term.
    angular_acceleration = (gravity / length) * np.sin(angle);
    state_derivative = np.array([angular_velocity, angular_acceleration])

    return state_derivative

def detect_event(t, state, params):
    """Check for outward contact at a fixed-step simulation endpoint.

    Return a Boolean, not a contact time. The adaptive solver instead uses
    continuous event functions to locate the boundary crossing.
    """
    slope_angle = params["slope_angle"]
    num_spokes = params["num_spokes"]

    alpha = np.pi / num_spokes
    angle = state[0]
    angular_velocity = state[1]

    # Check direction as well as position to avoid retriggering a departing contact.
    downhill_impact = (angle >= alpha + slope_angle) and (angular_velocity > 0)
    uphill_impact = (angle <= slope_angle - alpha) and (angular_velocity < 0)

    return downhill_impact or uphill_impact

def reset_state(t, state, params):
    """Return the post-impact state after the caller has confirmed contact.

    Switching the stance spoke shifts the angle; conservation of angular
    momentum about the new contact determines velocity. The simulation handles
    approximate two-contact rest separately from this collision reset.
    """
    slope_angle = params["slope_angle"]
    num_spokes = params["num_spokes"]

    angular_velocity = state[1]
    alpha = np.pi / num_spokes

    # Downhill contact resets upper to lower boundary; uphill contact reverses this.
    if angular_velocity > 0:
        new_angle = slope_angle - alpha
    else:
        new_angle = slope_angle + alpha
    
    # The plastic collision retains a fraction cos(2 * alpha)**2 of kinetic energy.
    new_angular_velocity = angular_velocity * np.cos(2 * alpha)

    return np.array([new_angle, new_angular_velocity])
