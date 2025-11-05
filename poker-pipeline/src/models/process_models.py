"""
Process models for poker betting dynamics.
These non-linear models capture bet position evolution and aggression velocity.
"""

import numpy as np
from src.config import (
    PROCESS_VELOCITY_OSCILLATION_AMPLITUDE,
    PROCESS_DAMPING_FACTOR,
)


def process_model(x, dt):
    """
    Non-linear process model for bet evolution.
    State: [bet_position, aggression_velocity]

    Models:
    - Position updates with velocity
    - Velocity has non-linear oscillating component (sin) representing varying aggression

    Parameters:
        x: State vector [position, velocity]
        dt: Time delta

    Returns:
        Updated state vector
    """
    x_flat = x.flatten()

    # Position evolves with velocity
    pos = x_flat[0] + x_flat[1] * dt

    # Velocity has oscillating acceleration (models variable aggression)
    # sin component creates non-linear dynamics
    vel = x_flat[1] + np.sin(x_flat[0]) * dt * PROCESS_VELOCITY_OSCILLATION_AMPLITUDE

    return np.array([pos, vel]).reshape(2, 1)


def simple_process_model(x, dt):
    """
    Simplified linear process model for testing.
    State: [bet_position, aggression_velocity]

    Parameters:
        x: State vector [position, velocity]
        dt: Time delta

    Returns:
        Updated state vector
    """
    x_flat = x.flatten()

    # Simple linear evolution
    pos = x_flat[0] + x_flat[1] * dt
    vel = x_flat[1]  # Constant velocity

    return np.array([pos, vel]).reshape(2, 1)


def damped_process_model(x, dt):
    """
    Damped process model with exponential velocity decay.
    Models decreasing aggression over time.

    Parameters:
        x: State vector [position, velocity]
        dt: Time delta

    Returns:
        Updated state vector
    """
    x_flat = x.flatten()
    pos = x_flat[0] + x_flat[1] * dt
    vel = x_flat[1] * np.exp(-PROCESS_DAMPING_FACTOR * dt)  # Damping factor
    return np.array([pos, vel]).reshape(2, 1)


# EKF-specific Jacobian functions for extended Kalman filter
def process_jacobian(x, dt):
    """
    Jacobian of the process model for EKF.

    For process_model: [pos + vel*dt, vel + sin(pos)*dt]

    Jacobian F = [
        [1, dt],
        [cos(pos)*dt, 1]
    ]
    """
    x_flat = x.flatten()
    F = np.array(
        [
            [1, dt],
            [np.cos(x_flat[0]) * dt * PROCESS_VELOCITY_OSCILLATION_AMPLITUDE, 1],
        ]
    )
    return F


def damped_process_jacobian(x, dt):
    """
    Jacobian for damped process model.
    """
    F = np.array([[1, dt], [0, np.exp(-PROCESS_DAMPING_FACTOR * dt)]])
    return F
