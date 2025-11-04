"""
Process and measurement models for poker betting dynamics.
These non-linear models capture bet position evolution and aggression velocity.
"""

import numpy as np


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
    vel = x_flat[1] + np.sin(x_flat[0]) * dt * 0.5

    return np.array([pos, vel]).reshape(2, 1)


def measurement_model(x):
    """
    Non-linear measurement model for observed bet amounts.

    The measurement is the position amplified by velocity factor.
    This models how bet size relates to both position and momentum.

    Parameters:
        x: State vector [position, velocity]

    Returns:
        Predicted measurement (scalar)
    """
    x_flat = x.flatten()

    # Measurement is position scaled by exponential velocity factor
    # This captures non-linear relationship between state and observation
    measurement = x_flat[0] * np.exp(x_flat[1] / 10.0)

    return np.array([[measurement]])


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


def simple_measurement_model(x):
    """
    Simplified linear measurement model for testing.
    Simply observe the position (bet amount).

    Parameters:
        x: State vector [position, velocity]

    Returns:
        Predicted measurement (scalar)
    """
    x_flat = x.flatten()
    return np.array([[x_flat[0]]])


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
    F = np.array([[1, dt], [np.cos(x_flat[0]) * dt * 0.5, 1]])
    return F


def measurement_jacobian(x):
    """
    Jacobian of the measurement model for EKF.

    For measurement_model: pos * exp(vel/10)

    Jacobian H = [exp(vel/10), pos * exp(vel/10) / 10]
    """
    x_flat = x.flatten()
    exp_term = np.exp(x_flat[1] / 10.0)
    H = np.array([[exp_term, x_flat[0] * exp_term / 10.0]])
    return H


# Alternative damped process model (from investigation 3)
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
    vel = x_flat[1] * np.exp(-0.1 * dt)  # Damping factor
    return np.array([pos, vel]).reshape(2, 1)


def damped_process_jacobian(x, dt):
    """
    Jacobian for damped process model.
    """
    F = np.array([[1, dt], [0, np.exp(-0.1 * dt)]])
    return F


def squared_measurement_model(x):
    """
    Squared position measurement (variance proxy).
    From investigation 3 - models bet variance observation.

    Parameters:
        x: State vector [position, velocity]

    Returns:
        Predicted measurement (scalar)
    """
    x_flat = x.flatten()
    return (x_flat[0] ** 2).reshape(1, 1)


def squared_measurement_jacobian(x):
    """
    Jacobian for squared measurement model.
    """
    x_flat = x.flatten()
    H = np.array([[2 * x_flat[0], 0]])
    return H
