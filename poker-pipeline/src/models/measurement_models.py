"""
Measurement models for poker betting dynamics.
These non-linear models capture how bet amounts relate to state.
"""

import numpy as np
from src.config import (
    MEASUREMENT_VELOCITY_SCALE,
    MEASUREMENT_VELOCITY_CLIP_MIN,
    MEASUREMENT_VELOCITY_CLIP_MAX,
)


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

    # Clip velocity to prevent exponential overflow
    velocity_clipped = np.clip(
        x_flat[1] / MEASUREMENT_VELOCITY_SCALE,
        MEASUREMENT_VELOCITY_CLIP_MIN,
        MEASUREMENT_VELOCITY_CLIP_MAX,
    )

    # Measurement is position scaled by exponential velocity factor
    # This captures non-linear relationship between state and observation
    measurement = x_flat[0] * np.exp(velocity_clipped)

    # Clip measurement to prevent extreme values
    measurement = np.clip(measurement, -1e6, 1e6)

    return np.array([[measurement]])


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


# EKF-specific Jacobian functions for extended Kalman filter
def measurement_jacobian(x):
    """
    Jacobian of the measurement model for EKF.

    For measurement_model: pos * exp(vel/scale)

    Jacobian H = [exp(vel/scale), pos * exp(vel/scale) / scale]
    """
    x_flat = x.flatten()
    exp_term = np.exp(x_flat[1] / MEASUREMENT_VELOCITY_SCALE)
    H = np.array(
        [
            [
                exp_term,
                x_flat[0] * exp_term / MEASUREMENT_VELOCITY_SCALE,
            ]
        ]
    )
    return H


def squared_measurement_jacobian(x):
    """
    Jacobian for squared measurement model.
    """
    x_flat = x.flatten()
    H = np.array([[2 * x_flat[0], 0]])
    return H
