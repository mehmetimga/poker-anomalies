"""
Multivariate Kalman filter for tracking position-velocity state.
State: [position, velocity] for bet trends over time.
"""

import numpy as np
from src.config import KALMAN_INITIAL_COVARIANCE


class KalmanFilter:
    """
    Multivariate Kalman filter for tracking position-velocity state.
    State: [position, velocity] for bet trends over time.
    """

    def __init__(self, dt=1.0, process_variance=1.0, measurement_variance=4.0):
        self.dt = dt
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

        # State transition matrix (position and velocity)
        self.F = np.array([[1, self.dt], [0, 1]])

        # Observation matrix (observe position only, e.g., bet size)
        self.H = np.array([[1, 0]])

        # Process noise covariance (for position-velocity)
        self.Q = (
            np.array([[self.dt**4 / 4, self.dt**3 / 2], [self.dt**3 / 2, self.dt**2]])
            * self.process_variance
        )

        # Measurement noise covariance
        self.R = np.array([[self.measurement_variance]])

        # Initial state [position, velocity]
        self.x = np.zeros((2, 1))

        # Initial covariance
        self.P = np.eye(2) * KALMAN_INITIAL_COVARIANCE

    def predict(self):
        """Predict state and covariance forward."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """Update with measurement z."""
        z = np.array([[z]])
        y = z - np.dot(self.H, self.x)  # Residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Gain
        self.x = self.x + np.dot(K, y)
        I = np.eye(2)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self):
        """Return current state [position, velocity]."""
        return self.x.flatten()
