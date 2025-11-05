"""
Extended Kalman Filter for non-linear dynamics.
Uses Jacobian linearization at each step.
"""

import numpy as np


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for non-linear dynamics.
    Uses Jacobian linearization at each step.
    """

    def __init__(self, x0, P0, Q, R, f, F_jac, h, H_jac):
        """
        Parameters:
            x0: Initial state
            P0: Initial covariance
            Q: Process noise covariance
            R: Measurement noise covariance
            f: Non-linear process function
            F_jac: Jacobian of process function
            h: Non-linear measurement function
            H_jac: Jacobian of measurement function
        """
        self.x = np.array(x0, dtype=float).reshape(-1, 1)
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.f = f
        self.F_jac = F_jac
        self.h = h
        self.H_jac = H_jac

    def predict(self, dt):
        """Predict state using non-linear process model."""
        # Predict state
        self.x = self.f(self.x, dt)
        # Jacobian of process model
        F = self.F_jac(self.x, dt)
        # Predicted covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """Update with measurement z."""
        z = np.array(z).reshape(-1, 1)
        # Measurement function
        h_x = self.h(self.x)
        # Jacobian of measurement model
        H = self.H_jac(self.x)
        # Innovation
        y = z - h_x
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        # Update state
        self.x = self.x + K @ y
        # Update covariance
        I = np.eye(self.x.shape[0])
        self.P = (I - K @ H) @ self.P

    def get_state(self):
        """Return current state."""
        return self.x.flatten()
