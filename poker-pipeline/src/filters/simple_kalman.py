"""
Simple 1D Kalman filter for tracking scalar state (e.g., bet size).
Assumes constant velocity model.
"""


class SimpleKalmanFilter:
    """
    1D Kalman filter for tracking scalar state (e.g., bet size).
    Assumes constant velocity model.
    """

    def __init__(self, process_variance=1.0, measurement_variance=1.0):
        # Initial state estimate (bet size)
        self.x = 0.0
        # Initial covariance
        self.P = 1.0
        # State transition (constant: x_t = x_{t-1})
        self.F = 1.0
        # Observation matrix
        self.H = 1.0
        # Process noise
        self.Q = process_variance
        # Measurement noise
        self.R = measurement_variance

    def predict(self):
        """Predict state and covariance forward in time."""
        self.x = self.F * self.x
        self.P = self.F * self.P * self.F + self.Q

    def update(self, z):
        """Update state with measurement z."""
        # Measurement residual (innovation)
        y = z - self.H * self.x
        # Innovation covariance
        S = self.H * self.P * self.H + self.R
        # Kalman gain
        K = self.P * self.H / S
        # Update state and covariance
        self.x = self.x + K * y
        self.P = (1 - K * self.H) * self.P

    def get_estimate(self):
        """Return current state estimate."""
        return self.x
