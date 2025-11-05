"""
Unscented Kalman Filter for non-linear dynamics.
Uses sigma point propagation instead of Jacobians.
"""

import numpy as np
from src.config import UKF_RESET_COVARIANCE


class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for non-linear dynamics.
    Uses sigma point propagation instead of Jacobians.
    """

    def __init__(self, n, alpha=1.0, beta=2.0, kappa=0.0):
        """
        Parameters:
            n: Dimension of state
            alpha: Spread of sigma points (typically 1e-3 to 1)
            beta: Prior knowledge (2 for Gaussian)
            kappa: Scaling parameter (typically 0 or 3-n)
        """
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (n + kappa) - n

        # Weights for mean and covariance
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + self.lambda_)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

        # Initial state and covariance
        self.x = np.zeros((n, 1))
        self.P = np.eye(n)

        self.Q = np.zeros((n, n))  # Process noise
        self.R = np.zeros((1, 1))  # Measurement noise

    def set_covariances(self, Q, R):
        """Set process and measurement noise covariances."""
        self.Q = Q
        self.R = R

    def sigma_points(self):
        """Generate sigma points around current state."""
        scale = self.n + self.lambda_
        if scale <= 0:
            scale = 1e-6  # Avoid zero or negative

        # Ensure P is positive semi-definite
        P_adjusted = self.P + np.eye(self.n) * 1e-9
        try:
            sqrtP = np.linalg.cholesky(scale * P_adjusted)
        except np.linalg.LinAlgError:
            # Fallback to eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(scale * P_adjusted)
            eigenvalues = np.maximum(eigenvalues, 1e-9)
            sqrtP = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        sigmas = np.hstack([self.x, self.x + sqrtP, self.x - sqrtP])
        return sigmas

    def predict(self, f, dt):
        """Predict state using non-linear process model f."""
        sigmas = self.sigma_points()

        # Propagate sigma points through process model
        pred_sigmas = np.zeros_like(sigmas)
        for i in range(sigmas.shape[1]):
            pred_sigmas[:, i : i + 1] = f(sigmas[:, i : i + 1], dt)
            # Clip predicted sigma points to prevent extreme values
            pred_sigmas[:, i : i + 1] = np.clip(pred_sigmas[:, i : i + 1], -1e5, 1e5)

        # Predicted state mean
        self.x = np.sum(self.Wm[None, :] * pred_sigmas, axis=1)[:, None]
        # Clip state to reasonable range
        self.x = np.clip(self.x, -1e5, 1e5)

        # Predicted covariance
        diff = pred_sigmas - self.x
        self.P = np.zeros((self.n, self.n))
        for i, w in enumerate(self.Wc):
            d = diff[:, i]
            # Clip outer product to prevent overflow
            outer = np.outer(d, d)
            outer = np.clip(outer, -1e10, 1e10)
            self.P += w * outer
        self.P += self.Q

        # Ensure P is positive semi-definite and symmetric
        self.P = (self.P + self.P.T) / 2  # Ensure symmetry
        eigenvalues = np.linalg.eigvals(self.P)
        if np.any(eigenvalues < 0):
            # Regularize if any eigenvalues are negative
            self.P += np.eye(self.n) * 1e-3

    def update(self, z, h):
        """Update with measurement z using measurement model h."""
        sigmas = self.sigma_points()

        # Propagate through measurement model
        meas_sigmas = np.zeros((1, sigmas.shape[1]))
        for i in range(sigmas.shape[1]):
            meas_sigmas[0, i] = h(sigmas[:, i : i + 1])[0, 0]

        # Predicted measurement mean
        z_pred = np.sum(self.Wm[None, :] * meas_sigmas, axis=1)[:, None]

        # Innovation
        diff_z = meas_sigmas.T - z_pred

        # Innovation covariance S (with numerical stability)
        diff_z_squared = diff_z.flatten() ** 2
        # Clip to prevent overflow
        diff_z_squared = np.clip(diff_z_squared, 0, 1e10)
        S = np.sum(self.Wc * diff_z_squared) + self.R[0, 0]
        S = max(S, 1e-6)  # Ensure positive definite
        S = np.array([[S]])

        # Cross covariance Pxz
        diff_x = sigmas - self.x
        Pxz = np.zeros((self.n, 1))
        for i, w in enumerate(self.Wc):
            diff_z_val = diff_z[i, 0]
            # Clip to prevent overflow
            if abs(diff_z_val) > 1e5:
                diff_z_val = np.sign(diff_z_val) * 1e5
            Pxz += w * (diff_x[:, i : i + 1] * diff_z_val)

        # Kalman gain K
        try:
            K = Pxz @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback if S is singular
            K = Pxz / S[0, 0]

        # Update state
        z_array = np.array([[z]])
        innovation = z_array - z_pred
        state_update = K @ innovation

        # Clip state update to prevent extreme changes
        state_update = np.clip(state_update, -1e3, 1e3)
        self.x = self.x + state_update

        # Clip state values to reasonable range
        self.x = np.clip(self.x, -1e5, 1e5)

        # Update covariance with numerical stability
        try:
            cov_update = K @ S @ K.T
            # Ensure covariance update doesn't make P negative definite
            self.P = self.P - cov_update
            # Regularize P to ensure positive semi-definite
            self.P = (self.P + self.P.T) / 2  # Ensure symmetry
            eigenvalues = np.linalg.eigvals(self.P)
            if np.any(eigenvalues < 0):
                # Add regularization if any eigenvalues are negative
                self.P += np.eye(self.n) * 1e-3
        except (np.linalg.LinAlgError, OverflowError):
            # Reset covariance if update fails
            self.P = np.eye(self.n) * UKF_RESET_COVARIANCE

        # Return innovation for anomaly detection
        innovation_val = float(innovation[0, 0])
        # Clip innovation to reasonable range
        innovation_val = np.clip(innovation_val, -1e6, 1e6)
        return innovation_val, float(S[0, 0])

    def get_state(self):
        """Return current state."""
        return self.x.flatten()
