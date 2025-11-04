"""
Kalman Filter implementations for poker anomaly detection.
Includes: Simple Kalman, Multivariate Kalman, Extended Kalman Filter (EKF),
and Unscented Kalman Filter (UKF).
"""

import numpy as np
from collections import deque


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
        self.P = np.eye(2) * 1000

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
            self.P = np.eye(self.n) * 10

        # Return innovation for anomaly detection
        innovation_val = float(innovation[0, 0])
        # Clip innovation to reasonable range
        innovation_val = np.clip(innovation_val, -1e6, 1e6)
        return innovation_val, float(S[0, 0])

    def get_state(self):
        """Return current state."""
        return self.x.flatten()


class PokerUKF:
    """
    Poker-specific UKF wrapper for player bet tracking and anomaly detection.
    Tracks bet position and aggression velocity per player.
    """

    def __init__(self, player_id, process_model, measurement_model):
        """
        Parameters:
            player_id: Unique identifier for player
            process_model: Non-linear process function
            measurement_model: Non-linear measurement function
        """
        self.player_id = player_id
        self.process_model = process_model
        self.measurement_model = measurement_model

        # Initialize UKF with 2D state [bet_position, aggression_velocity]
        self.ukf = UnscentedKalmanFilter(n=2, alpha=1.0, beta=2.0, kappa=0.0)
        self.ukf.set_covariances(Q=np.eye(2) * 0.1, R=np.array([[1.0]]))

        # Initial state: position=0, velocity=1
        self.ukf.x = np.array([[0.0], [1.0]])
        self.ukf.P = np.eye(2) * 10

        self.last_ts = None

        # Rolling statistics for adaptive thresholding
        self.residual_history = deque(maxlen=20)
        self.bet_history = deque(maxlen=20)

        # Warm-up period: need at least 5 samples before flagging anomalies
        self.min_samples_for_detection = 5

    def process_event(self, event):
        """
        Process a poker action event and return estimates and residuals.

        Parameters:
            event: Dict with keys: timestamp, action, amount, table_id, player_id

        Returns:
            Dict with: estimate, residual, action, innovation_std
        """
        # Initialize timestamp on first event
        if self.last_ts is None:
            self.last_ts = event["timestamp"]
            return {
                "estimate": 0.0,
                "residual": 0.0,
                "action": event["action"],
                "innovation_std": 1.0,
            }

        # Calculate time delta
        dt = event["timestamp"] - self.last_ts
        dt = max(dt, 0.01)  # Avoid zero dt
        dt = min(dt, 100.0)  # Cap maximum dt to prevent extreme predictions

        # Check for numerical issues before processing
        state = self.ukf.get_state()
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            # Reset filter if state is invalid
            self.ukf.x = np.array([[0.0], [1.0]])
            self.ukf.P = np.eye(2) * 10
            self.last_ts = event["timestamp"]
            return {
                "estimate": 0.0,
                "residual": 0.0,
                "action": event["action"],
                "innovation_std": 1.0,
            }

        # Predict state forward
        try:
            self.ukf.predict(self.process_model, dt)
        except (OverflowError, ValueError, np.linalg.LinAlgError):
            # Reset filter on numerical error
            self.ukf.x = np.array([[0.0], [1.0]])
            self.ukf.P = np.eye(2) * 10
            self.last_ts = event["timestamp"]
            return {
                "estimate": 0.0,
                "residual": 0.0,
                "action": event["action"],
                "innovation_std": 1.0,
            }

        # Get estimate and check for validity
        estimate = float(self.ukf.get_state()[0])
        if np.isnan(estimate) or np.isinf(estimate):
            estimate = 0.0

        result = {
            "estimate": estimate,
            "residual": 0.0,
            "action": event["action"],
            "innovation_std": 1.0,
        }

        # Update on bet/raise actions
        if event["action"] in ["bet", "raise", "call"]:
            amount = float(event["amount"])

            # Update filter with measurement
            try:
                innovation, innovation_var = self.ukf.update(
                    amount, self.measurement_model
                )

                # Check for invalid values
                if np.isnan(innovation) or np.isinf(innovation):
                    innovation = 0.0
                if np.isnan(innovation_var) or innovation_var <= 0:
                    innovation_var = 1.0

                result["estimate"] = float(self.ukf.get_state()[0])
                if np.isnan(result["estimate"]) or np.isinf(result["estimate"]):
                    result["estimate"] = 0.0

                result["residual"] = abs(innovation)
                result["innovation_std"] = np.sqrt(max(innovation_var, 0.01))

                # Track history (only if valid)
                # For extreme values, use a capped version to prevent filter from adapting too quickly
                if not (np.isnan(result["residual"]) or np.isinf(result["residual"])):
                    # Cap residual for history tracking to prevent extreme outliers from skewing stats
                    # But still use actual residual for anomaly detection
                    if len(self.residual_history) > 0:
                        median_residual = np.median(list(self.residual_history))
                        # Cap at 5x median to prevent single outlier from dominating
                        capped_residual = min(result["residual"], median_residual * 5)
                        self.residual_history.append(capped_residual)
                    else:
                        self.residual_history.append(result["residual"])

                    # Track bet amount (cap extreme bets for history to prevent adaptation)
                    if len(self.bet_history) > 0:
                        median_bet = np.median(list(self.bet_history))
                        # Cap at 3x median for history tracking
                        capped_bet = (
                            min(amount, median_bet * 3) if median_bet > 0 else amount
                        )
                        self.bet_history.append(capped_bet)
                    else:
                        self.bet_history.append(amount)
            except (OverflowError, ValueError, np.linalg.LinAlgError):
                # If update fails, use prediction only
                result["residual"] = abs(amount - estimate)
                result["innovation_std"] = 1.0

        # Update timestamp
        self.last_ts = event["timestamp"]

        return result

    def get_adaptive_threshold(self, default_std=2.0, sigma_multiplier=5.0):
        """
        Calculate adaptive threshold based on historical residuals.
        Uses robust statistics (median and IQR) to avoid outliers skewing the threshold.
        Returns 5 * std_dev of residuals (increased from 3σ to 4σ to 5σ to reduce false positives), but with better outlier resistance.

        Parameters:
            default_std: Default standard deviation for early samples
            sigma_multiplier: Multiplier for standard deviation (default: 5.0 for 5σ threshold)
        """
        if len(self.residual_history) < 3:
            return sigma_multiplier * default_std

        residuals = np.array(list(self.residual_history))

        # Use robust statistics: median and IQR instead of mean/std
        # This prevents outliers from inflating the threshold
        median_residual = np.median(residuals)
        q75 = np.percentile(residuals, 75)
        q25 = np.percentile(residuals, 25)
        iqr = q75 - q25

        # Use IQR-based std estimate (more robust than std)
        # IQR ≈ 1.35 * std for normal distributions
        robust_std = iqr / 1.35 if iqr > 0 else np.std(residuals)

        # Also compute traditional std as fallback
        traditional_std = np.std(residuals)

        # Use the smaller of the two to avoid over-inflating threshold
        # But ensure minimum threshold
        std = min(robust_std, traditional_std)

        # Minimum threshold based on typical bet sizes
        # Don't let threshold go below reasonable minimum (e.g., 10% of typical bet)
        avg_bet = np.mean(self.bet_history) if self.bet_history else 20.0
        min_threshold = max(0.5, avg_bet * 0.1)  # At least 10% of average bet

        return max(sigma_multiplier * max(std, 0.5), min_threshold)

    def is_warm_up_complete(self):
        """Check if we have enough samples to start anomaly detection."""
        return len(self.residual_history) >= self.min_samples_for_detection

    def get_absolute_bet_threshold(self):
        """
        Get absolute bet size threshold for detecting unusually large bets.
        Returns threshold based on historical bet sizes.
        """
        if len(self.bet_history) < 3:
            return 50.0  # Default threshold for large bets

        bet_amounts = np.array(list(self.bet_history))
        median_bet = np.median(bet_amounts)
        q75_bet = np.percentile(bet_amounts, 75)

        # Large bet threshold: 3x the 75th percentile or 2x median, whichever is larger
        threshold = max(3 * q75_bet, 2 * median_bet, 50.0)
        return threshold

    def get_statistics(self):
        """Return player statistics for monitoring."""
        return {
            "player_id": self.player_id,
            "state": self.ukf.get_state().tolist(),
            "residual_history": list(self.residual_history),
            "bet_history": list(self.bet_history),
            "avg_bet": np.mean(self.bet_history) if self.bet_history else 0.0,
            "std_residual": (
                np.std(self.residual_history) if self.residual_history else 0.0
            ),
        }
