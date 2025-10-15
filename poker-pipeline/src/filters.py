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
        self.F = np.array([[1, self.dt],
                           [0, 1]])
        
        # Observation matrix (observe position only, e.g., bet size)
        self.H = np.array([[1, 0]])
        
        # Process noise covariance (for position-velocity)
        self.Q = np.array([[self.dt**4/4, self.dt**3/2],
                           [self.dt**3/2, self.dt**2]]) * self.process_variance
        
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
            pred_sigmas[:, i:i+1] = f(sigmas[:, i:i+1], dt)
        
        # Predicted state mean
        self.x = np.sum(self.Wm[None, :] * pred_sigmas, axis=1)[:, None]
        
        # Predicted covariance
        diff = pred_sigmas - self.x
        self.P = np.zeros((self.n, self.n))
        for i, w in enumerate(self.Wc):
            d = diff[:, i]
            self.P += w * np.outer(d, d)
        self.P += self.Q
        
    def update(self, z, h):
        """Update with measurement z using measurement model h."""
        sigmas = self.sigma_points()
        
        # Propagate through measurement model
        meas_sigmas = np.zeros((1, sigmas.shape[1]))
        for i in range(sigmas.shape[1]):
            meas_sigmas[0, i] = h(sigmas[:, i:i+1])[0, 0]
        
        # Predicted measurement mean
        z_pred = np.sum(self.Wm[None, :] * meas_sigmas, axis=1)[:, None]
        
        # Innovation
        diff_z = meas_sigmas.T - z_pred
        
        # Innovation covariance S
        S = np.sum(self.Wc * (diff_z.flatten() ** 2)) + self.R[0, 0]
        S = np.array([[S]])
        
        # Cross covariance Pxz
        diff_x = sigmas - self.x
        Pxz = np.zeros((self.n, 1))
        for i, w in enumerate(self.Wc):
            Pxz += w * (diff_x[:, i:i+1] * diff_z[i, 0])
        
        # Kalman gain K
        K = Pxz @ np.linalg.inv(S)
        
        # Update state
        z_array = np.array([[z]])
        self.x = self.x + K @ (z_array - z_pred)
        
        # Update covariance
        self.P = self.P - K @ S @ K.T
        
        # Return innovation for anomaly detection
        return float((z_array - z_pred)[0, 0]), float(S[0, 0])
        
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
        self.ukf.set_covariances(Q=np.eye(2)*0.1, R=np.array([[1.0]]))
        
        # Initial state: position=0, velocity=1
        self.ukf.x = np.array([[0.0], [1.0]])
        self.ukf.P = np.eye(2) * 10
        
        self.last_ts = None
        
        # Rolling statistics for adaptive thresholding
        self.residual_history = deque(maxlen=20)
        self.bet_history = deque(maxlen=20)
        
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
            self.last_ts = event['timestamp']
            return {
                'estimate': 0.0,
                'residual': 0.0,
                'action': event['action'],
                'innovation_std': 1.0
            }
        
        # Calculate time delta
        dt = event['timestamp'] - self.last_ts
        dt = max(dt, 0.01)  # Avoid zero dt
        
        # Predict state forward
        self.ukf.predict(self.process_model, dt)
        
        result = {
            'estimate': float(self.ukf.get_state()[0]),
            'residual': 0.0,
            'action': event['action'],
            'innovation_std': 1.0
        }
        
        # Update on bet/raise actions
        if event['action'] in ['bet', 'raise', 'call']:
            amount = float(event['amount'])
            
            # Update filter with measurement
            innovation, innovation_var = self.ukf.update(amount, self.measurement_model)
            
            result['estimate'] = float(self.ukf.get_state()[0])
            result['residual'] = abs(innovation)
            result['innovation_std'] = np.sqrt(max(innovation_var, 0.01))
            
            # Track history
            self.residual_history.append(result['residual'])
            self.bet_history.append(amount)
        
        # Update timestamp
        self.last_ts = event['timestamp']
        
        return result
    
    def get_adaptive_threshold(self, default_std=2.0):
        """
        Calculate adaptive threshold based on historical residuals.
        Returns 3 * std_dev of residuals.
        """
        if len(self.residual_history) < 3:
            return 3 * default_std
        
        residuals = np.array(list(self.residual_history))
        std = np.std(residuals)
        return 3 * max(std, 0.5)  # Minimum threshold
    
    def get_statistics(self):
        """Return player statistics for monitoring."""
        return {
            'player_id': self.player_id,
            'state': self.ukf.get_state().tolist(),
            'residual_history': list(self.residual_history),
            'bet_history': list(self.bet_history),
            'avg_bet': np.mean(self.bet_history) if self.bet_history else 0.0,
            'std_residual': np.std(self.residual_history) if self.residual_history else 0.0
        }


