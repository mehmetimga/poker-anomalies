Unscented Kalman Filter (UKF) Code Snippet in Python
The Unscented Kalman Filter improves on the EKF by propagating a set of sigma points through the non-linear functions to better capture mean and covariance without Jacobian approximations. This is useful in poker for modeling complex dynamics like non-linear bet escalation or velocity-dependent aggression, where linearization might fail.
Below is a NumPy-based implementation for a 2D state (e.g., [bet_position, aggression_velocity]). It uses a sinusoidal non-linearity in the process (oscillating velocity) and exponential in measurement (amplifying position by velocity factor).
pythonimport numpy as np

class UnscentedKalmanFilter:
    def __init__(self, n, alpha=1e-3, beta=2.0, kappa=0.0):
        self.n = n  # Dimension of state
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambda_ = alpha**2 * (n + kappa) - n
        
        # Weights for mean and covariance (scalars array)
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
        self.Q = Q
        self.R = R
        
    def sigma_points(self):
        # Generate sigma points
        scale = self.n + self.lambda_
        if scale <= 0:
            scale = 1e-6  # Avoid zero or negative
        sqrtP = np.linalg.cholesky(scale * self.P)
        sigmas = np.hstack([self.x, self.x + sqrtP, self.x - sqrtP])
        return sigmas
    
    def predict(self, f, dt):
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
        sigmas = self.sigma_points()
        
        # Propagate through measurement model
        meas_sigmas = np.zeros((1, sigmas.shape[1]))
        for i in range(sigmas.shape[1]):
            meas_sigmas[0, i] = h(sigmas[:, i:i+1])[0, 0]
        
        # Predicted measurement mean
        z_pred = np.sum(self.Wm[None, :] * meas_sigmas, axis=1)[:, None]
        
        # Innovation
        diff_z = meas_sigmas.T - z_pred  # (2n+1, 1)
        
        # Innovation covariance S (scalar since 1D meas)
        S = np.sum(self.Wc * (diff_z.flatten() ** 2)) + self.R[0, 0]
        S = np.array([[S]])  # (1,1)
        
        # Cross covariance Pxz (n, 1)
        diff_x = sigmas - self.x  # (n, 2n+1)
        Pxz = np.zeros((self.n, 1))
        for i, w in enumerate(self.Wc):
            Pxz += w * (diff_x[:, i:i+1] * diff_z[i, 0])
        
        # Kalman gain K (n, 1)
        K = Pxz @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ (z - z_pred)
        
        # Update covariance
        self.P = self.P - K @ S @ K.T
        
    def get_state(self):
        return self.x.flatten()

# Example: Non-linear dynamics (sinusoidal velocity for oscillating aggression)
# State: [bet_position, aggression_velocity]
def process_model(x, dt):
    x_flat = x.flatten()
    pos = x_flat[0] + x_flat[1] * dt
    vel = x_flat[1] + np.sin(x_flat[0]) * dt  # Non-linear acceleration
    return np.array([pos, vel]).reshape(2, 1)

def measurement_model(x):
    x_flat = x.flatten()
    return (x_flat[0] * np.exp(x_flat[1]/10)).reshape(1, 1)  # Non-linear measurement

# Initialize with alpha=1.0 to avoid negative lambda
n = 2
ukf = UnscentedKalmanFilter(n, alpha=1.0)
ukf.set_covariances(Q=np.eye(2)*0.1, R=np.array([[1.0]]))

# Simulate data
np.random.seed(42)
dt = 1.0
true_pos = [0.0]
true_vel = [1.0]
measurements = []
for i in range(10):
    vel_update = true_vel[-1] + np.sin(true_pos[-1]) * dt
    pos_update = true_pos[-1] + true_vel[-1] * dt
    true_pos.append(pos_update)
    true_vel.append(vel_update)
    meas = pos_update * np.exp(true_vel[-1]/10) + np.random.normal(0, 1)
    measurements.append(meas)

measurements = np.array(measurements).reshape(-1, 1)

# Run UKF with smaller initial P
estimates = []
ukf.x = np.array([[0.0], [1.0]])  # Initial state
ukf.P = np.eye(2) * 10  # Smaller
for t in range(10):
    ukf.predict(process_model, dt)
    ukf.update(measurements[t], measurement_model)
    estimates.append(ukf.get_state()[0])

print("True positions:", np.round(true_pos[1:], 2))
print("Measurements:", np.round(measurements.flatten(), 2))
print("UKF estimates (pos):", np.round(estimates, 2))
Sample Output:
textTrue positions: [ 1.    2.    3.84  6.59  8.7  11.11 14.18 16.27 19.35 21.9 ]
Measurements: [ 1.6   2.27  5.71  9.66 10.84 14.87 19.05 22.9  24.5  30.18]
UKF estimates (pos): [ 0.67  1.26  3.97  7.24  8.89 11.2  14.42 16.78 19.86 22.15]
Integration Tips for Poker Pipeline

Customization: Tune alpha (spread of sigma points, 1e-3 to 1), beta (prior knowledge, 2 for Gaussian), and kappa (scaling, often 0 or 3-n). For multi-player, expand to higher dimensions.
Anomaly Detection: Monitor the normalized innovation squared ((z - z_pred)^T * inv(S) * (z - z_pred))—values > chi-squared threshold signal anomalies like collusive bets.
Extensions: For additive noise, this works; use libraries like filterpy for production. Handles stronger non-linearities than EKF without divergence.
Performance: O(n^3) per step due to cholesky (n=state dim), fine for small n like per-player tracking.

This fits seamlessly after your EKF—try it on poker hand sequences for smoother state estimates! If you need multi-dimensional or library-based versions, let me know.