Extended Kalman Filter (EKF) Code Snippet in Python
The Extended Kalman Filter extends the linear Kalman filter to handle non-linear dynamics in the process or measurement models, common in poker where bet patterns might follow non-linear growth (e.g., exponential pot increases) or observations are transformed (e.g., variance in bet sizes squared for risk modeling). It linearizes the models using Jacobians at each step.
Below is a Python implementation using NumPy. This example tracks a player's "position" (e.g., cumulative bet size) and velocity (e.g., aggression rate), with a non-linear process (damped velocity) and measurement (squared position to simulate variance observation). It's a 2D state vector for simplicity.
pythonimport numpy as np

class ExtendedKalmanFilter:
    def __init__(self, x0, P0, Q, R, f, F_jac, h, H_jac):
        self.x = np.array(x0, dtype=float).reshape(-1, 1)
        self.P = np.array(P0, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.f = f
        self.F_jac = F_jac
        self.h = h
        self.H_jac = H_jac
        
    def predict(self, dt):
        # Predict state
        self.x = self.f(self.x, dt)
        # Jacobian of process model
        F = self.F_jac(self.x, dt)
        # Predicted covariance
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, z):
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
        return self.x.flatten()

# Example: Non-linear dynamics (e.g., damped aggression in bet sizing over hands)
# State: [bet_position, aggression_velocity]
# Process: pos' = pos + vel * dt, vel' = vel * exp(-k * dt) (damping)
def process_model(x, dt):
    x_flat = x.flatten()
    pos = x_flat[0] + x_flat[1] * dt
    vel = x_flat[1] * np.exp(-0.1 * dt)
    return np.array([pos, vel]).reshape(2, 1)

def process_jac(x, dt):
    # Jacobian F
    F = np.array([[1, dt],
                  [0, np.exp(-0.1 * dt)]])
    return F

# Measurement: observe squared position (e.g., bet variance proxy) with noise
def measurement_model(x):
    x_flat = x.flatten()
    return (x_flat[0]**2).reshape(1, 1)

def measurement_jac(x):
    x_flat = x.flatten()
    H = np.array([[2 * x_flat[0], 0]])
    return H

# Initialize
x0 = [0.0, 1.0]  # initial bet_pos=0, vel=1
P0 = np.eye(2) * 100
Q = np.eye(2) * 0.1
R = np.array([[1.0]])

ekf = ExtendedKalmanFilter(x0, P0, Q, R, process_model, process_jac, measurement_model, measurement_jac)

# Simulate measurements: true pos grows linearly, but observe pos^2 + noise
np.random.seed(42)
dt = 1.0
true_pos = [0.0]
measurements = []
for i in range(10):
    true_pos.append(true_pos[-1] + 1.0 * dt)
    meas = (true_pos[-1]**2) + np.random.normal(0, 1)
    measurements.append(meas)

measurements = np.array(measurements).reshape(-1, 1)

# Run EKF
estimates = []
for t in range(10):
    ekf.predict(dt)
    ekf.update(measurements[t])
    estimates.append(ekf.get_state()[0])  # bet_pos estimate

print("True positions:", np.round(true_pos[1:], 2))
print("Measurements (pos^2 + noise):", np.round(measurements.flatten(), 2))
print("EKF estimates (pos):", np.round(estimates, 2))
Sample Output:
textTrue positions: [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
Measurements (pos^2 + noise): [  1.5    3.86   9.65  17.52  24.77  35.77  50.58  64.77  80.53 100.54]
EKF estimates (pos): [ 1.25  1.99  3.12  4.18  4.98  5.98  7.11  8.05  8.97 10.03]
Integration Tips for Poker Pipeline

Customization: Replace Jacobians with analytical derivatives of your poker model (e.g., non-linear bluff probability). Tune Q and R based on hand history variance.
Anomaly Detection: Use the innovation y from update—large |y| / sqrt(S) indicates potential collusion (e.g., unnatural bet jumps).
Extensions: For multi-player, expand state to include joint vectors; use libraries like filterpy for more features, but this is standalone.
Performance: Similar to Kalman (O(n^2) per step for n-state), efficient for real-time hand processing.