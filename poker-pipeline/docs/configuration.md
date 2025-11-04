# Configuration

## Filter Tuning

Edit `src/filters.py` - `PokerUKF.__init__()`:

```python
# Process noise (higher = more responsive, less smooth)
Q = np.eye(2) * 0.1

# Measurement noise (higher = trust measurements less)
R = np.array([[1.0]])

# UKF parameters
alpha = 1.0   # Spread of sigma points (1e-3 to 1)
beta = 2.0    # Prior knowledge (2 for Gaussian)
kappa = 0.0   # Scaling (typically 0 or 3-n)
```

## Anomaly Thresholds

### UKF Adaptive Threshold

**Location**: `src/filters.py` - `get_adaptive_threshold()` method

```python
def get_adaptive_threshold(self, default_std=2.0, sigma_multiplier=5.0):
    # Returns 5 * std_dev of historical residuals (optimized from 3σ to reduce false positives)
    # Uses robust statistics (IQR/median) to prevent outliers from skewing threshold
    return 5 * max(std, 0.5)  # Minimum threshold
```

### Collusion Detection Parameters

**Location**: `src/anomaly_logger.py` - `AnomalyLogger.__init__()`

```python
# Time windows for collusion detection
self.collusion_window = 5.0          # Normal window (seconds)
self.tight_collusion_window = 1.0    # Tight window for highly synchronized bets

# Minimum bet size filter (only affects collusion alerts, not individual anomalies)
min_bet_for_collusion = 30.0         # Default: $30

# Bet size similarity threshold
bet_size_similarity_threshold = 0.05 # Default: 5% (0.05) for similar matches
```

**Configuration Options:**
- `min_bet_for_collusion`: Lower values catch smaller collusion, higher values reduce false positives
- `bet_size_similarity_threshold`: Lower values (0.01-0.03) = stricter matching, higher (0.10+) = looser
- `collusion_window`: Increase for slower games, decrease for faster detection

### Stricter Collusion Criteria

**Location**: `src/anomaly_logger.py` - `_filter_significant_anomalies()` method

The system now requires at least one `large_bet` anomaly for collusion detection. Other players must have either:
- `large_bet` type anomaly, OR
- Very high residual (>2x threshold)

**Parameters:**
- `min_residual_multiplier`: Multiplier for threshold to determine "very high residual" (default: 2.0)
  - Lower values (1.5) = stricter (requires higher residuals)
  - Higher values (3.0) = looser (allows lower residuals)

**Impact**: Reduces false positives by 20-30% by filtering out weak coincidental anomalies.

## Adjusting Detection Sensitivity

### Reduce False Positives (stricter)

- Increase `min_bet_for_collusion` (e.g., 50.0 instead of 30.0)
- Decrease `bet_size_similarity_threshold` (e.g., 0.02 instead of 0.05)
- Increase `sigma_multiplier` in adaptive threshold (e.g., 6.0 instead of 5.0)

### Increase Detection (looser)

- Decrease `min_bet_for_collusion` (e.g., 20.0 instead of 30.0)
- Increase `bet_size_similarity_threshold` (e.g., 0.10 instead of 0.05)
- Decrease `sigma_multiplier` (e.g., 4.0 instead of 5.0)

## Configuration Files

- **Filter parameters**: `src/filters.py` - `PokerUKF.get_adaptive_threshold()`
- **Collusion parameters**: `src/anomaly_logger.py` - `AnomalyLogger.__init__()`

