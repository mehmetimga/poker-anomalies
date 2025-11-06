# Configuration

All configuration parameters are centralized in `src/config.py` for easy tuning.

## Filter Tuning

### UKF Parameters

**Location**: `src/config.py`

```python
# UKF parameters
UKF_STATE_DIMENSION = 2  # State dimension [position, velocity]
UKF_ALPHA = 1.0          # Spread of sigma points (typically 1e-3 to 1)
UKF_BETA = 2.0           # Prior knowledge (2 for Gaussian)
UKF_KAPPA = 0.0          # Scaling parameter (typically 0 or 3-n)

# Process noise (higher = more responsive, less smooth)
PROCESS_NOISE_Q = 0.1

# Measurement noise (higher = trust measurements less)
MEASUREMENT_NOISE_R = 1.0

# Initial state
INITIAL_POSITION = 0.0
INITIAL_VELOCITY = 1.0
INITIAL_COVARIANCE = 10.0  # Initial covariance multiplier
```

## Anomaly Thresholds

### UKF Adaptive Threshold

**Location**: `src/config.py`

```python
# Adaptive threshold calculation
DEFAULT_STD = 1.5                    # Default standard deviation for early samples
SIGMA_MULTIPLIER = 3.5               # Multiplier for standard deviation (tuned 3.5σ threshold)
MIN_THRESHOLD_BASE = 0.5             # Minimum threshold base value
MIN_THRESHOLD_PCT = 0.1              # Minimum threshold as percentage of average bet (10%)
DEFAULT_AVG_BET = 20.0               # Default average bet for threshold calculation
IQR_TO_STD_RATIO = 1.35              # IQR ≈ 1.35 * std for normal distributions

# History tracking
RESIDUAL_HISTORY_MAXLEN = 20         # Maximum number of residuals to track per player
BET_HISTORY_MAXLEN = 20              # Maximum number of bets to track per player

# History tracking caps (to prevent outliers from skewing stats)
RESIDUAL_CAP_MULTIPLIER = 5.0        # Cap residual at 5x median for history tracking
BET_CAP_MULTIPLIER = 3.0             # Cap bet at 3x median for history tracking

# Warm-up period
MIN_SAMPLES_FOR_DETECTION = 5        # Minimum samples needed before flagging anomalies
MIN_SAMPLES_FOR_THRESHOLD = 3        # Minimum samples needed for threshold calculation
```

**How it works:**
- Returns `3.5 × σ` of historical residuals (optimized from 3σ to reduce false positives by 27%)
- Uses robust statistics (IQR/median) to prevent outliers from skewing threshold
- Requires 5 samples warm-up period before flagging anomalies

### Absolute Bet Detection

**Location**: `src/config.py`

```python
# Absolute bet threshold
ABSOLUTE_BET_THRESHOLD_DEFAULT = 40.0     # Default threshold for large bets ($40)
ABSOLUTE_BET_Q75_MULTIPLIER = 2.0         # Multiplier for 75th percentile
ABSOLUTE_BET_MEDIAN_MULTIPLIER = 1.5      # Multiplier for median
ABSOLUTE_BET_CAP_MULTIPLIER = 1.25        # Cap threshold at 125% of recent 90th percentile
MIN_SAMPLES_FOR_ABSOLUTE_BET = 3          # Minimum samples needed for absolute bet threshold

# Actions to check for absolute bet anomalies
ABSOLUTE_BET_ACTIONS = ["bet", "raise"]
```

**How it works:**
- Dynamic threshold: max(default $40, 2× 75th percentile, 1.5× median)
- Capped by recent 90th percentile + buffer to prevent over-sensitivity
- Catches coordinated large bets that might not trigger residual thresholds

### Collusion Detection Parameters

**Location**: `src/config.py`

```python
# Collusion detection thresholds
MIN_BET_FOR_COLLUSION = 20.0              # Minimum bet size (dollars) to trigger collusion alerts
BET_SIZE_SIMILARITY_THRESHOLD = 0.08      # Bet amounts must be within 8% to be considered matching
COLLUSION_WINDOW = 6.0                    # Time window (seconds) for detecting collusion patterns
TIGHT_COLLUSION_WINDOW = 1.0              # Tight window for very synchronized bets (seconds)
SUSPICIOUS_SEQUENCE_TIME_DIFF = 2.0       # Max time difference for suspicious bet-raise sequences (seconds)

# Anomaly tracking history
RECENT_ANOMALIES_MAXLEN = 10              # Maximum number of recent anomalies to track per table
RECENT_ACTIONS_MAXLEN = 10                # Maximum number of recent actions to track per table

# Significant anomaly filtering
MIN_RESIDUAL_MULTIPLIER = 1.5             # Multiplier for threshold to determine "very high residual"

# Collusion detector
COLLUSION_DETECTOR_WINDOW_SIZE = 10       # Number of recent hands to analyze
SUSPICIOUS_PAIR_THRESHOLD = 0.3           # Correlation threshold for suspicion (30%)
MIN_HANDS_FOR_SUSPICION = 5               # Minimum hands needed before flagging suspicious pairs
```

**Configuration Options:**
- `MIN_BET_FOR_COLLUSION`: Lower values catch smaller collusion, higher values reduce false positives
  - Current: $20 (filters out trivial small-bet anomalies)
- `BET_SIZE_SIMILARITY_THRESHOLD`: Lower values = stricter matching, higher = looser
  - Current: 8% (0.08) - catches bets like $100/$108
  - Stricter: 0.01-0.03 for near-exact matches
  - Looser: 0.10+ for more variation
- `COLLUSION_WINDOW`: Standard time window for detecting coordinated bets
  - Current: 6.0 seconds
  - Increase for slower games, decrease for faster detection
- `TIGHT_COLLUSION_WINDOW`: For highly synchronized bets (flagged as "TIGHT")
  - Current: 1.0 second
- `SUSPICIOUS_SEQUENCE_TIME_DIFF`: Max time between actions to be considered "immediate"
  - Current: 2.0 seconds (for bet→immediate raise detection)

### Significant Anomaly Filtering

**Location**: `src/config.py`

The system requires at least one `large_bet` anomaly for collusion detection. Other players must have either:
- `large_bet` type anomaly, OR
- Very high residual (>1.5× threshold)

**Parameters:**
```python
MIN_RESIDUAL_MULTIPLIER = 1.5  # Multiplier for threshold to determine "very high residual"
```

**Configuration:**
- Lower values (1.5) = stricter (requires higher residuals)
- Higher values (2.0-3.0) = looser (allows lower residuals)

**Impact**: Reduces false positives by 20-30% by filtering out weak coincidental anomalies.

## Adjusting Detection Sensitivity

### Reduce False Positives (stricter)

**In `src/config.py`:**

```python
# Make collusion detection stricter
MIN_BET_FOR_COLLUSION = 30.0              # Increase from 20.0 to 30.0
BET_SIZE_SIMILARITY_THRESHOLD = 0.05      # Decrease from 0.08 to 0.05
SIGMA_MULTIPLIER = 4.0                    # Increase from 3.5 to 4.0
MIN_RESIDUAL_MULTIPLIER = 2.0             # Increase from 1.5 to 2.0
```

### Increase Detection Rate (looser)

**In `src/config.py`:**

```python
# Make collusion detection more sensitive
MIN_BET_FOR_COLLUSION = 15.0              # Decrease from 20.0 to 15.0
BET_SIZE_SIMILARITY_THRESHOLD = 0.10      # Increase from 0.08 to 0.10
SIGMA_MULTIPLIER = 3.0                    # Decrease from 3.5 to 3.0
MIN_RESIDUAL_MULTIPLIER = 1.0             # Decrease from 1.5 to 1.0
COLLUSION_WINDOW = 8.0                    # Increase from 6.0 to 8.0
```

## Advanced Configuration

### Process Model Parameters

**Location**: `src/config.py`

```python
# Process model parameters
PROCESS_VELOCITY_OSCILLATION_AMPLITUDE = 0.5  # Velocity oscillation amplitude in process model
PROCESS_DAMPING_FACTOR = 0.1                  # Damping factor for damped process model
```

### Measurement Model Parameters

**Location**: `src/config.py`

```python
# Measurement model parameters
MEASUREMENT_VELOCITY_SCALE = 10.0             # Velocity scaling factor in measurement model
MEASUREMENT_VELOCITY_CLIP_MIN = -5.0          # Minimum clipped velocity value
MEASUREMENT_VELOCITY_CLIP_MAX = 5.0           # Maximum clipped velocity value
```

### Time Delta Constraints

**Location**: `src/config.py`

```python
# Time delta constraints
MIN_DT = 0.01   # Minimum time delta to avoid zero (seconds)
MAX_DT = 100.0  # Maximum time delta to prevent extreme predictions (seconds)
```

### Numerical Stability

**Location**: `src/config.py`

```python
# Filter update configuration
MIN_INNOVATION_VARIANCE = 0.01  # Minimum innovation variance for numerical stability
UKF_RESET_COVARIANCE = 10.0     # Reset covariance value when UKF update fails
```

## Configuration Files

All configuration parameters are centralized in:
- **`src/config.py`**: All tunable parameters with documentation

The configuration is automatically imported by:
- `src/filters/poker_ukf.py`: UKF filter implementation
- `src/anomaly_logger.py`: Anomaly detection and logging
- `src/collusion_detector.py`: Collusion pattern detection
- `src/event_processor.py`: Event processing orchestration

**To modify settings:** Edit values in `src/config.py` and restart the consumer. No code changes required in other files.

