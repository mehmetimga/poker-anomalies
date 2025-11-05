"""
Configuration constants for the poker anomaly detection pipeline.
"""

# Threading configuration
MAX_WORKERS_LIMIT = 4  # Maximum number of parallel threads for file processing

# Kafka configuration
KAFKA_DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_DEFAULT_TOPIC = "poker-actions"
KAFKA_MAX_RETRIES = 10
KAFKA_RETRY_DELAY = 2  # seconds

# Producer configuration
PRODUCER_DEFAULT_DELAY = 0.5  # seconds between events
PRODUCER_ACKS = "all"
PRODUCER_RETRIES = 3
PRODUCER_SEND_TIMEOUT = 10  # seconds - timeout for Kafka send operations
PRODUCER_PROGRESS_INTERVAL = 10  # Print progress every N events

# Consumer configuration
CONSUMER_GROUP_ID = "poker-anomaly-detector"
CONSUMER_AUTO_OFFSET_RESET = "earliest"
CONSUMER_ENABLE_AUTO_COMMIT = True

# File discovery configuration
DEFAULT_DATA_DIR = "data"
TABLE_FILE_PATTERN = "table_*.txt"

# Logging configuration
DEFAULT_LOG_DIR = "logs"

# ============================================================================
# Anomaly Detection Configuration
# ============================================================================

# Collusion detection thresholds
MIN_BET_FOR_COLLUSION = 30.0  # Minimum bet size (dollars) to trigger collusion alerts
BET_SIZE_SIMILARITY_THRESHOLD = (
    0.05  # Bet amounts must be within 5% to be considered matching
)
COLLUSION_WINDOW = 5.0  # Time window (seconds) for detecting collusion patterns
TIGHT_COLLUSION_WINDOW = 1.0  # Tight window for very synchronized bets (seconds)
SUSPICIOUS_SEQUENCE_TIME_DIFF = (
    2.0  # Max time difference for suspicious bet-raise sequences (seconds)
)

# Anomaly tracking history
RECENT_ANOMALIES_MAXLEN = 10  # Maximum number of recent anomalies to track per table
RECENT_ACTIONS_MAXLEN = 10  # Maximum number of recent actions to track per table

# Significant anomaly filtering
MIN_RESIDUAL_MULTIPLIER = (
    2.0  # Multiplier for threshold to determine "very high residual"
)

# Collusion detector
COLLUSION_DETECTOR_WINDOW_SIZE = 10  # Number of recent hands to analyze
SUSPICIOUS_PAIR_THRESHOLD = 0.3  # Correlation threshold for suspicion (30%)
MIN_HANDS_FOR_SUSPICION = 5  # Minimum hands needed before flagging suspicious pairs

# ============================================================================
# UKF Filter Configuration
# ============================================================================

# UKF parameters
UKF_STATE_DIMENSION = 2  # State dimension [position, velocity]
UKF_ALPHA = 1.0  # Spread of sigma points (typically 1e-3 to 1)
UKF_BETA = 2.0  # Prior knowledge (2 for Gaussian)
UKF_KAPPA = 0.0  # Scaling parameter (typically 0 or 3-n)

# Process and measurement noise
PROCESS_NOISE_Q = 0.1  # Process noise covariance multiplier
MEASUREMENT_NOISE_R = 1.0  # Measurement noise covariance

# Initial state
INITIAL_POSITION = 0.0
INITIAL_VELOCITY = 1.0
INITIAL_COVARIANCE = 10.0  # Initial covariance multiplier

# History tracking
RESIDUAL_HISTORY_MAXLEN = 20  # Maximum number of residuals to track per player
BET_HISTORY_MAXLEN = 20  # Maximum number of bets to track per player

# Warm-up period
MIN_SAMPLES_FOR_DETECTION = 5  # Minimum samples needed before flagging anomalies
MIN_SAMPLES_FOR_THRESHOLD = 3  # Minimum samples needed for threshold calculation
MIN_SAMPLES_FOR_ABSOLUTE_BET = 3  # Minimum samples needed for absolute bet threshold

# Adaptive threshold calculation
DEFAULT_STD = 2.0  # Default standard deviation for early samples
SIGMA_MULTIPLIER = 5.0  # Multiplier for standard deviation (5σ threshold)
MIN_THRESHOLD_BASE = 0.5  # Minimum threshold base value
MIN_THRESHOLD_PCT = 0.1  # Minimum threshold as percentage of average bet (10%)
DEFAULT_AVG_BET = 20.0  # Default average bet for threshold calculation
IQR_TO_STD_RATIO = 1.35  # IQR ≈ 1.35 * std for normal distributions

# History tracking caps (to prevent outliers from skewing stats)
RESIDUAL_CAP_MULTIPLIER = 5.0  # Cap residual at 5x median for history tracking
BET_CAP_MULTIPLIER = 3.0  # Cap bet at 3x median for history tracking

# Time delta constraints
MIN_DT = 0.01  # Minimum time delta to avoid zero (seconds)
MAX_DT = 100.0  # Maximum time delta to prevent extreme predictions (seconds)

# Absolute bet threshold
ABSOLUTE_BET_THRESHOLD_DEFAULT = 50.0  # Default threshold for large bets
ABSOLUTE_BET_Q75_MULTIPLIER = 3.0  # 3x 75th percentile
ABSOLUTE_BET_MEDIAN_MULTIPLIER = 2.0  # 2x median

# Actions configuration
TRACKED_ACTIONS = ["bet", "raise", "call"]  # Actions to track in filter updates
ABSOLUTE_BET_ACTIONS = ["bet", "raise"]  # Actions to check for absolute bet anomalies

# Filter update configuration
MIN_INNOVATION_VARIANCE = 0.01  # Minimum innovation variance for numerical stability

# Kalman filter initial state
KALMAN_INITIAL_COVARIANCE = (
    1000.0  # Initial covariance multiplier for standard Kalman filter
)

# UKF numerical stability
UKF_RESET_COVARIANCE = 10.0  # Reset covariance value when UKF update fails

# Process model parameters
PROCESS_VELOCITY_OSCILLATION_AMPLITUDE = (
    0.5  # Velocity oscillation amplitude in process model
)
PROCESS_DAMPING_FACTOR = 0.1  # Damping factor for damped process model

# Measurement model parameters
MEASUREMENT_VELOCITY_SCALE = 10.0  # Velocity scaling factor in measurement model
MEASUREMENT_VELOCITY_CLIP_MIN = -5.0  # Minimum clipped velocity value
MEASUREMENT_VELOCITY_CLIP_MAX = 5.0  # Maximum clipped velocity value
