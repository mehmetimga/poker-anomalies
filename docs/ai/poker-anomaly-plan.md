# Poker Anomaly Detection Pipeline Implementation

## Overview

Build a local, end-to-end prototype for real-time poker collusion detection using Apache Kafka for streaming, Kalman/Unscented Kalman Filters (UKF) for state estimation on player betting patterns, and residual-based anomaly detection.

## Architecture

### Project Structure

```
poker-pipeline/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── data/
│   └── sample_hand_history.txt
├── src/
│   ├── __init__.py
│   ├── producer.py           # Kafka producer for hand history
│   ├── consumer.py            # Kafka consumer with anomaly detection
│   ├── filters.py             # Kalman, EKF, UKF implementations
│   ├── anomaly_logger.py      # Logging and alerting
│   └── models.py              # Process and measurement models
├── logs/
│   └── anomalies.log
└── scripts/
    └── run_local.sh
```

## Implementation Steps

### 1. Project Setup & Dependencies

- Create project directory structure
- Set up `requirements.txt` with: kafka-python, numpy, scipy
- Create `docker-compose.yml` for local Kafka (Apache Kafka 3.5.0)
- Initialize logging directory and configuration

### 2. Data Layer

**File: `data/sample_hand_history.txt`**

- Generate synthetic hand history with format: `timestamp|table_id|player_id|action|amount|pot`
- Include 6 players (P1-P6) across multiple hands
- Simulate realistic betting patterns with some anomalous synchronized bets for testing
- Example: `1234567890.1|1|P1|bet|10.0|50.0`

### 3. Filter Implementations

**File: `src/filters.py`**

Based on investigation files 2, 3, and 4, implement three filter types:

**SimpleKalmanFilter (1D):**

- Track scalar state (bet size)
- Constant velocity model
- Methods: `predict()`, `update(z)`, `get_estimate()`

**KalmanFilter (Multivariate):**

- 2D state: [position, velocity] for bet trends
- State transition with dt parameter
- Process noise covariance Q and measurement noise R

**ExtendedKalmanFilter (EKF):**

- Non-linear process/measurement via Jacobians
- Linearization at each step
- Handle damped velocity (exponential decay)

**UnscentedKalmanFilter (UKF):**

- Sigma point propagation (no Jacobians needed)
- Alpha, beta, kappa parameters for tuning
- Better for non-linear bet escalation patterns
- Methods: `sigma_points()`, `predict(f, dt)`, `update(z, h)`

### 4. Process & Measurement Models

**File: `src/models.py`**

Define non-linear models for poker betting:

**Process Model (UKF):**

```python
def process_model(x, dt):
    # State: [bet_position, aggression_velocity]
    # Non-linear: vel += sin(pos) * dt (oscillating aggression)
    pos = x[0] + x[1] * dt
    vel = x[1] + np.sin(x[0]) * dt
    return np.array([pos, vel])
```

**Measurement Model:**

```python
def measurement_model(x):
    # Observe position amplified by velocity
    return x[0] * np.exp(x[1]/10)
```

### 5. Poker-Specific UKF Wrapper

**File: `src/filters.py` (PokerUKF class)**

- Initialize per-player UKF instance
- Track last timestamp for dt calculation
- `process_event(event)` method:
  - Calculate dt from timestamp delta
  - Predict state forward
  - Update on 'bet'/'raise' actions (skip folds but still predict)
  - Return: estimate, residual, action type
- Maintain rolling statistics for adaptive thresholds

### 6. Kafka Producer

**File: `src/producer.py`**

- Read `data/sample_hand_history.txt` line by line
- Parse format: `timestamp|table_id|player_id|action|amount|pot`
- Serialize to JSON with KafkaProducer
- Simulate real-time with 0.5s delay per event
- Send to topic `poker-actions`
- Send END_STREAM event on completion
- Error handling with Kafka connection retry logic

### 7. Kafka Consumer with Anomaly Detection

**File: `src/consumer.py`**

- Subscribe to `poker-actions` topic
- Maintain dictionary of player UKF instances: `{player_id: PokerUKF}`
- For each event:
  - Initialize player UKF if new
  - Process event through UKF
  - Calculate residual (innovation)
  - Check anomaly threshold (>3σ)
  - Log if anomalous
  - Print summary: "Player P1: Est=8.5, Actual=10.0, Residual=1.5"
- Track per-table residuals for collusion correlation detection
- Exit on END_STREAM event

### 8. Anomaly Detection Logic

**File: `src/anomaly_logger.py`**

- Threshold: residual > 3 * std_dev (initialize std=2.0)
- Rolling standard deviation per player (adaptive)
- Collusion detection: Flag when 2+ players have simultaneous high residuals in same hand
- Log format (JSON):
```json
{
  "timestamp": 1234567890.1,
  "player_id": "P1",
  "table_id": 1,
  "residual": 5.2,
  "type": "high_residual",
  "details": "Bet deviated >3σ from trend"
}
```

- Console echo for real-time monitoring
- Use Python logging module with file handler

### 9. Docker & Kafka Setup

**File: `docker-compose.yml`**

- Single-node Kafka for local development
- Zookeeper configuration
- Kafka broker on port 9092
- Auto-create topic `poker-actions` with 1 partition, 1 replication factor

**File: `scripts/run_local.sh`**

- Start Docker Compose
- Wait for Kafka ready
- Create topic if not exists
- Run producer and consumer (with proper process management)

### 10. Documentation

**File: `README.md`**

- Project overview and architecture diagram
- Prerequisites (Python 3.10+, Docker)
- Installation instructions
- Quick start guide
- Configuration options (Q, R matrices, thresholds)
- Example output and interpretation
- Troubleshooting common issues

## Key Technical Details

### State Estimation Approach

- Use UKF as primary filter (best for non-linear poker dynamics)
- State vector: [bet_position, aggression_velocity]
- Process noise Q = 0.1 * I (tunable)
- Measurement noise R = 1.0 (tunable based on bet variance)

### Anomaly Detection Strategy

1. **Individual Anomalies**: Residual exceeds 3σ threshold
2. **Collusion Detection**: Synchronized anomalies across players in same hand
3. **Adaptive Thresholds**: Rolling statistics per player (window=20 hands)

### Performance Targets

- Latency: <100ms per event processing
- Throughput: Handle 1000s of hands for 6-player table
- Memory: O(n) where n = number of active players

## Testing Strategy

- Synthetic data with known anomalies (inject synchronized bets)
- Verify UKF convergence on normal play patterns
- Validate anomaly detection with precision/recall metrics
- Test Kafka producer/consumer integration end-to-end

## Future Enhancements (Not in Initial Scope)

- Multi-table support with multiple Kafka topics
- AWS integration (S3, Kinesis, EMR)
- Web dashboard for real-time monitoring
- Advanced collusion patterns (chip dumping, soft-play)
- Model persistence and retraining pipeline