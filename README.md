# Poker Collusion Detection Pipeline

A real-time anomaly detection system for identifying collusion in online poker using streaming data, Unscented Kalman Filters (UKF), and residual-based pattern analysis.

## Overview

This pipeline processes poker hand history events in real-time through Apache Kafka, applies UKF-based state estimation to track player betting behaviors, and flags anomalous patterns that may indicate collusion or cheating.

### Key Features

- **Real-time Streaming**: Kafka-based event processing with <100ms latency
- **Advanced Filtering**: Unscented Kalman Filter for non-linear bet pattern tracking
- **Anomaly Detection**: Adaptive threshold-based detection (3σ residual criterion)
- **Collusion Detection**: Identifies synchronized betting patterns across multiple players
- **Modular Architecture**: Easy to extend with additional filters and detection strategies

## Architecture

```
┌─────────────────┐      ┌──────────────┐      ┌────────────────────┐
│  Hand History   │────▶│    Kafka     │────▶│  UKF Consumer      │
│  (Producer)     │      │    Topic     │      │  + Anomaly Logger  │
└─────────────────┘      └──────────────┘      └────────────────────┘
                                                       │
                                                       ▼
                                               ┌────────────────┐
                                               │  anomalies.log │
                                               └────────────────┘
```

### State Estimation

**State Vector**: `[bet_position, aggression_velocity]`
- **Position**: Cumulative bet pattern over time
- **Velocity**: Rate of change in betting aggression

**Process Model** (non-linear):
```
pos' = pos + vel * dt
vel' = vel + sin(pos) * dt * 0.5
```

**Measurement Model**:
```
z = pos * exp(vel / 10)
```

## Prerequisites

- **Python 3.10+**
- **Docker** (for Kafka)
- **Docker Compose**

## Installation

### 1. Clone and Setup

```bash
cd poker-pipeline
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Kafka

```bash
docker-compose up -d
```

Wait ~10 seconds for Kafka to initialize.

### 3. Verify Kafka is Running

```bash
docker-compose ps
```

You should see both `zookeeper` and `kafka` containers running.

## Quick Start

### Option 1: Automated Run (Recommended)

```bash
./scripts/run_local.sh
```

This script will:
1. Check Docker and Python dependencies
2. Start Kafka (if not running)
3. Create virtual environment (if needed)
4. Install dependencies
5. Run producer and consumer
6. Display results and anomalies

### Option 2: Manual Run

**Terminal 1 - Start Consumer:**
```bash
source venv/bin/activate
python -m src.consumer --topic poker-actions --kafka localhost:9092
```

**Terminal 2 - Start Producer:**
```bash
source venv/bin/activate
python -m src.producer --input data/sample_hand_history.txt --topic poker-actions --delay 0.3
```

## Usage

### Producer Options

```bash
python -m src.producer \
    --input data/sample_hand_history.txt \
    --topic poker-actions \
    --delay 0.5 \
    --kafka localhost:9092
```

**Parameters:**
- `--input`: Path to hand history file (default: `data/sample_hand_history.txt`)
- `--topic`: Kafka topic name (default: `poker-actions`)
- `--delay`: Delay between events in seconds (default: 0.5)
- `--kafka`: Kafka bootstrap servers (default: `localhost:9092`)

### Consumer Options

```bash
python -m src.consumer \
    --topic poker-actions \
    --kafka localhost:9092 \
    --log logs/anomalies.log
```

**Parameters:**
- `--topic`: Kafka topic name (default: `poker-actions`)
- `--kafka`: Kafka bootstrap servers (default: `localhost:9092`)
- `--log`: Path to anomaly log file (default: `logs/anomalies.log`)

## Data Format

### Hand History File Format

```
timestamp|table_id|player_id|action|amount|pot
```

**Example:**
```
1697500000.0|1|P1|bet|10.0|10.0
1697500001.0|1|P2|call|10.0|20.0
1697500002.0|1|P3|fold|0.0|20.0
1697500003.0|1|P4|raise|20.0|40.0
```

**Fields:**
- `timestamp`: Unix timestamp (float)
- `table_id`: Table identifier (int)
- `player_id`: Player identifier (string)
- `action`: bet, call, raise, fold
- `amount`: Bet amount in currency units (float)
- `pot`: Total pot size (float)

### Anomaly Log Format

JSON entries in `logs/anomalies.log`:

```json
{
  "timestamp": 1697500280.0,
  "player_id": "P1",
  "table_id": 1,
  "action": "bet",
  "amount": 50.0,
  "residual": 8.2,
  "threshold": 6.0,
  "type": "high_residual",
  "details": "Deviation >3σ (threshold=6.00)"
}
```

**Collusion Pattern Entry:**
```json
{
  "timestamp": 1697500284.5,
  "table_id": 1,
  "type": "collusion_pattern",
  "players": ["P1", "P3"],
  "num_players": 2,
  "anomalies": [...],
  "details": "Synchronized betting anomaly detected among 2 players"
}
```

## Configuration

### Filter Tuning

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

### Anomaly Thresholds

Edit `src/anomaly_logger.py` - `AnomalyLogger.__init__()`:

```python
# Time window for collusion detection (seconds)
self.collusion_window = 5.0

# Adaptive threshold calculation in PokerUKF
def get_adaptive_threshold(self, default_std=2.0):
    # Returns 3 * std_dev of historical residuals
    return 3 * max(std, 0.5)  # Minimum threshold
```

## Example Output

### Console Output

```
==============================================================
POKER ANOMALY DETECTION PIPELINE
==============================================================
Topic: poker-actions
Kafka: localhost:9092
Log file: logs/anomalies.log
==============================================================

✓ Player P1: bet    $ 10.00 | Est: $  0.00 | Residual:   2.50 | Threshold: 6.00
✓ Player P2: call   $ 10.00 | Est: $  0.00 | Residual:   2.50 | Threshold: 6.00
✓ Player P3: fold   $  0.00 | Est: $  0.00 | Residual:   0.00 | Threshold: 6.00

⚠️  ANOMALY DETECTED: Player P1 at table 1
   Action: bet $50.00, Residual: 8.20 (threshold: 6.00)

🚨 COLLUSION DETECTED at table 1!
   Players involved: P1, P3
   Time window: 5.0s
   Pattern: Synchronized high residuals
```

### Summary Statistics

```
==============================================================
PIPELINE COMPLETED
==============================================================
Events processed: 180
Anomalies detected: 12
Players tracked: 6
Time elapsed: 54.23s
Events/sec: 3.32
==============================================================

ANOMALY DETECTION SUMMARY
==============================================================
Total anomalies detected: 12
Collusion patterns found: 3
Tables monitored: 1
==============================================================
```

## Testing

### Synthetic Data

The included `data/sample_hand_history.txt` contains:
- **20 hands** of 6-player poker
- **Normal betting patterns** (hands 1-14, 18-20)
- **Anomalous synchronized bets** (hands 15-17)
  - Hand 15: P1 & P3 collusion
  - Hand 16: P2 & P4 collusion
  - Hand 17: P1 & P3 collusion again

### Expected Results

Running the pipeline should detect:
- ~12-15 individual player anomalies
- 2-3 collusion patterns (synchronized anomalies)
- High correlation scores for pairs: (P1, P3) and (P2, P4)

## Troubleshooting

### Kafka Connection Failed

**Problem:** `NoBrokersAvailable` error

**Solution:**
```bash
# Check if Kafka is running
docker-compose ps

# Restart Kafka
docker-compose restart

# Check logs
docker-compose logs kafka
```

### No Events Received

**Problem:** Consumer receives no events

**Solution:**
1. Ensure producer completed successfully
2. Check topic exists:
```bash
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092
```
3. Reset consumer offset:
```bash
docker exec kafka kafka-consumer-groups --bootstrap-server localhost:9092 \
    --group poker-anomaly-detector --reset-offsets --to-earliest \
    --topic poker-actions --execute
```

### Import Errors

**Problem:** `ModuleNotFoundError`

**Solution:**
```bash
# Ensure you're in the project root
cd poker-pipeline

# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Negative Covariance Warnings

**Problem:** UKF numerical stability issues

**Solution:**
- Adjust UKF parameters (increase `alpha` to 1.0)
- Tune process/measurement noise (`Q`, `R`)
- Reduce time deltas (increase event frequency)

## Performance

### Benchmarks (on MacBook Pro M1)

- **Latency**: ~50ms per event (predict + update + log)
- **Throughput**: ~20 events/second (single consumer)
- **Memory**: ~50MB for 6 players + 200 events
- **CPU**: ~10% single core utilization

### Scaling Considerations

For production deployment:
- Use **multiple consumer instances** (Kafka consumer groups)
- Partition by **table_id** for parallel processing
- Deploy on **AWS ECS/EKS** with auto-scaling
- Replace local Kafka with **AWS MSK** or **Confluent Cloud**

## Project Structure

```
poker-anomalies/
├── README.md                   # This file (main documentation)
├── docs/                       # Research papers and investigations
│   ├── papers/                 # Research papers (PDFs)
│   ├── ai/                     # AI-generated documentation
│   └── investigations/
│       └── investigation*.md       # Investigation documents
│
└── poker-pipeline/
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Kafka setup
├── data/
│   └── sample_hand_history.txt # Synthetic hand data
├── src/
│   ├── __init__.py
│   ├── producer.py             # Kafka producer
│   ├── consumer.py             # Kafka consumer + detection
│   ├── filters.py              # Kalman/EKF/UKF implementations
│   ├── models.py               # Process/measurement models
│   └── anomaly_logger.py       # Logging and alerting
├── QUICKSTART.md               # Quick start guide
├── logs/
│   └── anomalies.log           # Anomaly output (generated)
├── scripts/
│   └── run_local.sh            # Automated runner
└── tests/
    └── test_filters.py         # Test suite
```

## Algorithm Details

### Unscented Kalman Filter (UKF)

The UKF propagates **sigma points** through non-linear functions to estimate state:

1. **Sigma Point Generation**: Create 2n+1 points around current state
2. **Predict Step**: Propagate through process model
3. **Update Step**: Correct with measurement via weighted mean

**Advantages over EKF:**
- No Jacobian computation required
- Better accuracy for highly non-linear systems
- More robust to strong non-linearities in poker betting

### Anomaly Detection Strategy

1. **Individual Anomalies**: 
   - Calculate residual (innovation) from UKF update
   - Flag if `|residual| > 3 * σ` (adaptive threshold)

2. **Collusion Detection**:
   - Track anomalies within time window (5s)
   - Flag if ≥2 players have simultaneous anomalies
   - Calculate correlation scores for player pairs

3. **Adaptive Thresholds**:
   - Rolling window of 20 hands per player
   - Dynamic σ based on historical residuals
   - Minimum threshold to avoid false positives

## Extensions

### Future Enhancements

- [ ] Multi-table support with separate Kafka topics
- [ ] Web dashboard (React + WebSockets) for real-time monitoring
- [ ] Additional collusion patterns (chip dumping, soft-play)
- [ ] LSTM autoencoder for sequence anomaly detection
- [ ] Model persistence and checkpoint/restore
- [ ] Grafana integration for metrics visualization
- [ ] Integration with poker platform APIs

### Advanced Models

- **Hidden Markov Models** for state transition anomalies
- **Graph Neural Networks** for player relationship modeling
- **Isolation Forest** for unsupervised multi-dimensional anomalies

## References

### Research Papers

Research papers are located in `docs/papers/`:

1. **Online Time Series Anomaly Detection with State Space Gaussian Processes**
   - Foundation for state-space anomaly detection approach

2. **Deep Learning for Time Series Anomaly Detection: A Survey**
   - Overview of modern anomaly detection techniques

### Libraries Used

- `kafka-python`: Kafka client for Python
- `numpy`: Numerical computing
- `scipy`: Scientific computing utilities

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
- Open a GitHub issue
- Check troubleshooting section above
- Review investigation files for methodology details

---

**Built with ❤️ for fraud detection in online gaming**


