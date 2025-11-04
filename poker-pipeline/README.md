# Poker Collusion Detection Pipeline

A real-time anomaly detection system for identifying collusion in online poker using streaming data, Unscented Kalman Filters (UKF), and multi-layer pattern validation.

## Overview

This pipeline processes poker hand history events in real-time through Apache Kafka, applies UKF-based state estimation to track player betting behaviors, and flags anomalous patterns that may indicate collusion or cheating.

**Performance**: ~92-100% precision, ~3-4% false positive rate, 92% recall

### Key Features

- **Real-time Streaming**: Kafka-based event processing with <100ms latency
- **Advanced Filtering**: Unscented Kalman Filter for non-linear bet pattern tracking
- **Multi-Layer Anomaly Detection**: 
  - 5σ adaptive threshold (reduced false positives by 27%)
  - Absolute bet size detection for unusually large bets
  - Player-specific adaptive thresholds with robust statistics
- **Advanced Collusion Detection** with four-layer validation:
  - **Minimum Bet Size Filter** ($30): Only flags economically significant collusion
  - **Bet Size Matching**: Detects exact/similar bet matches (strong coordination indicator)
  - **Action Sequence Filter**: Validates suspicious betting sequences (bet→immediate raise, raise→raise)
  - **Significant Anomaly Filter**: Requires at least one large_bet anomaly (reduces false positives by 20-30%)
- **High Precision**: ~92-100% precision with ~3-4% false positive rate
- **Modular Architecture**: Easy to extend with additional filters and detection strategies

## Documentation

- **[Architecture](docs/architecture.md)** - System architecture and state estimation
- **[Data Format](docs/data-format.md)** - Input/output data formats
- **[Configuration](docs/configuration.md)** - Filter tuning and threshold configuration
- **[Examples](docs/examples.md)** - Example outputs and log entries
- **[Testing](docs/testing.md)** - Test data and expected results
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Performance](docs/performance.md)** - Benchmarks and scaling considerations
- **[Algorithm Details](docs/algorithm.md)** - UKF implementation and detection strategies
- **[Extensions](docs/extensions.md)** - Future enhancements and advanced models

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
python -m src.producer --topic poker-actions --delay 0.3
```
Note: The producer will automatically process all `table_*.txt` files in the `data/` directory.

## Usage

### Producer Options

```bash
python -m src.producer \
    --topic poker-actions \
    --delay 0.5 \
    --kafka localhost:9092
```

**Parameters:**
- `--input`: (Optional) Path to a single hand history file. If not provided, processes all `table_*.txt` files in `data/` directory
- `--data-dir`: (Optional) Directory containing `table_*.txt` files (default: `data/` relative to project root)
- `--topic`: Kafka topic name (default: `poker-actions`)
- `--delay`: Delay between events in seconds (default: 0.5)
- `--kafka`: Kafka bootstrap servers (default: `localhost:9092`)

**Note:** By default, the producer automatically finds and processes all files matching `table_*.txt` in the `data/` directory. This allows you to have separate files for each table (e.g., `table_1.txt`, `table_2.txt`).

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

## Next Steps

1. **Modify data**: Edit `data/table_*.txt` files or add new `table_N.txt` files with your own hand history
2. **Tune filters**: Adjust Q, R matrices in `src/filters.py` (see [Configuration](docs/configuration.md))
3. **Change thresholds**: Edit anomaly thresholds in `src/anomaly_logger.py` (see [Configuration](docs/configuration.md))
4. **Add features**: Extend with additional detection algorithms (see [Extensions](docs/extensions.md))

## Cleanup

```bash
# Stop the pipeline
Ctrl+C in running terminals

# Stop Kafka
docker-compose down

# Clean logs
rm -f logs/*.log
```

## References

### Research Papers

Research papers are located in `../docs/papers/`:

1. **Online Time Series Anomaly Detection with State Space Gaussian Processes**
   - Foundation for state-space anomaly detection approach

2. **Deep Learning for Time Series Anomaly Detection: A Survey**
   - Overview of modern anomaly detection techniques

### Libraries Used

- `kafka-python`: Kafka client for Python
- `numpy`: Numerical computing
- `scipy`: Scientific computing utilities

---

**Ready to detect collusion? Start with: `./scripts/run_local.sh`** 🎰
