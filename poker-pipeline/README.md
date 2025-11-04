# Quick Start Guide

Get the poker anomaly detection pipeline running in 5 minutes!

## Prerequisites Check

```bash
# Check Python version (need 3.10+)
python3 --version

# Check Docker
docker --version
docker info

# Check Docker Compose
docker-compose --version
```

## Step 1: Install Dependencies

```bash
cd poker-pipeline

# Install Python packages
pip3 install -r requirements.txt

# Verify installation
python3 tests/test_filters.py
```

Expected output: "✅ ALL TESTS PASSED!"

## Step 2: Start Kafka

```bash
# Start Kafka and Zookeeper
docker-compose up -d

# Wait for Kafka to be ready (~10 seconds)
sleep 10

# Verify Kafka is running
docker-compose ps
```

You should see both `kafka` and `zookeeper` with status "Up".

## Step 3: Run the Pipeline

### Option A: Automated (Recommended)

```bash
./scripts/run_local.sh
```

This script handles everything automatically!

### Option B: Manual (Two Terminals)

**Terminal 1 - Consumer:**
```bash
cd poker-pipeline
python3 -m src.consumer
```

**Terminal 2 - Producer:**
```bash
cd poker-pipeline
python3 -m src.producer --delay 0.3
```

## Step 4: View Results

### Console Output

Watch the real-time anomaly detection in your terminal:
```
✓ Player P1: bet    $ 10.00 | Est: $  0.00 | Residual:   2.50
⚠️  ANOMALY DETECTED: Player P1 at table 1
🚨 COLLUSION DETECTED at table 1!
```

### Log File

```bash
# View anomaly log
cat logs/anomalies.log

# Live tail
tail -f logs/anomalies.log
```

### Example Output

```json
{"timestamp": 1697500280.0, "player_id": "P1", "table_id": 1, "residual": 8.2, "type": "high_residual"}
{"timestamp": 1697500284.5, "table_id": 1, "type": "collusion_pattern", "players": ["P1", "P3"]}
```

## What to Expect

The sample data includes:
- **20 poker hands** with 6 players (P1-P6)
- **3 anomalous hands** (15-17) with synchronized betting
- **Expected detections**: 12-15 anomalies, 2-3 collusion patterns

## Cleanup

```bash
# Stop the pipeline
Ctrl+C in running terminals

# Stop Kafka
docker-compose down

# Clean logs
rm -f logs/*.log
```

## Troubleshooting

### "Kafka not available"
```bash
docker-compose restart
sleep 10
```

### "No events received"
```bash
# Check topic exists
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Should show: poker-actions
```

### "ModuleNotFoundError"
```bash
pip3 install -r requirements.txt
```

## Next Steps

1. **Modify data**: Edit `data/sample_hand_history.txt` with your own hand history
2. **Tune filters**: Adjust Q, R matrices in `src/filters.py`
3. **Change thresholds**: Edit anomaly thresholds in `src/anomaly_logger.py`
4. **Add features**: Extend with additional detection algorithms

## Performance Tips

- Reduce `--delay` for faster processing (default: 0.5s)
- Increase Kafka partitions for multi-table support
- Use virtual environment for isolation

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Documentation

- Full details: See `README.md`
- Architecture: See investigation files in parent directory
- API reference: Docstrings in source files

---

**Ready to detect collusion? Start with: `./scripts/run_local.sh`** 🎰


