# Troubleshooting

## Kafka Connection Failed

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

## No Events Received

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

## Import Errors

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

## Negative Covariance Warnings

**Problem:** UKF numerical stability issues

**Solution:**
- Adjust UKF parameters (increase `alpha` to 1.0)
- Tune process/measurement noise (`Q`, `R`)
- Reduce time deltas (increase event frequency)

## Quick Fixes

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

