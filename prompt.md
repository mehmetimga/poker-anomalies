
python
content = """# Poker Collusion Detection Pipeline: Local Development Prompt for Vibe Coding Agent

## Overview
You are tasked with building a local, end-to-end prototype for a real-time poker collusion detection pipeline. This system will:

- **Ingest hand history data** from a text file (simulating game logs).
- **Feed events into Apache Kafka** (running locally) to mimic streaming.
- **Consume from Kafka**, process with Kalman/Unscented Kalman Filters (UKF) for state estimation on player actions (e.g., bet sizes).
- **Detect anomalies** (e.g., unusual bet patterns indicating collusion) using filter residuals.
- **Log anomalies** to a file or console for review.

This is for **local development only**—no AWS integration yet. Use Python 3.10+ with lightweight libraries (NumPy, Kafka-Python). Keep it modular, testable, and runnable in a single directory.

**Goals**:
- Simulate a poker table with 6 players.
- Process sequential hand actions (bets, folds, etc.) as time series.
- Use UKF for non-linear bet tracking (from provided snippets).
- Flag anomalies if residual > 3σ (e.g., synchronized bets).
- Output: Processed estimates, anomaly logs.

**Non-Goals** (for now):
- Full poker engine simulation—use static hand history.
- Multi-table scaling—single topic/consumer.
- AWS (S3, EMR)—defer to later phase.

Run everything locally: Start Kafka in Docker, feed data, process in real-time simulation.

## Prerequisites
Instruct the user (or assume) to install:
- Python 3.10+.
- Docker (for local Kafka).
- pip install: `kafka-python numpy scipy` (no heavy deps like Flink).
- No internet-required libs.

**Local Kafka Setup** (via Docker—add to your init script):
```bash
# Run Kafka in Docker (single-node for local)
docker run -d --name kafka-local -p 9092:9092 -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT -e KAFKA_INTER_BROKER_LISTENER_NAME=PLAINTEXT -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 apache/kafka:3.5.0
# Create topic
docker exec kafka-local kafka-topics.sh --create --topic poker-actions --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
Project Structure
Create this directory layout:
textpoker-pipeline/
├── README.md              # Your notes
├── requirements.txt       # kafka-python, numpy
├── docker-compose.yml     # Optional: For Kafka (if not using raw docker run)
├── data/
│   └── sample_hand_history.txt  # Input file (example below)
├── src/
│   ├── producer.py        # Reads text file, parses, sends to Kafka
│   ├── consumer.py        # Reads Kafka, processes with UKF, logs anomalies
│   ├── filters.py         # UKF/Kalman classes (from snippets)
│   └── anomaly_logger.py  # Simple file logger
├── logs/
│   └── anomalies.log      # Output: Timestamped anomaly entries
└── run_local.sh           # Script to start: Kafka + producer + consumer
Step-by-Step Implementation Guide
1. Data Ingestion: Producer (src/producer.py)

Read data/sample_hand_history.txt line-by-line.
Parse each line as a JSON-like event: e.g., {"timestamp": 1234567890.1, "table_id": 1, "player_id": "P1", "action": "bet", "amount": 10.0, "pot": 50.0}.
Simulate streaming: Add a small delay (0.5s per event) to mimic real-time.
Send to Kafka topic poker-actions using kafka-python.
Handle EOF: Send a "END_STREAM" event.

Key Logic:

Use json for serialization.
Buffer events if needed, but keep simple—send immediately.
Error handling: Retry on Kafka connect fail.

Example Code Snippet:
pythonimport json
import time
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def parse_hand_line(line):
    # Assume line format: timestamp|table_id|player_id|action|amount|pot
    parts = line.strip().split('|')
    if len(parts) != 6:
        return None
    ts, table, player, action, amount, pot = parts
    return {
        'timestamp': float(ts),
        'table_id': int(table),
        'player_id': player,
        'action': action,
        'amount': float(amount) if amount != 'fold' else 0.0,
        'pot': float(pot)
    }

with open('data/sample_hand_history.txt', 'r') as f:
    for line in f:
        event = parse_hand_line(line)
        if event:
            producer.send('poker-actions', value=event)
            time.sleep(0.5)  # Simulate real-time delay
        else:
            continue

# End signal
producer.send('poker-actions', value={'type': 'END_STREAM'})
producer.flush()
2. State Estimation & Processing: Filters (src/filters.py)

Implement UKF from the provided snippet (adapt for poker: state = [bet_position, aggression_velocity]).
Process Model: Non-linear bet evolution, e.g., pos += vel * dt, vel += sin(pos) * dt (oscillating aggression).
Measurement: Bet amount (or 0 for fold).
Initialize per player (use dict keyed by player_id).
On each event: If action=='bet', predict + update with amount.

Adaptations for Poker:

dt = timestamp delta from last event.
Only update on 'bet'/'raise'; skip folds but predict forward.
Multi-player: Maintain separate UKF instances per player.

Example Code Snippet (Full UKF Class from Previous—Paste Here)**:
python# [Insert full UnscentedKalmanFilter class from earlier response here]
# Additional: Poker-specific wrapper
class PokerUKF:
    def __init__(self, player_id):
        self.player_id = player_id
        self.ukf = UnscentedKalmanFilter(n=2, alpha=1.0)
        self.ukf.set_covariances(Q=np.eye(2)*0.1, R=np.array([[1.0]]))
        self.last_ts = None
        self.ukf.x = np.array([[0.0], [1.0]])  # Init pos=0, vel=1

    def process_event(self, event):
        if self.last_ts is None:
            self.last_ts = event['timestamp']
            return {'estimate': 0.0, 'residual': 0.0}
        
        dt = event['timestamp'] - self.last_ts
        if event['action'] in ['bet', 'raise']:
            self.ukf.predict(process_model, dt)
            z = np.array([[event['amount']]])
            self.ukf.update(z, measurement_model)
            estimate = self.ukf.get_state()[0]
            # Residual (innovation proxy)
            h_x = measurement_model(self.ukf.x)[0,0]
            residual = abs(event['amount'] - h_x)
        else:  # Fold or other: just predict
            self.ukf.predict(process_model, dt)
            estimate = self.ukf.get_state()[0]
            residual = 0.0
        
        self.last_ts = event['timestamp']
        return {'estimate': estimate, 'residual': residual, 'action': event['action']}
(Define process_model and measurement_model as in UKF snippet.)
3. Anomaly Detection & Logging (src/anomaly_logger.py)

Threshold: Anomaly if residual > 3 * std_dev (track rolling std per player, init=2.0).
Collusion Proxy: If two players' residuals spike in same hand (table_id match), flag as "potential sync".
Log Format: JSON to logs/anomalies.log: {"timestamp": ts, "player_id": "P1", "table_id": 1, "residual": 5.2, "type": "high_residual", "details": "Bet deviated from trend"}.
Use logging module with file handler.

Example Code Snippet:
pythonimport json
import logging
from datetime import datetime

logging.basicConfig(filename='logs/anomalies.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

def log_anomaly(event, residual, std_dev=2.0, anomaly_type='high_residual'):
    if abs(residual) > 3 * std_dev:
        log_entry = {
            'timestamp': event['timestamp'],
            'player_id': event['player_id'],
            'table_id': event['table_id'],
            'residual': residual,
            'type': anomaly_type,
            'details': f"Deviation >3σ (std={std_dev})"
        }
        logging.info(json.dumps(log_entry))
        print(f"ANOMALY LOGGED: {log_entry}")  # Console echo
4. Consumer (src/consumer.py)

Subscribe to poker-actions.
Maintain player UKFs in a dict: {player_id: PokerUKF}.
For each message:

If 'END_STREAM', exit.
Parse event, get player UKF (init if new).
Process: Get estimate/residual.
Log if anomaly.
Print summary: "Player P1: Est bet=8.5, Actual=10.0, Residual=1.5".


Run in loop until end.

Example Code Snippet:
pythonfrom kafka import KafkaConsumer
import json
from filters import PokerUKF
from anomaly_logger import log_anomaly

consumer = KafkaConsumer('poker-actions', bootstrap_servers='localhost:9092',
                         value_deserializer=lambda m: json.loads(m.decode('utf-8')))
player_filters = {}

for message in consumer:
    event = message.value
    if event.get('type') == 'END_STREAM':
        break
    
    player_id = event['player_id']
    if player_id not in player_filters:
        player_filters[player_id] = PokerUKF(player_id)
    
    result = player_filters[player_id].process_event(event)
    print(f"Processed {event['player_id']}: Est={result['estimate']:.2f}, Actual={event.get('amount',0):.2f}, Residual={result['residual']:.2f}")
    
    log_anomaly(event, result['residual'])
    
    # Collusion check: Track per-hand residuals (simplified—use dict by table_id/hand_id if extended)
5. Running Locally (run_local.sh)
bash#!/bin/bash
# Start Kafka
docker start kafka-local || docker run ...  # From prereqs

# Create topic if needed
docker exec kafka-local kafka-topics.sh --create ...  # If not exists

# In separate terminals:
python src/producer.py &
python src/consumer.py
# Or: python -m multiprocessing for parallel
Example Docket: Full Sample Data
Create data/sample_hand_history.txt