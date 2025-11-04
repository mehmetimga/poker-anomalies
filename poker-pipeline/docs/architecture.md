# Architecture

## System Overview

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

## State Estimation

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

## Components

- **Producer**: Reads hand history files and publishes events to Kafka
- **Kafka**: Message broker for real-time event streaming
- **Consumer**: Processes events, applies UKF filtering, and detects anomalies
- **Anomaly Logger**: Logs detected anomalies and collusion patterns to JSON log file

