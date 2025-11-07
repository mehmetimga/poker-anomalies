# Example Output

## Console Output

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
   Time window: 6.0s (normal)
   Max time difference: 4.00s
   Min bet threshold: $20.00
   ⚠️  EXACT bet matches: P1&P3: $100.00
   ⚠️  Suspicious sequence: raise→raise (P1&P3, 4.00s), bet→immediate_raise (P3&P1, 0.50s)
   Pattern: Synchronized betting
```

## Summary Statistics

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

## Log File Examples

### Individual Anomaly

```json
{
  "timestamp": 1697500280.0,
  "player_id": "P1",
  "table_id": 1,
  "action": "raise",
  "amount": 100.0,
  "residual": 195.93,
  "threshold": 72.0,
  "type": "large_bet",
  "details": "Large bet detected (amount=$100.00, threshold=$72.00)"
}
```

### Collusion Pattern (with all filters)

```json
{
  "timestamp": 1697500284.5,
  "table_id": 1,
  "type": "collusion_pattern",
  "players": ["P1", "P3"],
  "num_players": 2,
  "sync_level": "normal",
  "max_time_diff": 4.0,
  "min_bet_threshold": 30.0,
  "bet_size_similarity_threshold": 0.05,
  "bet_matching": {
    "exact_matches": [{"players": ["P1", "P3"], "bet_amount": 100.0}],
    "match_ratio": 1.0
  },
  "action_sequence": {
    "suspicious_patterns": [
      {"pattern": "raise_raise", "players": ["P1", "P3"], "time_diff": 4.0},
      {"pattern": "bet_immediate_raise", "players": ["P3", "P1"], "time_diff": 0.5}
    ]
  },
  "anomalies": [
    {"player_id": "P1", "timestamp": 1697500280.5, "residual": 195.93, "bet_amount": 100.0, "anomaly_type": "large_bet", "threshold": 72.0},
    {"player_id": "P3", "timestamp": 1697500284.5, "residual": 185.58, "bet_amount": 100.0, "anomaly_type": "large_bet", "threshold": 80.25}
  ],
  "details": "Synchronized betting anomaly detected among 2 players, max time diff: 4.00s, min bet: $20.00, EXACT bet matches: 1 pairs, match ratio: 100.0%, suspicious sequence: bet_immediate_raise, raise_raise"
}
```

