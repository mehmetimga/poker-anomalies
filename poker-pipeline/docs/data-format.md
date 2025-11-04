# Data Format

## Hand History File Format

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

## Anomaly Log Format

JSON entries in `logs/anomalies.log`:

### Individual Anomaly Entry

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
  "details": "Deviation >5σ (threshold=6.00)"
}
```

### Collusion Pattern Entry

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
    "similar_matches": [],
    "match_ratio": 1.0
  },
  "action_sequence": {
    "recent_actions": [
      {"player": "P3", "action": "bet", "timestamp": 1697500280.0},
      {"player": "P1", "action": "raise", "timestamp": 1697500280.5},
      {"player": "P3", "action": "raise", "timestamp": 1697500284.5}
    ],
    "suspicious_patterns": [
      {"pattern": "raise_raise", "players": ["P1", "P3"], "time_diff": 4.0},
      {"pattern": "bet_immediate_raise", "players": ["P3", "P1"], "time_diff": 0.5}
    ],
    "normal_patterns": []
  },
  "anomalies": [
    {"player_id": "P1", "timestamp": 1697500280.5, "residual": 195.93, "bet_amount": 100.0, "anomaly_type": "large_bet", "threshold": 72.0},
    {"player_id": "P3", "timestamp": 1697500284.5, "residual": 185.58, "bet_amount": 100.0, "anomaly_type": "large_bet", "threshold": 80.25}
  ],
  "details": "Synchronized betting anomaly detected among 2 players, max time diff: 4.00s, min bet: $30.00, EXACT bet matches: 1 pairs, match ratio: 100.0%, suspicious sequence: bet_immediate_raise, raise_raise"
}
```

## Anomaly Types

- `high_residual`: Traditional residual-based detection
- `large_bet`: Absolute bet size detection (unusually large bets)
- `large_bet_high_residual`: Both conditions met

