# Poker Collusion Detection Pipeline – Deep Dive Brief

## 1. Executive Snapshot
- **Mission**: Detect coordinated cheating in online poker without drowning analysts in false positives.
- **Result**: Real-time Kafka/UKF pipeline that catches 11/12 staged collusion hands (91.7% recall) at 92–100% precision and 3–4% false positives, all with ~50ms per-event processing.
- **Assets**: Modular Python codebase (`poker-pipeline/`), scripted demos, anomaly logs, documentation, and roadmap for advanced models.

---

## 2. Industry Pain & Value Proposition
- **High Noise**: Manual reviews + rigid rules engines flag everything; precision was 7% at baseline.
- **Stakeholder Risk**: Undetected collusion erodes trust, drives high-value players away, and invites regulatory fines.
- **Value Delivered**: High-confidence alerts targeted at meaningful pots, enabling compliance teams to act quickly without scaling headcount. Investors hear: "We convert raw hand histories into actionable, analyst-ready leads."

---

## 3. Data & System Architecture
```
data/table_*.txt ─▶ src/producer.py ─▶ Kafka (docker-compose)
                                         │
                                         ▼
                                  src/consumer.py
                                         │
                   ┌────────── Filter Stack (UKF, thresholds, validators) ──────────┐
                   ▼                                                                ▼
      logs/table_*.log (per-table JSON)                          Collusion statistics cache
```
- **Producer**: Streams historical or live hand histories into Kafka topic `poker-actions`.
- **Consumer**: Subscribes in real time, routes each action through the analytics core, and updates alerting logic.
- **Storage**: JSON entries in `logs/table_{id}.log` with full context for analysts or downstream tools.
- **Docs & Scripts**: Located in `poker-pipeline/docs/` and `scripts/run_local.sh` / `run_detection.sh`.

---

## 4. Analytics Core – How We Model Behaviour
- **State Representation**: Each player maintains `[bet_position, aggression_velocity]`, describing cumulative betting profile and rate of change.
- **Process Model**: Predicts how aggression evolves between actions, capturing oscillations (players tend to shift gears rather than jump randomly).
- **Measurement Model**: Maps hidden state to observed bet size, incorporating exponential scaling to handle aggression spikes.
- **Unscented Kalman Filter (UKF)**:
  - Generates sigma points around the current state, propagates them through non-linear models, and fuses back into a posterior estimate.
  - Handles non-linear betting dynamics better than Extended Kalman Filter (no Jacobians, more stable).
  - Operates per player with per-event processing of ~50ms (predict + update + log), supporting sub-100 ms pipeline SLA.
- **Numerical Safeguards**: Automatic resets if NaNs/inf arise, capped innovation variance, and damped process/measurement parameters guard against divergence.

---

## 5. Filter Stack – From Single Bets to Coordinated Play
### 5.1 Warm-Up & History Management
- Require 5 observed actions before judging a player, preventing cold-start false alarms.
- Keep rolling deques (length 20) of residuals and bet sizes; cap outliers to 5× median residual or 3× median bet so one wild hand cannot desensitise thresholds.

### 5.2 UKF Residual Anomaly
- Compute absolute innovation residual for each tracked action (`bet`, `raise`, `call`).
- Adaptive threshold = `max(3.5σ, 10% of avg bet, MIN_THRESHOLD_BASE)` using median/IQR (robust to outliers).
- When residual > threshold, log a `high_residual` anomaly; provides per-player deviation signal.

### 5.3 Absolute Bet Spike Detector
- Detects economic significance even when UKF expects an aggressive bet.
- Threshold = max(default $40, 2× median bet, 2× 75th percentile) but capped by recent 90th percentile + buffer.
- Flags `large_bet` anomalies to highlight chip dumps/coordinated over-betting.

### 5.4 Combined Large Bet + Residual
- If both detectors trigger simultaneously, we label `large_bet_high_residual`, a strong indicator of scripted collusion.

### 5.5 Action Sequence Tracker
- Every event is stored in `recent_actions`: timestamp, player, action.
- Supports detection of suspicious choreography: immediate raises (within 2 seconds), back-to-back raises, and other "we planned this" patterns.

---

## 6. Multi-Layer Collusion Detection – How We Confirm the Play
Implemented in `src/anomaly_logger.py` and backed by `src/collusion_detector.py`.

The system employs a four-layer validation approach, with each layer filtering out false positives:

1. **Minimum Bet Size Filter**  
   - Ignores anomalies under $20, ensuring alerts focus on pots worth chasing.

2. **Bet Size Matching**  
   - Requires partners' bets to be within 8% (configurable), catching mirroring behaviour such as $100 / $104 raises.

3. **Suspicious Sequence Filter**  
   - Looks for immediate or chained raises (`bet → raise` within 2s, `raise → raise`) that indicate coordination; eliminates normal call/check rhythms.

4. **Significant Anomaly Filter**  
   - Demands at least one player show a `large_bet` anomaly. Partner must show either a `large_bet` or a residual >1.5× threshold.

Additional detection features:

5. **Temporal Windowing**  
   - Standard window: 6 s; tight window: 1 s. Alerts are tagged `tight` when moves land <1 s apart—classic covert signalling.

6. **Correlation Tracker**  
   - Maintains joint anomaly counts per player pair over the last 10 hands.  
   - Pairs above 30% joint anomaly rate (with ≥5 shared hands) are escalated for analyst review, giving historical context.

7. **Alert Output**  
   - JSON payload includes players, bets, residuals, sequence tags, sync level, and historical correlation scores.  
   - Analysts receive a pre-vetted, high-confidence "playbook" of the suspected colluders.

---

## 7. Detection Walkthrough (Table 4 Scenario)
1. **Setup**: Producer streams `table_4.txt`—tight collusion case.  
2. **Early Hands**: Warm-up collects each player’s baseline aggression. No alerts yet.  
3. **Trigger Hand**: Player A places a $160 raise (large bet + high residual). Player B mirrors with $156 within 0.7 s.  
4. **Filter Response**:
   - Residual filter flags both players (>3.5σ).  
   - Absolute detector fires (`large_bet` type).  
   - Action tracker records `raise → raise` within 1 s.  
   - Collusion validator checks all four layers, qualifies as `tight` sync.  
5. **Alert**: `logs/table_4.log` entry summarises both players, matched bets, timeline, residuals, and correlation score.  
6. **Follow-Up**: `run_detection.sh` tallies this as 1/1 detection; analysts can open logs for context or replay the hand.

---

## 8. Performance & Validation
### 8.1 KPI Summary
| Metric | Current | Baseline | Notes |
|--------|---------|----------|-------|
| Precision | 92–100% | 7% | Driven by multi-layer validation |
| False Positive Rate | 3–4% | 93% | Filters enforce economic + behavioural significance |
| Recall (planted cases) | 91.7% (11/12) | N/A | One miss attributed to edge-case non-economic collusion |
| Latency | ~50ms per event | N/A | Predict + update + log; <100ms end-to-end with Kafka |

### 8.2 Testing Stack
- `scripts/run_detection.sh`: Replays anomaly logs, compares against ground truth, and prints detection scorecard.
- `tests/test_filters.py`: Unit-level coverage for UKF math and thresholding.
- Regression methodology: Stage multiple table scenarios (standard play, loose players, tight colluders) and monitor precision/recall.

---

## 9. Operations & Tooling
- **Demo**: `./scripts/run_local.sh` handles environment setup, Kafka boot, producer/consumer launch, and displays alerts live.
- **Configuration**: Tunables live in `src/config.py`—everything from sigma multipliers to collusion windows.
- **Logging & Monitoring**: Each table outputs a dedicated log file (`logs/table_{id}.log`); ready for ingestion into ELK/Splunk/Datadog dashboards.
- **Documentation Hub**: `poker-pipeline/docs/` covers architecture, configuration, troubleshooting, and algorithm specifics.

---

## 10. Extensibility & Research Backbone
- **AI Documentation (`docs/ai/`)**:
  - Algorithm quick reference, recommended next steps, and alternative model reviews.
  - Provides talking points on supervised vs unsupervised upgrades, deep learning options, and hybrid strategies.
- **Research Papers (`docs/papers/`)**:
  - State-space Gaussian processes and deep time-series anomaly surveys underpin our approach.
- **Modularity**:
  - Filters live under `src/filters/`; easy to plug in Isolation Forest, LSTM autoencoders, or graph-based models.
  - `event_processor.py` orchestrates data flow, making component swaps straightforward.

---

## 11. Roadmap – What’s Next
1. **Short Term (1–2 weeks)**  
   - Ship Isolation Forest hybrid for +5–7 % accuracy without labels.  
   - Enhance correlation tracker with weighted time decay and seat position awareness.

2. **Mid Term (1–2 months)**  
   - Label 1k historic hands, train XGBoost/stacking ensemble for 90 %+ accuracy with ~2 ms scoring.  
   - Integrate alert dashboard (web or BI tool) pulling from anomaly logs.

3. **Long Term (Quarter+)**  
   - Develop graph-based collusion network scoring (community detection).  
   - Explore LSTM autoencoders or Transformer-based streaming models for richer behavioural signatures.  
   - Productionise model serving with feature stores, alert routing, and analyst feedback loop.

---

## 12. Talking Points & Q&A Cheat Sheet
**“How do you avoid false positives?”**  
  By combining statistical outliers with economic thresholds and behavioural rules; all four layers must agree before we shout collusion.

**“Can this run live?”**  
  Yes. Kafka buffering + <100 ms processing keeps pace with real-time tables. Docker scripts make deployment repeatable.

**“What if colluders change tactics?”**  
  Adaptive thresholds, correlation tracking, and extensible filter modules let us tune quickly. Roadmap includes continuous-learning models.

**"How do analysts action an alert?"**  
  JSON logs contain players, amounts, timing, and rationale. Easy to replay the hand or integrate with case management tools.

---

## 13. Call to Action
- Schedule a live demo (`./scripts/run_local.sh`) to watch the pipeline catch planted collusion.
- Integrate `logs/table_*.log` files with your monitoring stack and kick off a pilot review cycle.
- Prioritise the hybrid Isolation Forest sprint, aligning product, data science, and compliance teams on success metrics.
