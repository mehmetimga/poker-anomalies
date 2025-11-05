"""
Pytest-based checks for filter implementations and anomaly logging helpers.
"""

import sys
from pathlib import Path

import numpy as np

# Ensure src package is importable when tests run directly
TESTS_DIR = Path(__file__).resolve().parent
PROJECT_SRC = TESTS_DIR.parent / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from src.anomaly_logger import AnomalyLogger  # noqa: E402
from src.filters import (  # noqa: E402
    KalmanFilter,
    PokerUKF,
    SimpleKalmanFilter,
    UnscentedKalmanFilter,
)
from src.models import (  # noqa: E402
    measurement_model,
    process_model,
    simple_measurement_model,
    simple_process_model,
)


def test_simple_kalman_converges_to_constant_measurements():
    kf = SimpleKalmanFilter(process_variance=0.05, measurement_variance=0.5)
    measurements = [5.0] * 12

    for value in measurements:
        kf.predict()
        kf.update(value)

    assert abs(kf.get_estimate() - 5.0) < 0.5


def test_kalman_filter_tracks_linear_trend():
    kf = KalmanFilter(dt=1.0, process_variance=0.1, measurement_variance=0.5)
    positions = np.arange(10, dtype=float)

    for position in positions:
        kf.predict()
        kf.update(position)

    position_estimate, velocity_estimate = kf.get_state()
    assert abs(position_estimate - positions[-1]) < 0.6
    assert abs(velocity_estimate - 1.0) < 0.6


def test_unscented_kalman_filter_tracks_linear_motion():
    ukf = UnscentedKalmanFilter(n=2, alpha=0.5, beta=2.0, kappa=0.0)
    ukf.set_covariances(Q=np.eye(2) * 0.01, R=np.array([[0.1]]))
    ukf.x = np.array([[0.0], [1.0]])
    ukf.P = np.eye(2)

    for step in range(1, 6):
        ukf.predict(simple_process_model, dt=1.0)
        measurement = float(step)
        ukf.update(measurement, simple_measurement_model)

    state = ukf.get_state()
    assert abs(state[0] - 5.0) < 0.6
    assert abs(state[1] - 1.0) < 0.6


def test_poker_ukf_generates_large_residual_for_big_bet():
    ukf = PokerUKF(
        player_id="P1", process_model=process_model, measurement_model=measurement_model
    )

    timestamp = 1_000.0
    for amount in [20.0, 22.0, 24.0, 26.0, 28.0, 30.0]:
        event = {
            "timestamp": timestamp,
            "player_id": "P1",
            "action": "bet",
            "amount": amount,
            "table_id": 1,
        }
        ukf.process_event(event)
        timestamp += 1.0

    assert ukf.is_warm_up_complete()
    baseline_threshold = ukf.get_adaptive_threshold()
    baseline_absolute = ukf.get_absolute_bet_threshold()

    anomaly_event = {
        "timestamp": timestamp,
        "player_id": "P1",
        "action": "bet",
        "amount": 200.0,
        "table_id": 1,
    }
    result = ukf.process_event(anomaly_event)

    assert result["residual"] > baseline_threshold
    assert baseline_absolute < anomaly_event["amount"]


def test_anomaly_logger_writes_log_entry(tmp_path: Path):
    logger = AnomalyLogger(log_dir=str(tmp_path), console_output=False)
    event = {
        "timestamp": 1234.0,
        "player_id": "P1",
        "table_id": 7,
        "action": "bet",
        "amount": 42.0,
    }

    logger.log_anomaly(event, residual=10.0, threshold=5.0, anomaly_type="high_residual")

    stats = logger.get_statistics()
    assert stats["total_anomalies"] == 1
    log_path = tmp_path / "table_7.log"
    assert log_path.exists()
    assert log_path.read_text().strip() != ""
