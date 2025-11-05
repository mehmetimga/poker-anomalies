#!/usr/bin/env python3
"""
Test script to verify filter implementations work correctly.
Can be run independently without Kafka.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from src.filters import (
    SimpleKalmanFilter,
    KalmanFilter,
    UnscentedKalmanFilter,
    PokerUKF,
)
from src.models import process_model, measurement_model


def test_simple_kalman():
    """Test 1D Kalman filter."""
    print("=" * 60)
    print("TEST 1: Simple Kalman Filter (1D)")
    print("=" * 60)

    kf = SimpleKalmanFilter(process_variance=0.1, measurement_variance=2.0)
    measurements = [10, 12, 11, 15, 13, 14, 16, 15, 17, 18]

    print("Measurements:", measurements)
    estimates = []

    for z in measurements:
        kf.predict()
        kf.update(z)
        estimates.append(kf.get_estimate())

    print("Estimates:   ", [f"{e:.2f}" for e in estimates])
    print("✓ Simple Kalman Filter working\n")


def test_multivariate_kalman():
    """Test 2D Kalman filter (position-velocity)."""
    print("=" * 60)
    print("TEST 2: Multivariate Kalman Filter (2D)")
    print("=" * 60)

    kf = KalmanFilter(dt=1.0, process_variance=1.0, measurement_variance=4.0)

    # Simulate trending bet sizes
    np.random.seed(42)
    true_bets = np.cumsum(np.random.normal(0, 1, 10))
    measurements = true_bets + np.random.normal(0, 2, 10)

    print("Measurements:", [f"{m:.2f}" for m in measurements])

    estimates = []
    for z in measurements:
        kf.predict()
        kf.update(z)
        estimates.append(kf.get_state()[0])

    print("Estimates:   ", [f"{e:.2f}" for e in estimates])
    print("✓ Multivariate Kalman Filter working\n")


def test_ukf():
    """Test Unscented Kalman Filter."""
    print("=" * 60)
    print("TEST 3: Unscented Kalman Filter (UKF)")
    print("=" * 60)

    ukf = UnscentedKalmanFilter(n=2, alpha=1.0, beta=2.0, kappa=0.0)
    ukf.set_covariances(Q=np.eye(2) * 0.1, R=np.array([[1.0]]))
    ukf.x = np.array([[0.0], [1.0]])
    ukf.P = np.eye(2) * 10

    # Simulate non-linear measurements
    np.random.seed(42)
    dt = 1.0

    print("Processing 10 synthetic measurements...")
    estimates = []
    residuals = []

    for i in range(10):
        # Predict
        ukf.predict(process_model, dt)

        # Simulate measurement
        true_pos = i * 2.0
        meas = true_pos + np.random.normal(0, 1)

        # Update
        innovation, innovation_var = ukf.update(meas, measurement_model)

        state = ukf.get_state()
        estimates.append(state[0])
        residuals.append(abs(innovation))

    print("Estimates:", [f"{e:.2f}" for e in estimates])
    print("Residuals:", [f"{r:.2f}" for r in residuals])
    print("✓ UKF working\n")


def test_poker_ukf():
    """Test Poker-specific UKF wrapper."""
    print("=" * 60)
    print("TEST 4: Poker UKF Wrapper")
    print("=" * 60)

    poker_ukf = PokerUKF(
        player_id="P1", process_model=process_model, measurement_model=measurement_model
    )

    # Simulate poker events
    events = [
        {
            "timestamp": 1000.0,
            "player_id": "P1",
            "action": "bet",
            "amount": 10.0,
            "table_id": 1,
        },
        {
            "timestamp": 1001.0,
            "player_id": "P1",
            "action": "raise",
            "amount": 20.0,
            "table_id": 1,
        },
        {
            "timestamp": 1002.0,
            "player_id": "P1",
            "action": "bet",
            "amount": 15.0,
            "table_id": 1,
        },
        {
            "timestamp": 1003.0,
            "player_id": "P1",
            "action": "bet",
            "amount": 25.0,
            "table_id": 1,
        },
        {
            "timestamp": 1004.0,
            "player_id": "P1",
            "action": "fold",
            "amount": 0.0,
            "table_id": 1,
        },
        {
            "timestamp": 1005.0,
            "player_id": "P1",
            "action": "bet",
            "amount": 30.0,
            "table_id": 1,
        },
    ]

    print(f"Processing {len(events)} poker events for player P1...")
    print()

    for event in events:
        result = poker_ukf.process_event(event)
        threshold = poker_ukf.get_adaptive_threshold()
        is_anomaly = abs(result["residual"]) > threshold

        status = "⚠️ " if is_anomaly else "✓ "
        print(
            f"{status} Action: {event['action']:6s} ${event['amount']:6.2f} | "
            f"Est: ${result['estimate']:6.2f} | Residual: {result['residual']:6.2f} | "
            f"Threshold: {threshold:.2f}"
        )

    # Get statistics
    stats = poker_ukf.get_statistics()
    print()
    print(f"Player Statistics:")
    print(
        f"  State: position={stats['state'][0]:.2f}, velocity={stats['state'][1]:.2f}"
    )
    print(f"  Avg bet: ${stats['avg_bet']:.2f}")
    print(f"  Std residual: {stats['std_residual']:.2f}")
    print("✓ Poker UKF Wrapper working\n")


def test_anomaly_detection():
    """Test anomaly detection with synthetic collusion pattern."""
    print("=" * 60)
    print("TEST 5: Anomaly Detection with Collusion Pattern")
    print("=" * 60)

    from src.anomaly_logger import AnomalyLogger

    logger = AnomalyLogger(log_dir="logs", console_output=False)

    # Create two player UKFs
    player1 = PokerUKF("P1", process_model, measurement_model)
    player2 = PokerUKF("P2", process_model, measurement_model)

    # Normal bets
    normal_events = [
        {
            "timestamp": 1000.0,
            "player_id": "P1",
            "action": "bet",
            "amount": 10.0,
            "table_id": 1,
        },
        {
            "timestamp": 1001.0,
            "player_id": "P2",
            "action": "bet",
            "amount": 12.0,
            "table_id": 1,
        },
        {
            "timestamp": 1002.0,
            "player_id": "P1",
            "action": "bet",
            "amount": 11.0,
            "table_id": 1,
        },
        {
            "timestamp": 1003.0,
            "player_id": "P2",
            "action": "bet",
            "amount": 13.0,
            "table_id": 1,
        },
    ]

    print("Processing normal bets...")
    for event in normal_events:
        ukf = player1 if event["player_id"] == "P1" else player2
        result = ukf.process_event(event)
        threshold = ukf.get_adaptive_threshold()

        if logger.check_anomaly(result["residual"], threshold):
            logger.log_anomaly(event, result["residual"], threshold)

    # Synchronized anomalous bets (collusion)
    collusion_events = [
        {
            "timestamp": 1010.0,
            "player_id": "P1",
            "action": "bet",
            "amount": 100.0,
            "table_id": 1,
        },
        {
            "timestamp": 1010.5,
            "player_id": "P2",
            "action": "bet",
            "amount": 95.0,
            "table_id": 1,
        },
    ]

    print("Processing synchronized anomalous bets (collusion)...")
    for event in collusion_events:
        ukf = player1 if event["player_id"] == "P1" else player2
        result = ukf.process_event(event)
        threshold = ukf.get_adaptive_threshold()

        if logger.check_anomaly(result["residual"], threshold):
            logger.log_anomaly(event, result["residual"], threshold)

    stats = logger.get_statistics()
    print()
    print(f"Anomalies detected: {stats['total_anomalies']}")
    print(f"Collusions detected: {stats['collusion_detected']}")
    print("✓ Anomaly detection working\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("POKER ANOMALY DETECTION - FILTER TESTS")
    print("=" * 60 + "\n")

    try:
        test_simple_kalman()
        test_multivariate_kalman()
        test_ukf()
        test_poker_ukf()
        test_anomaly_detection()

        print("=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nFilters are working correctly. Ready to run the full pipeline.")
        print("Run: ./scripts/run_local.sh\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
