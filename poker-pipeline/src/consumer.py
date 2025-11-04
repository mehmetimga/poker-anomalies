"""
Kafka consumer with UKF-based anomaly detection for poker collusion.
Processes streaming poker events and flags suspicious betting patterns.
"""

import json
import sys
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import time

from src.filters import PokerUKF
from src.models import process_model, measurement_model
from src.anomaly_logger import AnomalyLogger, CollusionDetector


def create_consumer(
    topic="poker-actions",
    bootstrap_servers="localhost:9092",
    max_retries=10,
    retry_delay=2,
):
    """
    Create Kafka consumer with retry logic.

    Parameters:
        topic: Kafka topic to subscribe to
        bootstrap_servers: Kafka broker address
        max_retries: Maximum connection attempts
        retry_delay: Seconds between retries

    Returns:
        KafkaConsumer instance
    """
    for attempt in range(max_retries):
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id="poker-anomaly-detector",
            )
            print(f"✓ Connected to Kafka at {bootstrap_servers}")
            print(f"✓ Subscribed to topic: {topic}")
            return consumer
        except NoBrokersAvailable:
            if attempt < max_retries - 1:
                print(
                    f"⚠️  Kafka not available, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)
            else:
                print(f"✗ Failed to connect to Kafka after {max_retries} attempts")
                raise


def consume_and_detect(
    topic="poker-actions",
    bootstrap_servers="localhost:9092",
    log_file="logs/anomalies.log",
):
    """
    Consume poker events from Kafka and detect anomalies.

    Parameters:
        topic: Kafka topic name
        bootstrap_servers: Kafka broker address
        log_file: Path to anomaly log file
    """
    print("=" * 60)
    print("POKER ANOMALY DETECTION PIPELINE")
    print("=" * 60)
    print(f"Topic: {topic}")
    print(f"Kafka: {bootstrap_servers}")
    print(f"Log file: {log_file}")
    print("=" * 60)
    print("\nInitializing...")

    # Create consumer
    consumer = create_consumer(topic, bootstrap_servers)

    # Initialize anomaly logger
    anomaly_logger = AnomalyLogger(log_file=log_file, console_output=True)

    # Initialize collusion detector
    collusion_detector = CollusionDetector()

    # Dictionary of player filters {player_id: PokerUKF}
    player_filters = {}

    # Statistics
    events_processed = 0
    anomalies_detected = 0
    start_time = time.time()

    print("✓ Ready to process events\n")
    print("-" * 60)

    try:
        for message in consumer:
            event = message.value

            # Check for end signal
            if event.get("type") == "END_STREAM":
                print("\n" + "-" * 60)
                print("✓ Received END_STREAM signal")
                break

            # Process event
            player_id = event["player_id"]

            # Initialize player filter if new
            if player_id not in player_filters:
                player_filters[player_id] = PokerUKF(
                    player_id=player_id,
                    process_model=process_model,
                    measurement_model=measurement_model,
                )
                print(f"✓ Initialized filter for player {player_id}")

            # Get player's filter
            player_ukf = player_filters[player_id]

            # Process event through UKF
            result = player_ukf.process_event(event)

            events_processed += 1

            # Get adaptive threshold for this player
            threshold = player_ukf.get_adaptive_threshold(default_std=2.0)

            # Check for anomaly
            is_anomaly = anomaly_logger.check_anomaly(result["residual"], threshold)

            # Print processing info
            status = "⚠️ ANOMALY" if is_anomaly else "✓"
            print(
                f"{status} Player {player_id}: {event['action']:6s} ${event.get('amount', 0):6.2f} | "
                f"Est: ${result['estimate']:6.2f} | Residual: {result['residual']:6.2f} | "
                f"Threshold: {threshold:.2f}"
            )

            # Log if anomaly detected
            if is_anomaly:
                anomaly_logger.log_anomaly(
                    event=event,
                    residual=result["residual"],
                    threshold=threshold,
                    anomaly_type="high_residual",
                )
                anomalies_detected += 1

        # Print final summary
        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Events processed: {events_processed}")
        print(f"Anomalies detected: {anomalies_detected}")
        print(f"Players tracked: {len(player_filters)}")
        print(f"Time elapsed: {elapsed_time:.2f}s")
        print(f"Events/sec: {events_processed/elapsed_time:.2f}")
        print("=" * 60)

        # Print anomaly summary
        anomaly_logger.print_summary()

        # Print player statistics
        print("\nPLAYER STATISTICS")
        print("-" * 60)
        for player_id, ukf in sorted(player_filters.items()):
            stats = ukf.get_statistics()
            print(f"\nPlayer {player_id}:")
            print(
                f"  State: position={stats['state'][0]:.2f}, velocity={stats['state'][1]:.2f}"
            )
            print(f"  Avg bet: ${stats['avg_bet']:.2f}")
            print(f"  Std residual: {stats['std_residual']:.2f}")
            print(f"  Hands played: {len(stats['bet_history'])}")

        # Check for suspicious pairs
        suspicious_pairs = collusion_detector.get_suspicious_pairs(threshold=0.3)
        if suspicious_pairs:
            print("\n" + "=" * 60)
            print("SUSPICIOUS PLAYER PAIRS")
            print("=" * 60)
            for pair_info in suspicious_pairs:
                print(f"Players: {pair_info['players'][0]} & {pair_info['players'][1]}")
                print(f"  Correlation: {pair_info['correlation']:.2%}")
                print(
                    f"  Joint anomalies: {pair_info['joint_anomalies']}/{pair_info['total_hands']}"
                )
                print()

    except KeyboardInterrupt:
        print("\n⚠️  Consumer interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        consumer.close()
        print("\n✓ Consumer closed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Kafka consumer for poker anomaly detection"
    )
    parser.add_argument(
        "--topic",
        default="poker-actions",
        help="Kafka topic name (default: poker-actions)",
    )
    parser.add_argument(
        "--kafka",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )
    parser.add_argument(
        "--log",
        default="logs/anomalies.log",
        help="Anomaly log file path (default: logs/anomalies.log)",
    )

    args = parser.parse_args()

    consume_and_detect(
        topic=args.topic, bootstrap_servers=args.kafka, log_file=args.log
    )
