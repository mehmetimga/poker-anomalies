"""
Kafka consumer with UKF-based anomaly detection for poker collusion.
Processes streaming poker events and flags suspicious betting patterns.
"""

import os

from src.config import (
    KAFKA_DEFAULT_BOOTSTRAP_SERVERS,
    KAFKA_DEFAULT_TOPIC,
    DEFAULT_LOG_DIR,
    SUSPICIOUS_PAIR_THRESHOLD,
)
from src.kafka_utils import create_consumer_with_retry
from src.models import process_model, measurement_model
from src.anomaly_logger import AnomalyLogger
from src.collusion_detector import CollusionDetector
from src.event_processor import EventProcessor


def consume_and_detect(
    topic=KAFKA_DEFAULT_TOPIC,
    bootstrap_servers=KAFKA_DEFAULT_BOOTSTRAP_SERVERS,
    log_dir=DEFAULT_LOG_DIR,
):
    """
    Consume poker events from Kafka and detect anomalies.

    Parameters:
        topic: Kafka topic name
        bootstrap_servers: Kafka broker address
        log_dir: Directory for log files (default: logs/)
                 Each table will have its own log file: table_{table_id}.log
    """
    print("=" * 60)
    print("POKER ANOMALY DETECTION PIPELINE")
    print("=" * 60)
    print(f"Topic: {topic}")
    print(f"Kafka: {bootstrap_servers}")
    print(f"Log directory: {log_dir} (per-table logs: table_*.log)")
    print("=" * 60)
    print("\nInitializing...")

    # Create consumer
    consumer = create_consumer_with_retry(topic, bootstrap_servers)

    # Initialize collusion detector
    collusion_detector = CollusionDetector()

    # Initialize anomaly logger with collusion detector
    anomaly_logger = AnomalyLogger(
        log_dir=log_dir,
        console_output=True,
        collusion_detector=collusion_detector,
    )

    # Initialize event processor
    event_processor = EventProcessor(anomaly_logger, collusion_detector)

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
            processing_result = event_processor.process_event(
                event, process_model, measurement_model
            )

            # Print new filter initialization message if needed
            if processing_result["is_new_filter"]:
                player_id = event["player_id"]
                table_id = event["table_id"]
                print(
                    f"✓ Initialized filter for player {player_id} at table {table_id}"
                )

            # Print processing info
            player_id = event["player_id"]
            result = processing_result["result"]
            threshold = processing_result["threshold"]
            warm_up_complete = processing_result["warm_up_complete"]
            is_anomaly = processing_result["is_anomaly"]

            status = "⚠️ ANOMALY" if is_anomaly else "✓"
            warm_up_status = "" if warm_up_complete else " [WARM-UP]"
            print(
                f"{status} Player {player_id}: {event['action']:6s} ${event.get('amount', 0):6.2f} | "
                f"Est: ${result['estimate']:6.2f} | Residual: {result['residual']:6.2f} | "
                f"Threshold: {threshold:.2f}{warm_up_status}"
            )

        # Print final summary
        stats = event_processor.get_statistics()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED")
        print("=" * 60)
        print(f"Events processed: {stats['events_processed']}")
        print(f"Anomalies detected: {stats['anomalies_detected']}")
        print(f"Players tracked: {stats['players_tracked']}")
        print(f"Time elapsed: {stats['elapsed_time']:.2f}s")
        print(f"Events/sec: {stats['events_per_sec']:.2f}")

        existing_logs = []
        for table_id in sorted(event_processor.active_players_by_table.keys()):
            log_file = os.path.join(log_dir, f"table_{table_id}.log")
            if os.path.exists(log_file):
                existing_logs.append(log_file)

        if existing_logs:
            print("\nLog files created:")
            for log_file in existing_logs:
                print(f"  - {log_file}")
        else:
            print("\nNo anomaly log files were created.")
        print("=" * 60)

        # Print anomaly summary
        anomaly_logger.print_summary()

        # Print player statistics (grouped by table)
        print("\nPLAYER STATISTICS")
        print("-" * 60)

        by_table = event_processor.get_player_statistics_by_table()

        # Print stats per table
        for table_id in sorted(by_table.keys()):
            print(f"\nTable {table_id}:")
            for player_id, stats_dict in sorted(by_table[table_id]):
                print(f"\n  Player {player_id}:")
                print(
                    f"    State: position={stats_dict['state'][0]:.2f}, velocity={stats_dict['state'][1]:.2f}"
                )
                print(f"    Avg bet: ${stats_dict['avg_bet']:.2f}")
                print(f"    Std residual: {stats_dict['std_residual']:.2f}")
                print(f"    Hands played: {len(stats_dict['bet_history'])}")

        # Check for suspicious pairs
        suspicious_pairs = collusion_detector.get_suspicious_pairs(
            threshold=SUSPICIOUS_PAIR_THRESHOLD
        )
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
        default=KAFKA_DEFAULT_TOPIC,
        help=f"Kafka topic name (default: {KAFKA_DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--kafka",
        default=KAFKA_DEFAULT_BOOTSTRAP_SERVERS,
        help=f"Kafka bootstrap servers (default: {KAFKA_DEFAULT_BOOTSTRAP_SERVERS})",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help=f"Directory for log files (default: {DEFAULT_LOG_DIR}/). Each table will have its own log file: table_{{table_id}}.log",
    )

    args = parser.parse_args()

    consume_and_detect(
        topic=args.topic,
        bootstrap_servers=args.kafka,
        log_dir=args.log_dir,
    )
