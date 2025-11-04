"""
Kafka producer for streaming poker hand history events.
Reads hand history files and publishes events to Kafka topic.
Supports multiple table files (table_*.txt) in the data directory.
"""

import json
import time
import sys
import os
import glob
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable


def parse_hand_line(line):
    """
    Parse a line from hand history file.

    Format: timestamp|table_id|player_id|action|amount|pot

    Parameters:
        line: Raw line from file

    Returns:
        dict: Parsed event or None if invalid
    """
    # Skip comments and empty lines
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.split("|")
    if len(parts) != 6:
        return None

    try:
        ts, table, player, action, amount, pot = parts
        return {
            "timestamp": float(ts),
            "table_id": int(table),
            "player_id": player,
            "action": action,
            "amount": float(amount),
            "pot": float(pot),
        }
    except ValueError as e:
        print(f"Error parsing line: {line} - {e}")
        return None


def create_producer(bootstrap_servers="localhost:9092", max_retries=10, retry_delay=2):
    """
    Create Kafka producer with retry logic.

    Parameters:
        bootstrap_servers: Kafka broker address
        max_retries: Maximum connection attempts
        retry_delay: Seconds between retries

    Returns:
        KafkaProducer instance
    """
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks="all",
                retries=3,
            )
            print(f"✓ Connected to Kafka at {bootstrap_servers}")
            return producer
        except NoBrokersAvailable:
            if attempt < max_retries - 1:
                print(
                    f"⚠️  Kafka not available, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(retry_delay)
            else:
                print(f"✗ Failed to connect to Kafka after {max_retries} attempts")
                raise


def produce_events_from_file(
    input_file, producer, topic, delay, events_sent, events_failed
):
    """
    Read a single hand history file and produce events to Kafka.

    Parameters:
        input_file: Path to hand history file
        producer: KafkaProducer instance
        topic: Kafka topic name
        delay: Delay between events (seconds) to simulate real-time
        events_sent: Current count of events sent (will be updated)
        events_failed: Current count of events failed (will be updated)

    Returns:
        tuple: (events_sent, events_failed) - updated counts
    """
    file_events = 0
    try:
        with open(input_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                event = parse_hand_line(line)

                if event is None:
                    continue

                try:
                    # Send event to Kafka
                    future = producer.send(topic, value=event)

                    # Wait for send to complete (optional, for debugging)
                    record_metadata = future.get(timeout=10)

                    events_sent += 1
                    file_events += 1

                    # Print progress
                    if events_sent % 10 == 0:
                        print(
                            f"Sent {events_sent} events... (latest: {event['player_id']} {event['action']} ${event['amount']:.2f} from {os.path.basename(input_file)})"
                        )

                    # Simulate real-time delay
                    time.sleep(delay)

                except Exception as e:
                    events_failed += 1
                    print(
                        f"✗ Error sending event from {input_file} line {line_num}: {e}"
                    )

        print(f"✓ Processed {file_events} events from {os.path.basename(input_file)}")

    except FileNotFoundError:
        print(f"✗ Error: Input file not found: {input_file}")
        events_failed += 1
    except Exception as e:
        print(f"✗ Error reading file {input_file}: {e}")
        events_failed += 1

    return events_sent, events_failed


def produce_events(
    data_dir=None,
    input_file=None,
    topic="poker-actions",
    delay=0.5,
    bootstrap_servers="localhost:9092",
):
    """
    Read hand history files and produce events to Kafka.
    Can process either:
    - All table_*.txt files in a directory (if data_dir is provided)
    - A single file (if input_file is provided)

    Parameters:
        data_dir: Directory containing table_*.txt files (default: None)
        input_file: Path to single hand history file (default: None, overrides data_dir)
        topic: Kafka topic name
        delay: Delay between events (seconds) to simulate real-time
        bootstrap_servers: Kafka broker address
    """
    print(f"Starting Kafka Producer")
    print(f"Topic: {topic}")
    print(f"Delay: {delay}s per event")
    print("-" * 60)

    # Create producer with retry logic
    producer = create_producer(bootstrap_servers)

    events_sent = 0
    events_failed = 0

    try:
        # Determine which files to process
        files_to_process = []

        if input_file:
            # Single file mode (backward compatibility)
            if os.path.exists(input_file):
                files_to_process = [input_file]
                print(f"Processing single file: {input_file}")
            else:
                print(f"✗ Error: Input file not found: {input_file}")
                sys.exit(1)
        elif data_dir:
            # Directory mode - find all table_*.txt files
            data_path = Path(data_dir)
            if not data_path.exists():
                print(f"✗ Error: Data directory not found: {data_dir}")
                sys.exit(1)

            # Find all files matching table_*.txt pattern
            pattern = str(data_path / "table_*.txt")
            files_to_process = sorted(glob.glob(pattern))

            if not files_to_process:
                print(f"⚠️  No table_*.txt files found in {data_dir}")
                print(f"   Looking for files matching: {pattern}")
                sys.exit(1)

            print(f"Found {len(files_to_process)} table file(s):")
            for f in files_to_process:
                print(f"  - {os.path.basename(f)}")
        else:
            # Default: look in data/ directory relative to script location
            script_dir = Path(__file__).parent.parent
            default_data_dir = script_dir / "data"
            pattern = str(default_data_dir / "table_*.txt")
            files_to_process = sorted(glob.glob(pattern))

            if not files_to_process:
                print(f"✗ Error: No table_*.txt files found in {default_data_dir}")
                print(
                    f"   Please provide --input or --data-dir, or add table_*.txt files to data/"
                )
                sys.exit(1)

            print(
                f"Found {len(files_to_process)} table file(s) in default data directory:"
            )
            for f in files_to_process:
                print(f"  - {os.path.basename(f)}")

        print("-" * 60)

        # Process each file
        for input_file in files_to_process:
            events_sent, events_failed = produce_events_from_file(
                input_file, producer, topic, delay, events_sent, events_failed
            )

        # Send END_STREAM signal
        end_event = {"type": "END_STREAM", "timestamp": time.time()}
        producer.send(topic, value=end_event)
        producer.flush()

        print("\n" + "-" * 60)
        print(f"✓ Producer finished")
        print(f"  Files processed: {len(files_to_process)}")
        print(f"  Events sent: {events_sent}")
        print(f"  Events failed: {events_failed}")
        print(f"  END_STREAM signal sent")
        print("-" * 60)

    except KeyboardInterrupt:
        print("\n⚠️  Producer interrupted by user")
    finally:
        producer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Kafka producer for poker hand history. Processes all table_*.txt files in data directory by default."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input hand history file (optional, overrides --data-dir). Default: process all table_*.txt files in data/",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing table_*.txt files (default: data/ relative to project root)",
    )
    parser.add_argument(
        "--topic",
        default="poker-actions",
        help="Kafka topic name (default: poker-actions)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between events in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--kafka",
        default="localhost:9092",
        help="Kafka bootstrap servers (default: localhost:9092)",
    )

    args = parser.parse_args()

    produce_events(
        data_dir=args.data_dir,
        input_file=args.input,
        topic=args.topic,
        delay=args.delay,
        bootstrap_servers=args.kafka,
    )
