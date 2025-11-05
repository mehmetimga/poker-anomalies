"""
Kafka producer for streaming poker hand history events.
Reads hand history files and publishes events to Kafka topic.
Supports multiple table files (table_*.txt) in the data directory.
"""

import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from src.config import (
    MAX_WORKERS_LIMIT,
    KAFKA_DEFAULT_BOOTSTRAP_SERVERS,
    KAFKA_DEFAULT_TOPIC,
    PRODUCER_DEFAULT_DELAY,
    PRODUCER_SEND_TIMEOUT,
    PRODUCER_PROGRESS_INTERVAL,
)
from src.kafka_utils import create_producer_with_retry
from src.file_utils import find_table_files, get_effective_workers
from src.parser import parse_hand_line


def produce_events_from_file(
    input_file,
    producer,
    topic,
    delay,
    events_counter,
    events_failed_counter,
    print_lock,
):
    """
    Read a single hand history file and produce events to Kafka.
    Thread-safe version for parallel processing.

    Parameters:
        input_file: Path to hand history file
        producer: KafkaProducer instance (thread-safe)
        topic: Kafka topic name
        delay: Delay between events (seconds) to simulate real-time
        events_counter: Thread-safe counter for events sent (list with single int)
        events_failed_counter: Thread-safe counter for events failed (list with single int)
        print_lock: Lock for thread-safe printing

    Returns:
        tuple: (file_events_sent, file_events_failed) - counts for this file only
    """
    file_events = 0
    file_failed = 0
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
                    record_metadata = future.get(timeout=PRODUCER_SEND_TIMEOUT)

                    # Thread-safe counter update
                    with print_lock:
                        events_counter[0] += 1
                        events_sent = events_counter[0]
                        file_events += 1

                        # Print progress
                        if events_sent % PRODUCER_PROGRESS_INTERVAL == 0:
                            print(
                                f"Sent {events_sent} events... (latest: {event['player_id']} {event['action']} ${event['amount']:.2f} from {os.path.basename(input_file)})"
                            )

                    # Simulate real-time delay
                    time.sleep(delay)

                except Exception as e:
                    with print_lock:
                        events_failed_counter[0] += 1
                        file_failed += 1
                        print(
                            f"✗ Error sending event from {input_file} line {line_num}: {e}"
                        )

        with print_lock:
            print(
                f"✓ Processed {file_events} events from {os.path.basename(input_file)}"
            )

    except FileNotFoundError:
        with print_lock:
            events_failed_counter[0] += 1
            file_failed += 1
            print(f"✗ Error: Input file not found: {input_file}")
    except Exception as e:
        with print_lock:
            events_failed_counter[0] += 1
            file_failed += 1
            print(f"✗ Error reading file {input_file}: {e}")

    return file_events, file_failed


def produce_events(
    data_dir=None,
    input_file=None,
    topic="poker-actions",
    delay=0.5,
    bootstrap_servers="localhost:9092",
    max_workers=1,
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
        max_workers: Maximum number of threads for parallel processing (default: 1, sequential)
                     Set to None or number of files for full parallelism
    """
    print(f"Starting Kafka Producer")
    print(f"Topic: {topic}")
    print(f"Delay: {delay}s per event")

    # Calculate effective workers (capped at MAX_WORKERS_LIMIT)
    effective_workers = get_effective_workers(max_workers, MAX_WORKERS_LIMIT)

    if effective_workers > 1:
        print(
            f"Parallel processing: {effective_workers} threads (max limit: {MAX_WORKERS_LIMIT})"
        )
        if max_workers and max_workers > MAX_WORKERS_LIMIT:
            print(
                f"  Note: Requested {max_workers} threads, capped at {MAX_WORKERS_LIMIT}"
            )
    else:
        print(f"Sequential processing")
    print("-" * 60)

    # Create producer with retry logic
    # Note: KafkaProducer is thread-safe, so we can share it across threads
    producer = create_producer_with_retry(bootstrap_servers)

    # Thread-safe counters (using lists so they're mutable across threads)
    events_counter = [0]
    events_failed_counter = [0]
    print_lock = Lock()

    try:
        # Find files to process using utility function
        files_to_process, info_message = find_table_files(data_dir, input_file)
        print(info_message)
        print("-" * 60)

        # Process files - either sequentially or in parallel
        if effective_workers > 1 and len(files_to_process) > 1:
            # Parallel processing
            # ThreadPoolExecutor automatically queues tasks when there are more files than workers
            # So if we have 6 files and 4 workers, 2 files will wait in queue until a worker is free
            if len(files_to_process) > effective_workers:
                print(
                    f"Processing {len(files_to_process)} files with {effective_workers} threads "
                    f"(files will be queued as threads become available)"
                )
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                # Submit all file processing tasks
                future_to_file = {
                    executor.submit(
                        produce_events_from_file,
                        input_file,
                        producer,
                        topic,
                        delay,
                        events_counter,
                        events_failed_counter,
                        print_lock,
                    ): input_file
                    for input_file in files_to_process
                }

                # Wait for all tasks to complete
                for future in as_completed(future_to_file):
                    file = future_to_file[future]
                    try:
                        file_events, file_failed = future.result()
                    except Exception as e:
                        with print_lock:
                            print(f"✗ File {file} generated an exception: {e}")
                            events_failed_counter[0] += 1
        else:
            # Sequential processing (backward compatible)
            for input_file in files_to_process:
                produce_events_from_file(
                    input_file,
                    producer,
                    topic,
                    delay,
                    events_counter,
                    events_failed_counter,
                    print_lock,
                )

        # Send END_STREAM signal
        end_event = {"type": "END_STREAM", "timestamp": time.time()}
        producer.send(topic, value=end_event)
        producer.flush()

        print("\n" + "-" * 60)
        print(f"✓ Producer finished")
        print(f"  Files processed: {len(files_to_process)}")
        print(f"  Events sent: {events_counter[0]}")
        print(f"  Events failed: {events_failed_counter[0]}")
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
        default=KAFKA_DEFAULT_TOPIC,
        help=f"Kafka topic name (default: {KAFKA_DEFAULT_TOPIC})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=PRODUCER_DEFAULT_DELAY,
        help=f"Delay between events in seconds (default: {PRODUCER_DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--kafka",
        default=KAFKA_DEFAULT_BOOTSTRAP_SERVERS,
        help=f"Kafka bootstrap servers (default: {KAFKA_DEFAULT_BOOTSTRAP_SERVERS})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for parallel file processing (default: 1, sequential). Set to number of files or higher for full parallelism.",
    )

    args = parser.parse_args()

    produce_events(
        data_dir=args.data_dir,
        input_file=args.input,
        topic=args.topic,
        delay=args.delay,
        bootstrap_servers=args.kafka,
        max_workers=args.threads,
    )
