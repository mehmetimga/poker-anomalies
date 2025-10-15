"""
Kafka producer for streaming poker hand history events.
Reads hand history file and publishes events to Kafka topic.
"""
import json
import time
import sys
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
    if not line or line.startswith('#'):
        return None
    
    parts = line.split('|')
    if len(parts) != 6:
        return None
    
    try:
        ts, table, player, action, amount, pot = parts
        return {
            'timestamp': float(ts),
            'table_id': int(table),
            'player_id': player,
            'action': action,
            'amount': float(amount),
            'pot': float(pot)
        }
    except ValueError as e:
        print(f"Error parsing line: {line} - {e}")
        return None


def create_producer(bootstrap_servers='localhost:9092', max_retries=10, retry_delay=2):
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
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                acks='all',
                retries=3
            )
            print(f"✓ Connected to Kafka at {bootstrap_servers}")
            return producer
        except NoBrokersAvailable:
            if attempt < max_retries - 1:
                print(f"⚠️  Kafka not available, retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"✗ Failed to connect to Kafka after {max_retries} attempts")
                raise


def produce_events(input_file, topic='poker-actions', delay=0.5, bootstrap_servers='localhost:9092'):
    """
    Read hand history and produce events to Kafka.
    
    Parameters:
        input_file: Path to hand history file
        topic: Kafka topic name
        delay: Delay between events (seconds) to simulate real-time
        bootstrap_servers: Kafka broker address
    """
    print(f"Starting Kafka Producer")
    print(f"Input file: {input_file}")
    print(f"Topic: {topic}")
    print(f"Delay: {delay}s per event")
    print("-" * 60)
    
    # Create producer with retry logic
    producer = create_producer(bootstrap_servers)
    
    events_sent = 0
    events_failed = 0
    
    try:
        with open(input_file, 'r') as f:
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
                    
                    # Print progress
                    if events_sent % 10 == 0:
                        print(f"Sent {events_sent} events... (latest: {event['player_id']} {event['action']} ${event['amount']:.2f})")
                    
                    # Simulate real-time delay
                    time.sleep(delay)
                    
                except Exception as e:
                    events_failed += 1
                    print(f"✗ Error sending event on line {line_num}: {e}")
        
        # Send END_STREAM signal
        end_event = {'type': 'END_STREAM', 'timestamp': time.time()}
        producer.send(topic, value=end_event)
        producer.flush()
        
        print("\n" + "-" * 60)
        print(f"✓ Producer finished")
        print(f"  Events sent: {events_sent}")
        print(f"  Events failed: {events_failed}")
        print(f"  END_STREAM signal sent")
        print("-" * 60)
        
    except FileNotFoundError:
        print(f"✗ Error: Input file not found: {input_file}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Producer interrupted by user")
    finally:
        producer.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Kafka producer for poker hand history')
    parser.add_argument(
        '--input',
        default='data/sample_hand_history.txt',
        help='Input hand history file (default: data/sample_hand_history.txt)'
    )
    parser.add_argument(
        '--topic',
        default='poker-actions',
        help='Kafka topic name (default: poker-actions)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between events in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--kafka',
        default='localhost:9092',
        help='Kafka bootstrap servers (default: localhost:9092)'
    )
    
    args = parser.parse_args()
    
    produce_events(
        input_file=args.input,
        topic=args.topic,
        delay=args.delay,
        bootstrap_servers=args.kafka
    )


