"""
Kafka utility functions for connection management with retry logic.
"""

import json
import time
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable
from src.config import (
    KAFKA_MAX_RETRIES,
    KAFKA_RETRY_DELAY,
    PRODUCER_ACKS,
    PRODUCER_RETRIES,
    CONSUMER_GROUP_ID,
    CONSUMER_AUTO_OFFSET_RESET,
    CONSUMER_ENABLE_AUTO_COMMIT,
)


def create_producer_with_retry(
    bootstrap_servers="localhost:9092",
    max_retries=KAFKA_MAX_RETRIES,
    retry_delay=KAFKA_RETRY_DELAY,
):
    """
    Create Kafka producer with retry logic.

    Parameters:
        bootstrap_servers: Kafka broker address
        max_retries: Maximum connection attempts
        retry_delay: Seconds between retries

    Returns:
        KafkaProducer instance

    Raises:
        NoBrokersAvailable: If connection fails after all retries
    """
    for attempt in range(max_retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                acks=PRODUCER_ACKS,
                retries=PRODUCER_RETRIES,
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


def create_consumer_with_retry(
    topic="poker-actions",
    bootstrap_servers="localhost:9092",
    max_retries=KAFKA_MAX_RETRIES,
    retry_delay=KAFKA_RETRY_DELAY,
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

    Raises:
        NoBrokersAvailable: If connection fails after all retries
    """
    for attempt in range(max_retries):
        try:
            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset=CONSUMER_AUTO_OFFSET_RESET,
                enable_auto_commit=CONSUMER_ENABLE_AUTO_COMMIT,
                group_id=CONSUMER_GROUP_ID,
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
