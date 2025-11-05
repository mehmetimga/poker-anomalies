#!/usr/bin/env python3
"""
Analyze detection rate of collusion patterns in poker anomaly detection system.
Compares planted anomalies in data files with detected patterns in log files.
"""

import json
import re
from pathlib import Path
from collections import defaultdict


def parse_data_anomalies(data_file):
    """Extract anomalies from data file comments."""
    anomalies = []
    with open(data_file, "r") as f:
        content = f.read()

    # Find all hand comments with ANOMALY
    pattern = r"# Hand (\d+) - ANOMALY: (P\d+) and (P\d+) synchronized"
    matches = re.finditer(pattern, content)

    for match in matches:
        hand_num = int(match.group(1))
        player1 = match.group(2)
        player2 = match.group(3)
        anomalies.append(
            {
                "hand": hand_num,
                "players": sorted([player1, player2]),
                "description": match.group(0),
            }
        )

    return anomalies


def parse_log_detections(log_file):
    """Extract collusion pattern detections from log file."""
    detections = []
    with open(log_file, "r") as f:
        for line in f:
            if "collusion_pattern" in line:
                # Extract JSON from log line
                json_start = line.find("{")
                if json_start != -1:
                    try:
                        data = json.loads(line[json_start:])
                        if data.get("type") == "collusion_pattern":
                            detections.append(
                                {
                                    "timestamp": data["timestamp"],
                                    "players": sorted(data["players"]),
                                    "num_players": data["num_players"],
                                    "sync_level": data.get("sync_level", "unknown"),
                                    "details": data.get("details", ""),
                                }
                            )
                    except json.JSONDecodeError:
                        continue

    return detections


def map_timestamp_to_hand(data_file):
    """Create mapping of timestamps to hand numbers."""
    timestamp_to_hand = {}
    current_hand = None

    with open(data_file, "r") as f:
        for line in f:
            # Check for hand marker
            hand_match = re.match(r"# Hand (\d+)", line)
            if hand_match:
                current_hand = int(hand_match.group(1))
                continue

            # Parse data line
            if not line.startswith("#") and line.strip() and "|" in line:
                parts = line.strip().split("|")
                if len(parts) >= 2:
                    try:
                        timestamp = float(parts[0])
                        if current_hand is not None:
                            timestamp_to_hand[timestamp] = current_hand
                    except ValueError:
                        continue

    return timestamp_to_hand


def analyze_table(table_num):
    """Analyze detection rate for a single table."""
    data_file = Path(f"data/table_{table_num}.txt")
    log_file = Path(f"logs/table_{table_num}.log")

    if not data_file.exists() or not log_file.exists():
        return None

    anomalies = parse_data_anomalies(data_file)
    detections = parse_log_detections(log_file)
    timestamp_to_hand = map_timestamp_to_hand(data_file)

    # Map detections to hands
    detected_hands = set()
    for detection in detections:
        timestamp = detection["timestamp"]
        # Find closest hand (detection timestamp might be slightly offset)
        closest_hand = None
        min_diff = float("inf")
        for ts, hand in timestamp_to_hand.items():
            diff = abs(ts - timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_hand = hand

        if closest_hand is not None and min_diff < 10:  # Within 10 seconds
            detected_hands.add((closest_hand, tuple(detection["players"])))

    # Check which anomalies were detected
    planted_hands = {(a["hand"], tuple(a["players"])) for a in anomalies}
    detected = planted_hands & detected_hands
    missed = planted_hands - detected_hands

    return {
        "table": table_num,
        "planted": len(anomalies),
        "detected": len(detected),
        "missed": len(missed),
        "anomalies": anomalies,
        "detections": detections,
        "detected_items": detected,
        "missed_items": missed,
    }


def main():
    """Analyze all tables and compute overall detection rate."""
    print("=" * 80)
    print("POKER ANOMALY DETECTION RATE ANALYSIS")
    print("=" * 80)
    print()

    total_planted = 0
    total_detected = 0
    table_results = []

    for table_num in range(1, 5):
        result = analyze_table(table_num)
        if result:
            table_results.append(result)
            total_planted += result["planted"]
            total_detected += result["detected"]

            print(f"TABLE {table_num}")
            print("-" * 80)
            print(f"Planted Anomalies: {result['planted']}")

            for anomaly in result["anomalies"]:
                status = (
                    "✓ DETECTED"
                    if (anomaly["hand"], tuple(anomaly["players"]))
                    in result["detected_items"]
                    else "✗ MISSED"
                )
                print(
                    f"  Hand {anomaly['hand']:2d}: {anomaly['players'][0]} & {anomaly['players'][1]} - {status}"
                )

            print(
                f"\nDetected: {result['detected']}/{result['planted']} = {result['detected']/result['planted']*100:.1f}%"
            )

            if result["missed"]:
                print(f"Missed Anomalies:")
                for hand, players in result["missed_items"]:
                    print(f"  Hand {hand}: {players[0]} & {players[1]}")

            print()

    print("=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"Total Planted Anomalies: {total_planted}")
    print(f"Total Detected: {total_detected}")
    print(f"Total Missed: {total_planted - total_detected}")
    print()
    print(
        f"DETECTION RATE: {total_detected}/{total_planted} = {total_detected/total_planted*100:.1f}%"
    )
    print("=" * 80)

    # Additional statistics
    print()
    print("DETECTION STATISTICS BY TABLE")
    print("-" * 80)
    for result in table_results:
        rate = result["detected"] / result["planted"] * 100
        bar_length = int(rate / 2)  # 50 chars = 100%
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"Table {result['table']}: {bar} {rate:5.1f}%")

    print()


if __name__ == "__main__":
    main()
