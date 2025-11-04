"""
Anomaly detection and logging for poker collusion patterns.
Tracks individual player anomalies and multi-player collusion patterns.
"""

import json
import logging
from collections import defaultdict, deque
from datetime import datetime
import os


class AnomalyLogger:
    """
    Logs and detects anomalies in poker betting patterns.
    Supports both individual player anomalies and multi-player collusion detection.
    """

    def __init__(
        self,
        log_file="logs/anomalies.log",
        console_output=True,
        collusion_detector=None,
        min_bet_for_collusion=30.0,
        bet_size_similarity_threshold=0.05,
    ):
        """
        Initialize the anomaly logger.

        Parameters:
            log_file: Path to log file
            console_output: Whether to echo anomalies to console
            collusion_detector: Optional CollusionDetector instance to update
            min_bet_for_collusion: Minimum bet size (in dollars) to trigger collusion alerts.
                                   Individual anomalies are still logged regardless of bet size.
                                   Default: 30.0
            bet_size_similarity_threshold: Bet amounts must be within this percentage to be
                                           considered matching. Default: 0.05 (5%)
                                           Lower values = stricter matching (exact or very close)
        """
        self.log_file = log_file
        self.console_output = console_output
        self.collusion_detector = collusion_detector
        self.min_bet_for_collusion = min_bet_for_collusion
        self.bet_size_similarity_threshold = bet_size_similarity_threshold

        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Setup logging
        self.logger = logging.getLogger("AnomalyLogger")
        self.logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Track recent anomalies per table for collusion detection
        # Format: {table_id: deque([(timestamp, player_id, residual, bet_amount), ...])}
        self.recent_anomalies = defaultdict(lambda: deque(maxlen=10))

        # Track recent actions per table for sequence analysis
        # Format: {table_id: deque([(timestamp, player_id, action), ...])}
        self.recent_actions = defaultdict(lambda: deque(maxlen=10))

        # Track active players per table (for pair statistics)
        # Format: {table_id: set([player_id, ...])}
        self.active_players = defaultdict(set)

        # Collusion tracking window (seconds)
        self.collusion_window = 5.0
        # Tight collusion window for very synchronized bets (e.g., 0.5s apart)
        self.tight_collusion_window = 1.0

        # Statistics
        self.total_anomalies = 0
        self.collusion_detected = 0

    def track_action(self, event):
        """
        Track all actions at a table (not just anomalies) for sequence analysis.
        This should be called for every event to maintain accurate action sequence tracking.

        Parameters:
            event: Poker event dict with keys: timestamp, player_id, table_id, action
        """
        table_id = event["table_id"]
        self.recent_actions[table_id].append(
            (event["timestamp"], event["player_id"], event["action"])
        )

    def log_anomaly(self, event, residual, threshold, anomaly_type="high_residual"):
        """
        Log an individual player anomaly.

        Parameters:
            event: Poker event dict
            residual: Residual value from filter
            threshold: Threshold that was exceeded
            anomaly_type: Type of anomaly
        """
        # Build details string based on anomaly type
        if anomaly_type == "large_bet":
            details = f"Large bet detected (amount=${event.get('amount', 0):.2f}, threshold=${threshold:.2f})"
        elif anomaly_type == "large_bet_high_residual":
            details = f"Large bet with high residual (amount=${event.get('amount', 0):.2f}, residual={residual:.2f}, threshold={threshold:.2f})"
        else:
            details = f"Deviation >5σ (threshold={threshold:.2f})"

        log_entry = {
            "timestamp": event["timestamp"],
            "player_id": event["player_id"],
            "table_id": event["table_id"],
            "action": event["action"],
            "amount": event.get("amount", 0.0),
            "residual": float(residual),
            "threshold": float(threshold),
            "type": anomaly_type,
            "details": details,
        }

        # Log to file
        self.logger.info(json.dumps(log_entry))

        # Console output
        if self.console_output:
            print(
                f"⚠️  ANOMALY DETECTED: Player {event['player_id']} at table {event['table_id']}"
            )
            print(
                f"   Action: {event['action']} ${event.get('amount', 0):.2f}, Residual: {residual:.2f} (threshold: {threshold:.2f})"
            )

        # Track for collusion detection (include bet amount, anomaly type, and threshold for filtering)
        table_id = event["table_id"]
        bet_amount = float(event.get("amount", 0.0))
        self.recent_anomalies[table_id].append(
            (
                event["timestamp"],
                event["player_id"],
                residual,
                bet_amount,
                anomaly_type,
                threshold,
            )
        )

        # Track active players at this table
        self.active_players[table_id].add(event["player_id"])

        self.total_anomalies += 1

        # Update collusion detector statistics for all player pairs
        if self.collusion_detector:
            self._update_collusion_statistics(
                table_id, event["player_id"], event["timestamp"]
            )

        # Check for collusion pattern
        self._check_collusion(event["table_id"], event["timestamp"])

    def _update_collusion_statistics(
        self, table_id, current_player_id, current_timestamp
    ):
        """
        Update collusion detector statistics for player pairs.
        Compares current player with all other active players at the table.
        Tracks both cases: when both players have anomalies and when only one does.

        Parameters:
            table_id: Table identifier
            current_player_id: Player who just had an anomaly
            current_timestamp: Current event timestamp
        """
        if not self.collusion_detector:
            return

        anomalies = self.recent_anomalies[table_id]
        active_players = self.active_players[table_id]

        # Find other players who had anomalies within the collusion window
        # Note: tuple format is (timestamp, player_id, residual, bet_amount, anomaly_type, threshold)
        recent_anomalous_players = {
            player_id
            for ts, player_id, res, bet_amount, _, _ in anomalies
            if current_timestamp - ts <= self.collusion_window
            and player_id != current_player_id
        }

        # Update statistics for all active player pairs
        for other_player_id in active_players:
            if other_player_id == current_player_id:
                continue

            # Check if both players had anomalies in the window
            both_anomalous = other_player_id in recent_anomalous_players

            # Update pair statistics
            self.collusion_detector.update_pair_statistics(
                current_player_id, other_player_id, both_anomalous=both_anomalous
            )

    def _is_suspicious_action_sequence(
        self, table_id, current_timestamp, involved_players
    ):
        """
        Check if the recent action sequence is suspicious (indicating potential collusion).
        Normal sequences (bet → call → raise) are less suspicious than immediate sequences
        (bet → immediate raise, raise → raise).

        Parameters:
            table_id: Table identifier
            current_timestamp: Current event timestamp
            involved_players: Set of player IDs involved in the potential collusion

        Returns:
            tuple: (is_suspicious, sequence_info)
                - is_suspicious: bool indicating if sequence is suspicious
                - sequence_info: dict with details about the sequence
        """
        actions = list(self.recent_actions[table_id])

        if len(actions) < 2:
            # Not enough actions to analyze
            return True, {
                "reason": "insufficient_actions",
                "action_count": len(actions),
            }

        sequence_info = {
            "recent_actions": [],
            "suspicious_patterns": [],
            "normal_patterns": [],
        }

        # Get recent actions within collusion window (last 5 seconds)
        recent_actions = [
            (ts, player_id, action)
            for ts, player_id, action in actions
            if current_timestamp - ts <= self.collusion_window
            and player_id in involved_players
        ]

        if len(recent_actions) < 2:
            # Not enough actions from involved players
            return True, {
                "reason": "insufficient_player_actions",
                "action_count": len(recent_actions),
            }

        # Sort by timestamp
        recent_actions.sort(key=lambda x: x[0])

        # Extract action sequence
        action_sequence = [action for _, _, action in recent_actions]
        player_sequence = [player_id for _, player_id, _ in recent_actions]

        sequence_info["recent_actions"] = [
            {"player": p, "action": a, "timestamp": ts} for ts, p, a in recent_actions
        ]

        # Check for suspicious patterns
        is_suspicious = False

        # Pattern 1: raise → raise (both players raising consecutively)
        for i in range(len(action_sequence) - 1):
            if action_sequence[i] == "raise" and action_sequence[i + 1] == "raise":
                if player_sequence[i] != player_sequence[i + 1]:  # Different players
                    is_suspicious = True
                    sequence_info["suspicious_patterns"].append(
                        {
                            "pattern": "raise_raise",
                            "players": [player_sequence[i], player_sequence[i + 1]],
                            "time_diff": recent_actions[i + 1][0]
                            - recent_actions[i][0],
                        }
                    )

        # Pattern 2: bet → immediate raise (no call/fold in between)
        for i in range(len(action_sequence) - 1):
            if action_sequence[i] == "bet" and action_sequence[i + 1] == "raise":
                if player_sequence[i] != player_sequence[i + 1]:  # Different players
                    time_diff = recent_actions[i + 1][0] - recent_actions[i][0]
                    # Check if there are intervening actions from other players
                    intervening_actions = [
                        (ts, p, a)
                        for ts, p, a in actions
                        if recent_actions[i][0] < ts < recent_actions[i + 1][0]
                        and p not in involved_players  # Actions from other players
                    ]

                    if (
                        len(intervening_actions) == 0 and time_diff < 2.0
                    ):  # No intervening actions, < 2s apart
                        is_suspicious = True
                        sequence_info["suspicious_patterns"].append(
                            {
                                "pattern": "bet_immediate_raise",
                                "players": [player_sequence[i], player_sequence[i + 1]],
                                "time_diff": time_diff,
                            }
                        )

        # Pattern 3: Normal sequences (bet → call → raise) - less suspicious
        for i in range(len(action_sequence) - 2):
            if (
                action_sequence[i] == "bet"
                and action_sequence[i + 1] == "call"
                and action_sequence[i + 2] == "raise"
            ):
                if (
                    len(set(player_sequence[i : i + 3])) > 1
                ):  # Multiple players involved
                    sequence_info["normal_patterns"].append(
                        {
                            "pattern": "bet_call_raise",
                            "players": player_sequence[i : i + 3],
                        }
                    )
                    # If we have a normal pattern with enough actions in between, it's less suspicious
                    # But we don't immediately return False because other patterns might be suspicious

        # If we found suspicious patterns, return True
        # If we only found normal patterns and no suspicious ones, return False
        if sequence_info["suspicious_patterns"]:
            return True, sequence_info
        elif (
            sequence_info["normal_patterns"]
            and not sequence_info["suspicious_patterns"]
        ):
            # Only normal patterns found - less suspicious
            return False, sequence_info
        else:
            # Default: if we can't determine, err on the side of caution (flag as suspicious)
            return True, sequence_info

    def _filter_significant_anomalies(
        self, anomalies_list, min_residual_multiplier=2.0
    ):
        """
        Filter to only significant anomalies for collusion detection.
        Requires at least one large_bet anomaly, and others must have very high residuals.

        Parameters:
            anomalies_list: List of (timestamp, player_id, residual, bet_amount, anomaly_type, threshold) tuples
            min_residual_multiplier: Multiplier for threshold to determine "very high residual" (default: 2.0)

        Returns:
            Filtered list of significant anomalies (same format as input)
        """
        if len(anomalies_list) < 2:
            return []

        significant = []
        for anomaly in anomalies_list:
            # Handle both old format (4 elements) and new format (6 elements)
            if len(anomaly) == 6:
                ts, player_id, res, bet_amount, anomaly_type, threshold = anomaly
            elif len(anomaly) == 4:
                # Old format: (timestamp, player_id, residual, bet_amount)
                ts, player_id, res, bet_amount = anomaly
                anomaly_type = "high_residual"
                threshold = res / 5.0  # Estimate threshold (assuming 5σ)
            else:
                # Skip unknown formats
                continue

            is_large_bet = anomaly_type in ["large_bet", "large_bet_high_residual"]
            is_very_high_residual = res > (threshold * min_residual_multiplier)

            if is_large_bet or is_very_high_residual:
                significant.append(anomaly)

        # Require at least one large_bet in the group
        has_large_bet = False
        for anomaly in significant:
            if len(anomaly) == 6:
                _, _, _, _, anomaly_type, _ = anomaly
            elif len(anomaly) == 4:
                anomaly_type = "high_residual"
            else:
                continue
            if anomaly_type in ["large_bet", "large_bet_high_residual"]:
                has_large_bet = True
                break

        return significant if has_large_bet else []

    def _check_bet_size_matching(self, anomalies_list):
        """
        Check if bet amounts in anomalies are similar (indicating coordinated collusion).
        Returns True if at least one pair of different players have matching bet sizes.

        Parameters:
            anomalies_list: List of (timestamp, player_id, residual, bet_amount, ...) tuples
                           (handles both 4 and 6 element formats)

        Returns:
            tuple: (has_matching_bets, matching_info)
                - has_matching_bets: bool indicating if any bets match
                - matching_info: dict with details about matches
        """
        if len(anomalies_list) < 2:
            return False, {}

        matching_info = {
            "exact_matches": [],
            "similar_matches": [],
            "match_ratio": 0.0,
        }

        # Check all pairs of anomalies
        for i, anomaly1 in enumerate(anomalies_list):
            # Extract bet amount (3rd element in 4-element format, 4th element in 6-element format)
            if len(anomaly1) >= 4:
                p1 = anomaly1[1]
                bet1 = anomaly1[3]
            else:
                continue

            for anomaly2 in anomalies_list[i + 1 :]:
                if len(anomaly2) >= 4:
                    p2 = anomaly2[1]
                    bet2 = anomaly2[3]
                else:
                    continue
                if p1 != p2:  # Different players
                    # Calculate similarity (relative difference)
                    max_bet = max(bet1, bet2)
                    min_bet = min(bet1, bet2)

                    if max_bet == 0:
                        continue

                    # Relative difference
                    relative_diff = abs(bet1 - bet2) / max_bet

                    # Check for exact match
                    if bet1 == bet2:
                        matching_info["exact_matches"].append(
                            {
                                "players": [p1, p2],
                                "bet_amount": bet1,
                            }
                        )
                    # Check for similar match (within threshold)
                    elif relative_diff <= self.bet_size_similarity_threshold:
                        matching_info["similar_matches"].append(
                            {
                                "players": [p1, p2],
                                "bet_amounts": [bet1, bet2],
                                "difference_pct": relative_diff * 100,
                            }
                        )

        # Check if we have at least one matching pair
        has_matching = (
            len(matching_info["exact_matches"]) > 0
            or len(matching_info["similar_matches"]) > 0
        )

        # Calculate match ratio (how many pairs match vs total pairs)
        # Total pairs = n * (n-1) / 2 for n anomalies
        n = len(anomalies_list)
        total_pairs = (n * (n - 1)) // 2 if n > 1 else 0
        matching_pairs = len(matching_info["exact_matches"]) + len(
            matching_info["similar_matches"]
        )
        matching_info["match_ratio"] = (
            matching_pairs / total_pairs if total_pairs > 0 else 0.0
        )

        return has_matching, matching_info

    def _check_collusion(self, table_id, current_timestamp):
        """
        Check for collusion patterns (synchronized anomalies).
        Uses both a tight window (for very synchronized bets) and a wider window.
        Only flags collusion if:
        1. Both players' bets exceed min_bet_for_collusion threshold
        2. Bet sizes match (exact or within similarity threshold) - strong indicator of coordination
        3. At least one player has a large_bet anomaly
        4. Other players have either large_bet or very high residuals (>2x threshold)
        5. Action sequence is suspicious (not normal betting patterns)

        Parameters:
            table_id: Table identifier
            current_timestamp: Current event timestamp
        """
        anomalies = self.recent_anomalies[table_id]

        if len(anomalies) < 2:
            return

        # Find anomalies within the tight collusion window (very synchronized)
        # Filter by minimum bet size: only consider anomalies with bet >= min_bet_for_collusion
        # Handle both old format (4 elements) and new format (6 elements)
        tight_recent = []
        for anomaly in anomalies:
            if len(anomaly) == 6:
                ts, player_id, res, bet_amount, anomaly_type, threshold = anomaly
            elif len(anomaly) == 4:
                ts, player_id, res, bet_amount = anomaly
                anomaly_type = "high_residual"
                threshold = res / 5.0  # Estimate
            else:
                continue
            if (
                current_timestamp - ts <= self.tight_collusion_window
                and bet_amount >= self.min_bet_for_collusion
            ):
                tight_recent.append(anomaly)

        # Find anomalies within the wider collusion window (also filtered by bet size)
        recent = []
        for anomaly in anomalies:
            if len(anomaly) == 6:
                ts, player_id, res, bet_amount, anomaly_type, threshold = anomaly
            elif len(anomaly) == 4:
                ts, player_id, res, bet_amount = anomaly
                anomaly_type = "high_residual"
                threshold = res / 5.0  # Estimate
            else:
                continue
            if (
                current_timestamp - ts <= self.collusion_window
                and bet_amount >= self.min_bet_for_collusion
            ):
                recent.append(anomaly)

        # Filter to only significant anomalies (at least one large_bet)
        tight_recent = self._filter_significant_anomalies(tight_recent)
        recent = self._filter_significant_anomalies(recent)

        # Check tight window first (more suspicious)
        # Extract player_id (2nd element in both formats)
        unique_players_tight = set(
            anomaly[1] for anomaly in tight_recent if len(anomaly) >= 2
        )
        if len(unique_players_tight) >= 2 and len(tight_recent) >= 2:
            # Check for bet size matching (strong indicator of collusion)
            has_matching, matching_info = self._check_bet_size_matching(tight_recent)

            if has_matching:  # Only flag if bet sizes match
                # Check if action sequence is suspicious
                is_suspicious, sequence_info = self._is_suspicious_action_sequence(
                    table_id, current_timestamp, unique_players_tight
                )

                if is_suspicious:  # Only flag if sequence is suspicious
                    # Very synchronized collusion detected with matching bets and suspicious sequence!
                    # Calculate max time difference (timestamp is 1st element)
                    timestamps = [
                        anomaly[0] for anomaly in tight_recent if len(anomaly) >= 1
                    ]
                    max_time_diff = (
                        max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
                    )
                    self._log_collusion(
                        table_id,
                        current_timestamp,
                        tight_recent,
                        unique_players_tight,
                        window_type="tight",
                        max_time_diff=max_time_diff,
                        bet_matching_info=matching_info,
                        sequence_info=sequence_info,
                    )
                    return

        # Check wider window
        if len(recent) < 2:
            return

        unique_players = set(anomaly[1] for anomaly in recent if len(anomaly) >= 2)

        if len(unique_players) >= 2:
            # Check for bet size matching
            has_matching, matching_info = self._check_bet_size_matching(recent)

            if has_matching:  # Only flag if bet sizes match
                # Check if action sequence is suspicious
                is_suspicious, sequence_info = self._is_suspicious_action_sequence(
                    table_id, current_timestamp, unique_players
                )

                if is_suspicious:  # Only flag if sequence is suspicious
                    # Calculate max time difference (timestamp is 1st element)
                    timestamps = [anomaly[0] for anomaly in recent if len(anomaly) >= 1]
                    max_time_diff = (
                        max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0
                    )
                    # Collusion detected with matching bets and suspicious sequence!
                    self._log_collusion(
                        table_id,
                        current_timestamp,
                        recent,
                        unique_players,
                        window_type="normal",
                        max_time_diff=max_time_diff,
                        bet_matching_info=matching_info,
                        sequence_info=sequence_info,
                    )

    def _log_collusion(
        self,
        table_id,
        timestamp,
        anomalies,
        players,
        window_type="normal",
        max_time_diff=0.0,
        bet_matching_info=None,
        sequence_info=None,
    ):
        """
        Log a detected collusion pattern.

        Parameters:
            table_id: Table identifier
            timestamp: Current timestamp
            anomalies: List of recent anomalies
            players: Set of involved players
            window_type: "tight" or "normal" to indicate synchronization level
            max_time_diff: Maximum time difference between anomalies (seconds)
            bet_matching_info: Optional dict with bet matching details from _check_bet_size_matching
            sequence_info: Optional dict with action sequence details from _is_suspicious_action_sequence
        """
        window_size = (
            self.tight_collusion_window
            if window_type == "tight"
            else self.collusion_window
        )
        sync_level = "HIGHLY SYNCHRONIZED" if window_type == "tight" else "Synchronized"

        # Build anomalies list - handle both old format (4-tuple) and new format (6-tuple with anomaly_type, threshold)
        anomalies_list = []
        for anomaly in anomalies:
            if len(anomaly) == 6:
                # New format: (timestamp, player_id, residual, bet_amount, anomaly_type, threshold)
                ts, player_id, res, bet_amount, anomaly_type, threshold = anomaly
                anomalies_list.append(
                    {
                        "player_id": player_id,
                        "timestamp": ts,
                        "residual": float(res),
                        "bet_amount": float(bet_amount),
                        "anomaly_type": anomaly_type,
                        "threshold": float(threshold),
                    }
                )
            elif len(anomaly) == 4:
                # Old format: (timestamp, player_id, residual, bet_amount) - for backward compatibility
                ts, player_id, res, bet_amount = anomaly
                anomalies_list.append(
                    {
                        "player_id": player_id,
                        "timestamp": ts,
                        "residual": float(res),
                        "bet_amount": float(bet_amount),
                        "anomaly_type": "high_residual",
                        "threshold": res / 5.0,  # Estimate
                    }
                )
            else:
                # Very old format: (timestamp, player_id, residual) - for backward compatibility
                ts, player_id, res = anomaly
                anomalies_list.append(
                    {
                        "player_id": player_id,
                        "timestamp": ts,
                        "residual": float(res),
                        "bet_amount": 0.0,  # Unknown bet amount in old format
                        "anomaly_type": "high_residual",
                        "threshold": res / 5.0,  # Estimate
                    }
                )

        # Build details string with bet matching information
        details_parts = [
            f"{sync_level} betting anomaly detected among {len(players)} players",
            f"max time diff: {max_time_diff:.2f}s",
            f"min bet: ${self.min_bet_for_collusion:.2f}",
        ]

        if bet_matching_info:
            exact_matches = bet_matching_info.get("exact_matches", [])
            similar_matches = bet_matching_info.get("similar_matches", [])

            if exact_matches:
                details_parts.append(f"EXACT bet matches: {len(exact_matches)} pairs")
            if similar_matches:
                details_parts.append(
                    f"similar bet matches: {len(similar_matches)} pairs"
                )
            if bet_matching_info.get("match_ratio", 0) > 0:
                details_parts.append(
                    f"match ratio: {bet_matching_info['match_ratio']:.1%}"
                )

        if sequence_info:
            suspicious_patterns = sequence_info.get("suspicious_patterns", [])
            if suspicious_patterns:
                pattern_types = [p["pattern"] for p in suspicious_patterns]
                details_parts.append(
                    f"suspicious sequence: {', '.join(set(pattern_types))}"
                )

        details = f"{', '.join(details_parts)}"

        collusion_entry = {
            "timestamp": timestamp,
            "table_id": table_id,
            "type": "collusion_pattern",
            "players": list(players),
            "num_players": len(players),
            "anomalies": anomalies_list,
            "sync_level": window_type,
            "max_time_diff": max_time_diff,
            "min_bet_threshold": self.min_bet_for_collusion,
            "bet_size_similarity_threshold": self.bet_size_similarity_threshold,
            "bet_matching": bet_matching_info if bet_matching_info else {},
            "action_sequence": sequence_info if sequence_info else {},
            "details": details,
        }

        # Log to file
        self.logger.info(json.dumps(collusion_entry))

        # Console output
        if self.console_output:
            urgency = "🚨🚨" if window_type == "tight" else "🚨"
            print(f"\n{urgency} COLLUSION DETECTED at table {table_id}!")
            print(f"   Players involved: {', '.join(players)}")
            print(f"   Time window: {window_size}s ({window_type})")
            print(f"   Max time difference: {max_time_diff:.2f}s")
            print(f"   Min bet threshold: ${self.min_bet_for_collusion:.2f}")

            if bet_matching_info:
                exact_matches = bet_matching_info.get("exact_matches", [])
                similar_matches = bet_matching_info.get("similar_matches", [])

                if exact_matches:
                    match_details = []
                    for match in exact_matches:
                        match_details.append(
                            f"{match['players'][0]}&{match['players'][1]}: ${match['bet_amount']:.2f}"
                        )
                    print(f"   ⚠️  EXACT bet matches: {', '.join(match_details)}")

                if similar_matches:
                    match_details = []
                    for match in similar_matches:
                        match_details.append(
                            f"{match['players'][0]}&{match['players'][1]}: "
                            f"${match['bet_amounts'][0]:.2f}≈${match['bet_amounts'][1]:.2f} "
                            f"({match['difference_pct']:.1f}% diff)"
                        )
                    print(f"   ⚠️  Similar bet matches: {', '.join(match_details)}")

            if sequence_info:
                suspicious_patterns = sequence_info.get("suspicious_patterns", [])
                if suspicious_patterns:
                    pattern_details = []
                    for pattern in suspicious_patterns:
                        if pattern["pattern"] == "raise_raise":
                            pattern_details.append(
                                f"raise→raise ({pattern['players'][0]}&{pattern['players'][1]}, "
                                f"{pattern['time_diff']:.2f}s)"
                            )
                        elif pattern["pattern"] == "bet_immediate_raise":
                            pattern_details.append(
                                f"bet→immediate_raise ({pattern['players'][0]}&{pattern['players'][1]}, "
                                f"{pattern['time_diff']:.2f}s)"
                            )
                    if pattern_details:
                        print(
                            f"   ⚠️  Suspicious sequence: {', '.join(pattern_details)}"
                        )

            print(f"   Pattern: {sync_level} betting\n")

        self.collusion_detected += 1

    def check_anomaly(self, residual, threshold):
        """
        Check if a residual exceeds the anomaly threshold.

        Parameters:
            residual: Residual value
            threshold: Anomaly threshold

        Returns:
            bool: True if anomaly detected
        """
        return abs(residual) > threshold

    def get_statistics(self):
        """
        Get anomaly detection statistics.

        Returns:
            dict: Statistics dictionary
        """
        return {
            "total_anomalies": self.total_anomalies,
            "collusion_detected": self.collusion_detected,
            "active_tables": len(self.recent_anomalies),
        }

    def print_summary(self):
        """Print summary of anomaly detection."""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("ANOMALY DETECTION SUMMARY")
        print("=" * 60)
        print(f"Total anomalies detected: {stats['total_anomalies']}")
        print(f"Collusion patterns found: {stats['collusion_detected']}")
        print(f"Tables monitored: {stats['active_tables']}")
        print("=" * 60 + "\n")


class CollusionDetector:
    """
    Advanced collusion detection with pattern analysis.
    """

    def __init__(self, window_size=10):
        """
        Initialize collusion detector.

        Parameters:
            window_size: Number of recent hands to analyze
        """
        self.window_size = window_size

        # Track player pairs and their correlation
        self.player_pairs = defaultdict(
            lambda: {"joint_anomalies": 0, "total_hands": 0, "correlation_score": 0.0}
        )

    def update_pair_statistics(self, player1, player2, both_anomalous):
        """
        Update statistics for a player pair.

        Parameters:
            player1: First player ID
            player2: Second player ID
            both_anomalous: Whether both players had anomalies
        """
        pair_key = tuple(sorted([player1, player2]))
        stats = self.player_pairs[pair_key]

        stats["total_hands"] += 1
        if both_anomalous:
            stats["joint_anomalies"] += 1

        # Calculate correlation score
        if stats["total_hands"] > 0:
            stats["correlation_score"] = stats["joint_anomalies"] / stats["total_hands"]

    def get_suspicious_pairs(self, threshold=0.3):
        """
        Get player pairs with high collusion correlation.

        Parameters:
            threshold: Correlation threshold for suspicion

        Returns:
            list: List of suspicious player pairs
        """
        suspicious = []
        for pair, stats in self.player_pairs.items():
            if stats["total_hands"] >= 5 and stats["correlation_score"] >= threshold:
                suspicious.append(
                    {
                        "players": pair,
                        "correlation": stats["correlation_score"],
                        "joint_anomalies": stats["joint_anomalies"],
                        "total_hands": stats["total_hands"],
                    }
                )

        return sorted(suspicious, key=lambda x: x["correlation"], reverse=True)
