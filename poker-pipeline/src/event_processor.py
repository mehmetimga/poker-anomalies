"""
Event processing logic for poker anomaly detection.
Handles event processing, anomaly detection, and statistics tracking.
"""

import time
from src.filters.poker_ukf import PokerUKF
from src.anomaly_logger import AnomalyLogger
from src.config import DEFAULT_STD, ABSOLUTE_BET_ACTIONS


class EventProcessor:
    """
    Processes poker events and detects anomalies.
    """

    def __init__(self, anomaly_logger, collusion_detector):
        """
        Initialize event processor.

        Parameters:
            anomaly_logger: AnomalyLogger instance
            collusion_detector: CollusionDetector instance
        """
        self.anomaly_logger = anomaly_logger
        self.collusion_detector = collusion_detector

        # Dictionary of player filters {(table_id, player_id): PokerUKF}
        # Each table has independent filter state for each player
        self.player_filters = {}

        # Track active players per table (for collusion detector)
        # Format: {table_id: set([player_id, ...])}
        self.active_players_by_table = {}

        # Statistics
        self.events_processed = 0
        self.anomalies_detected = 0
        self.start_time = time.time()

    def initialize_player_filter(
        self, table_id, player_id, process_model, measurement_model
    ):
        """
        Initialize a player filter if it doesn't exist.

        Parameters:
            table_id: Table identifier
            player_id: Player identifier
            process_model: Process model function
            measurement_model: Measurement model function
        """
        filter_key = (table_id, player_id)
        if filter_key not in self.player_filters:
            self.player_filters[filter_key] = PokerUKF(
                player_id=player_id,
                process_model=process_model,
                measurement_model=measurement_model,
            )
            return True  # New filter created
        return False  # Filter already exists

    def process_event(self, event, process_model, measurement_model):
        """
        Process a single poker event and detect anomalies.

        Parameters:
            event: Poker event dictionary
            process_model: Process model function
            measurement_model: Measurement model function

        Returns:
            dict: Processing result with status and information
        """
        player_id = event["player_id"]
        table_id = event["table_id"]

        # Track active players at table
        if table_id not in self.active_players_by_table:
            self.active_players_by_table[table_id] = set()
        self.active_players_by_table[table_id].add(player_id)

        # Create unique key for this table+player combination
        filter_key = (table_id, player_id)

        # Initialize player filter if new (per table)
        is_new_filter = self.initialize_player_filter(
            table_id, player_id, process_model, measurement_model
        )

        # Get player's filter for this table
        player_ukf = self.player_filters[filter_key]

        # Track action for sequence analysis (all actions, not just anomalies)
        self.anomaly_logger.track_action(event)

        # Process event through UKF
        result = player_ukf.process_event(event)

        self.events_processed += 1

        # Check if warm-up is complete (need enough samples before detecting anomalies)
        warm_up_complete = player_ukf.is_warm_up_complete()

        # Get adaptive threshold for this player
        threshold = player_ukf.get_adaptive_threshold(default_std=DEFAULT_STD)

        # Check for anomaly based on residual
        residual_anomaly = False
        if warm_up_complete:
            residual_anomaly = self.anomaly_logger.check_anomaly(
                result["residual"], threshold
            )

        # Check for absolute bet size anomaly (large bets that might indicate collusion)
        absolute_bet_anomaly = False
        if event["action"] in ABSOLUTE_BET_ACTIONS and warm_up_complete:
            bet_amount = float(event.get("amount", 0))
            if bet_amount > 0:
                absolute_threshold = player_ukf.get_absolute_bet_threshold()
                if bet_amount >= absolute_threshold:
                    absolute_bet_anomaly = True

        # Anomaly if either residual or absolute bet size is anomalous
        is_anomaly = residual_anomaly or absolute_bet_anomaly

        # Determine anomaly type
        anomaly_type = "high_residual"
        if absolute_bet_anomaly and residual_anomaly:
            anomaly_type = "large_bet_high_residual"
        elif absolute_bet_anomaly:
            anomaly_type = "large_bet"
            # Use absolute bet threshold as the threshold for logging
            threshold = player_ukf.get_absolute_bet_threshold()

        # Log if anomaly detected
        if is_anomaly:
            # Update active players in logger before logging anomaly
            self.anomaly_logger.active_players[table_id].update(
                self.active_players_by_table[table_id]
            )

            self.anomaly_logger.log_anomaly(
                event=event,
                residual=result["residual"],
                threshold=threshold,
                anomaly_type=anomaly_type,
            )
            self.anomalies_detected += 1

        return {
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "result": result,
            "threshold": threshold,
            "warm_up_complete": warm_up_complete,
            "is_new_filter": is_new_filter,
        }

    def get_statistics(self):
        """
        Get processing statistics.

        Returns:
            dict: Statistics dictionary
        """
        elapsed_time = time.time() - self.start_time
        return {
            "events_processed": self.events_processed,
            "anomalies_detected": self.anomalies_detected,
            "players_tracked": len(self.player_filters),
            "elapsed_time": elapsed_time,
            "events_per_sec": (
                self.events_processed / elapsed_time if elapsed_time > 0 else 0
            ),
            "active_players_by_table": {
                table_id: len(players)
                for table_id, players in self.active_players_by_table.items()
            },
        }

    def get_player_statistics_by_table(self):
        """
        Get player statistics grouped by table.

        Returns:
            dict: {table_id: [(player_id, stats_dict), ...]}
        """
        by_table = {}
        for (table_id, player_id), ukf in self.player_filters.items():
            if table_id not in by_table:
                by_table[table_id] = []
            stats = ukf.get_statistics()
            by_table[table_id].append((player_id, stats))
        return by_table
