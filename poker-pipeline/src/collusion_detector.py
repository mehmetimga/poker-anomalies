"""
Advanced collusion detection with pattern analysis.
"""

from collections import defaultdict
from src.config import (
    COLLUSION_DETECTOR_WINDOW_SIZE,
    SUSPICIOUS_PAIR_THRESHOLD,
    MIN_HANDS_FOR_SUSPICION,
)


class CollusionDetector:
    """
    Advanced collusion detection with pattern analysis.
    """

    def __init__(self, window_size=None):
        """
        Initialize collusion detector.

        Parameters:
            window_size: Number of recent hands to analyze (default: from config)
        """
        self.window_size = (
            window_size if window_size is not None else COLLUSION_DETECTOR_WINDOW_SIZE
        )

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

    def get_suspicious_pairs(self, threshold=None):
        """
        Get player pairs with high collusion correlation.

        Parameters:
            threshold: Correlation threshold for suspicion (default: from config)

        Returns:
            list: List of suspicious player pairs
        """
        if threshold is None:
            threshold = SUSPICIOUS_PAIR_THRESHOLD

        suspicious = []
        for pair, stats in self.player_pairs.items():
            if (
                stats["total_hands"] >= MIN_HANDS_FOR_SUSPICION
                and stats["correlation_score"] >= threshold
            ):
                suspicious.append(
                    {
                        "players": pair,
                        "correlation": stats["correlation_score"],
                        "joint_anomalies": stats["joint_anomalies"],
                        "total_hands": stats["total_hands"],
                    }
                )

        return sorted(suspicious, key=lambda x: x["correlation"], reverse=True)
