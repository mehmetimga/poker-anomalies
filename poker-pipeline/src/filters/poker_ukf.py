"""
Poker-specific UKF wrapper for player bet tracking and anomaly detection.
Tracks bet position and aggression velocity per player.
"""

import numpy as np
from collections import deque
from src.filters.ukf import UnscentedKalmanFilter
from src.config import (
    UKF_STATE_DIMENSION,
    UKF_ALPHA,
    UKF_BETA,
    UKF_KAPPA,
    PROCESS_NOISE_Q,
    MEASUREMENT_NOISE_R,
    INITIAL_POSITION,
    INITIAL_VELOCITY,
    INITIAL_COVARIANCE,
    RESIDUAL_HISTORY_MAXLEN,
    BET_HISTORY_MAXLEN,
    MIN_SAMPLES_FOR_DETECTION,
    MIN_SAMPLES_FOR_THRESHOLD,
    MIN_SAMPLES_FOR_ABSOLUTE_BET,
    MIN_DT,
    MAX_DT,
    DEFAULT_STD,
    SIGMA_MULTIPLIER,
    MIN_THRESHOLD_BASE,
    MIN_THRESHOLD_PCT,
    DEFAULT_AVG_BET,
    IQR_TO_STD_RATIO,
    RESIDUAL_CAP_MULTIPLIER,
    BET_CAP_MULTIPLIER,
    ABSOLUTE_BET_THRESHOLD_DEFAULT,
    ABSOLUTE_BET_Q75_MULTIPLIER,
    ABSOLUTE_BET_MEDIAN_MULTIPLIER,
    ABSOLUTE_BET_CAP_MULTIPLIER,
    TRACKED_ACTIONS,
    MIN_INNOVATION_VARIANCE,
)


class PokerUKF:
    """
    Poker-specific UKF wrapper for player bet tracking and anomaly detection.
    Tracks bet position and aggression velocity per player.
    """

    def __init__(self, player_id, process_model, measurement_model):
        """
        Parameters:
            player_id: Unique identifier for player
            process_model: Non-linear process function
            measurement_model: Non-linear measurement function
        """
        self.player_id = player_id
        self.process_model = process_model
        self.measurement_model = measurement_model

        # Initialize UKF with 2D state [bet_position, aggression_velocity]
        self.ukf = UnscentedKalmanFilter(
            n=UKF_STATE_DIMENSION,
            alpha=UKF_ALPHA,
            beta=UKF_BETA,
            kappa=UKF_KAPPA,
        )
        self.ukf.set_covariances(
            Q=np.eye(UKF_STATE_DIMENSION) * PROCESS_NOISE_Q,
            R=np.array([[MEASUREMENT_NOISE_R]]),
        )

        # Initial state: position=0, velocity=1
        self.ukf.x = np.array([[INITIAL_POSITION], [INITIAL_VELOCITY]])
        self.ukf.P = np.eye(UKF_STATE_DIMENSION) * INITIAL_COVARIANCE

        self.last_ts = None

        # Rolling statistics for adaptive thresholding
        self.residual_history = deque(maxlen=RESIDUAL_HISTORY_MAXLEN)
        self.bet_history = deque(maxlen=BET_HISTORY_MAXLEN)

        # Warm-up period: need at least 5 samples before flagging anomalies
        self.min_samples_for_detection = MIN_SAMPLES_FOR_DETECTION

    def process_event(self, event):
        """
        Process a poker action event and return estimates and residuals.

        Parameters:
            event: Dict with keys: timestamp, action, amount, table_id, player_id

        Returns:
            Dict with: estimate, residual, action, innovation_std
        """
        # Initialize timestamp on first event
        if self.last_ts is None:
            self.last_ts = event["timestamp"]
            return {
                "estimate": 0.0,
                "residual": 0.0,
                "action": event["action"],
                "innovation_std": 1.0,
            }

        # Calculate time delta
        dt = event["timestamp"] - self.last_ts
        dt = max(dt, MIN_DT)  # Avoid zero dt
        dt = min(dt, MAX_DT)  # Cap maximum dt to prevent extreme predictions

        # Check for numerical issues before processing
        state = self.ukf.get_state()
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            # Reset filter if state is invalid
            self.ukf.x = np.array([[INITIAL_POSITION], [INITIAL_VELOCITY]])
            self.ukf.P = np.eye(UKF_STATE_DIMENSION) * INITIAL_COVARIANCE
            self.last_ts = event["timestamp"]
            return {
                "estimate": 0.0,
                "residual": 0.0,
                "action": event["action"],
                "innovation_std": 1.0,
            }

        # Predict state forward
        try:
            self.ukf.predict(self.process_model, dt)
        except (OverflowError, ValueError, np.linalg.LinAlgError):
            # Reset filter on numerical error
            self.ukf.x = np.array([[INITIAL_POSITION], [INITIAL_VELOCITY]])
            self.ukf.P = np.eye(UKF_STATE_DIMENSION) * INITIAL_COVARIANCE
            self.last_ts = event["timestamp"]
            return {
                "estimate": 0.0,
                "residual": 0.0,
                "action": event["action"],
                "innovation_std": 1.0,
            }

        # Get estimate and check for validity
        estimate = float(self.ukf.get_state()[0])
        if np.isnan(estimate) or np.isinf(estimate):
            estimate = 0.0

        result = {
            "estimate": estimate,
            "residual": 0.0,
            "action": event["action"],
            "innovation_std": 1.0,
        }

        # Update on bet/raise actions
        if event["action"] in TRACKED_ACTIONS:
            amount = float(event["amount"])

            # Update filter with measurement
            try:
                innovation, innovation_var = self.ukf.update(
                    amount, self.measurement_model
                )

                # Check for invalid values
                if np.isnan(innovation) or np.isinf(innovation):
                    innovation = 0.0
                if np.isnan(innovation_var) or innovation_var <= 0:
                    innovation_var = 1.0

                result["estimate"] = float(self.ukf.get_state()[0])
                if np.isnan(result["estimate"]) or np.isinf(result["estimate"]):
                    result["estimate"] = 0.0

                result["residual"] = abs(innovation)
                result["innovation_std"] = np.sqrt(
                    max(innovation_var, MIN_INNOVATION_VARIANCE)
                )

                # Track history (only if valid)
                # For extreme values, use a capped version to prevent filter from adapting too quickly
                if not (np.isnan(result["residual"]) or np.isinf(result["residual"])):
                    # Cap residual for history tracking to prevent extreme outliers from skewing stats
                    # But still use actual residual for anomaly detection
                    if len(self.residual_history) > 0:
                        median_residual = np.median(list(self.residual_history))
                        # Cap at multiplier x median to prevent single outlier from dominating
                        capped_residual = min(
                            result["residual"],
                            median_residual * RESIDUAL_CAP_MULTIPLIER,
                        )
                        self.residual_history.append(capped_residual)
                    else:
                        self.residual_history.append(result["residual"])

                    # Track bet amount (cap extreme bets for history to prevent adaptation)
                    if len(self.bet_history) > 0:
                        median_bet = np.median(list(self.bet_history))
                        # Cap at multiplier x median for history tracking
                        capped_bet = (
                            min(amount, median_bet * BET_CAP_MULTIPLIER)
                            if median_bet > 0
                            else amount
                        )
                        self.bet_history.append(capped_bet)
                    else:
                        self.bet_history.append(amount)
            except (OverflowError, ValueError, np.linalg.LinAlgError):
                # If update fails, use prediction only
                result["residual"] = abs(amount - estimate)
                result["innovation_std"] = 1.0

        # Update timestamp
        self.last_ts = event["timestamp"]

        return result

    def get_adaptive_threshold(self, default_std=None, sigma_multiplier=None):
        """
        Calculate adaptive threshold based on historical residuals.
        Uses robust statistics (median and IQR) to avoid outliers skewing the threshold.
        Returns sigma_multiplier * std_dev of residuals with robust statistics.

        Parameters:
            default_std: Default standard deviation for early samples (default: from config)
            sigma_multiplier: Multiplier for standard deviation (default: from config)
        """
        if default_std is None:
            default_std = DEFAULT_STD
        if sigma_multiplier is None:
            sigma_multiplier = SIGMA_MULTIPLIER

        if len(self.residual_history) < MIN_SAMPLES_FOR_THRESHOLD:
            return sigma_multiplier * default_std

        residuals = np.array(list(self.residual_history))

        # Use robust statistics: median and IQR instead of mean/std
        # This prevents outliers from inflating the threshold
        median_residual = np.median(residuals)
        q75 = np.percentile(residuals, 75)
        q25 = np.percentile(residuals, 25)
        iqr = q75 - q25

        # Use IQR-based std estimate (more robust than std)
        # IQR ≈ 1.35 * std for normal distributions
        robust_std = iqr / IQR_TO_STD_RATIO if iqr > 0 else np.std(residuals)

        # Also compute traditional std as fallback
        traditional_std = np.std(residuals)

        # Use the smaller of the two to avoid over-inflating threshold
        # But ensure minimum threshold
        std = min(robust_std, traditional_std)

        # Minimum threshold based on typical bet sizes
        # Don't let threshold go below reasonable minimum (e.g., 10% of typical bet)
        avg_bet = np.mean(self.bet_history) if self.bet_history else DEFAULT_AVG_BET
        min_threshold = max(MIN_THRESHOLD_BASE, avg_bet * MIN_THRESHOLD_PCT)

        return max(sigma_multiplier * max(std, MIN_THRESHOLD_BASE), min_threshold)

    def is_warm_up_complete(self):
        """Check if we have enough samples to start anomaly detection."""
        return len(self.residual_history) >= self.min_samples_for_detection

    def get_absolute_bet_threshold(self):
        """
        Get absolute bet size threshold for detecting unusually large bets.
        Returns threshold based on historical bet sizes.
        """
        if len(self.bet_history) < MIN_SAMPLES_FOR_ABSOLUTE_BET:
            return ABSOLUTE_BET_THRESHOLD_DEFAULT

        bet_amounts = np.array(list(self.bet_history))
        median_bet = np.median(bet_amounts)
        q75_bet = np.percentile(bet_amounts, 75)
        q90_bet = np.percentile(bet_amounts, 90)

        # Base threshold from robust statistics
        threshold_candidates = [
            ABSOLUTE_BET_THRESHOLD_DEFAULT,
            ABSOLUTE_BET_Q75_MULTIPLIER * q75_bet,
            ABSOLUTE_BET_MEDIAN_MULTIPLIER * median_bet,
        ]
        threshold = max(threshold_candidates)

        # Cap threshold so it cannot grow far beyond recent high-percentile bets
        if q90_bet > 0:
            cap_floor = q90_bet + MIN_THRESHOLD_BASE
            cap_scaled = q90_bet * ABSOLUTE_BET_CAP_MULTIPLIER
            threshold = min(threshold, max(cap_floor, cap_scaled))

        threshold = max(threshold, ABSOLUTE_BET_THRESHOLD_DEFAULT)
        return float(threshold)

    def get_statistics(self):
        """Return player statistics for monitoring."""
        return {
            "player_id": self.player_id,
            "state": self.ukf.get_state().tolist(),
            "residual_history": list(self.residual_history),
            "bet_history": list(self.bet_history),
            "avg_bet": np.mean(self.bet_history) if self.bet_history else 0.0,
            "std_residual": (
                np.std(self.residual_history) if self.residual_history else 0.0
            ),
        }
