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
    
    def __init__(self, log_file='logs/anomalies.log', console_output=True):
        """
        Initialize the anomaly logger.
        
        Parameters:
            log_file: Path to log file
            console_output: Whether to echo anomalies to console
        """
        self.log_file = log_file
        self.console_output = console_output
        
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Setup logging
        self.logger = logging.getLogger('AnomalyLogger')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Track recent anomalies per table for collusion detection
        # Format: {table_id: deque([(timestamp, player_id, residual), ...])}
        self.recent_anomalies = defaultdict(lambda: deque(maxlen=10))
        
        # Collusion tracking window (seconds)
        self.collusion_window = 5.0
        
        # Statistics
        self.total_anomalies = 0
        self.collusion_detected = 0
        
    def log_anomaly(self, event, residual, threshold, anomaly_type='high_residual'):
        """
        Log an individual player anomaly.
        
        Parameters:
            event: Poker event dict
            residual: Residual value from filter
            threshold: Threshold that was exceeded
            anomaly_type: Type of anomaly
        """
        log_entry = {
            'timestamp': event['timestamp'],
            'player_id': event['player_id'],
            'table_id': event['table_id'],
            'action': event['action'],
            'amount': event.get('amount', 0.0),
            'residual': float(residual),
            'threshold': float(threshold),
            'type': anomaly_type,
            'details': f"Deviation >3σ (threshold={threshold:.2f})"
        }
        
        # Log to file
        self.logger.info(json.dumps(log_entry))
        
        # Console output
        if self.console_output:
            print(f"⚠️  ANOMALY DETECTED: Player {event['player_id']} at table {event['table_id']}")
            print(f"   Action: {event['action']} ${event.get('amount', 0):.2f}, Residual: {residual:.2f} (threshold: {threshold:.2f})")
        
        # Track for collusion detection
        self.recent_anomalies[event['table_id']].append(
            (event['timestamp'], event['player_id'], residual)
        )
        
        self.total_anomalies += 1
        
        # Check for collusion pattern
        self._check_collusion(event['table_id'], event['timestamp'])
        
    def _check_collusion(self, table_id, current_timestamp):
        """
        Check for collusion patterns (synchronized anomalies).
        
        Parameters:
            table_id: Table identifier
            current_timestamp: Current event timestamp
        """
        anomalies = self.recent_anomalies[table_id]
        
        if len(anomalies) < 2:
            return
        
        # Find anomalies within the collusion window
        recent = [
            (ts, player_id, res) 
            for ts, player_id, res in anomalies 
            if current_timestamp - ts <= self.collusion_window
        ]
        
        if len(recent) < 2:
            return
        
        # Check if multiple distinct players have anomalies in the window
        unique_players = set(player_id for _, player_id, _ in recent)
        
        if len(unique_players) >= 2:
            # Collusion detected!
            self._log_collusion(table_id, current_timestamp, recent, unique_players)
    
    def _log_collusion(self, table_id, timestamp, anomalies, players):
        """
        Log a detected collusion pattern.
        
        Parameters:
            table_id: Table identifier
            timestamp: Current timestamp
            anomalies: List of recent anomalies
            players: Set of involved players
        """
        collusion_entry = {
            'timestamp': timestamp,
            'table_id': table_id,
            'type': 'collusion_pattern',
            'players': list(players),
            'num_players': len(players),
            'anomalies': [
                {
                    'player_id': player_id,
                    'timestamp': ts,
                    'residual': float(res)
                }
                for ts, player_id, res in anomalies
            ],
            'details': f"Synchronized betting anomaly detected among {len(players)} players"
        }
        
        # Log to file
        self.logger.info(json.dumps(collusion_entry))
        
        # Console output
        if self.console_output:
            print(f"\n🚨 COLLUSION DETECTED at table {table_id}!")
            print(f"   Players involved: {', '.join(players)}")
            print(f"   Time window: {self.collusion_window}s")
            print(f"   Pattern: Synchronized high residuals\n")
        
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
            'total_anomalies': self.total_anomalies,
            'collusion_detected': self.collusion_detected,
            'active_tables': len(self.recent_anomalies)
        }
    
    def print_summary(self):
        """Print summary of anomaly detection."""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("ANOMALY DETECTION SUMMARY")
        print("="*60)
        print(f"Total anomalies detected: {stats['total_anomalies']}")
        print(f"Collusion patterns found: {stats['collusion_detected']}")
        print(f"Tables monitored: {stats['active_tables']}")
        print("="*60 + "\n")


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
        self.player_pairs = defaultdict(lambda: {
            'joint_anomalies': 0,
            'total_hands': 0,
            'correlation_score': 0.0
        })
    
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
        
        stats['total_hands'] += 1
        if both_anomalous:
            stats['joint_anomalies'] += 1
        
        # Calculate correlation score
        if stats['total_hands'] > 0:
            stats['correlation_score'] = stats['joint_anomalies'] / stats['total_hands']
    
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
            if stats['total_hands'] >= 5 and stats['correlation_score'] >= threshold:
                suspicious.append({
                    'players': pair,
                    'correlation': stats['correlation_score'],
                    'joint_anomalies': stats['joint_anomalies'],
                    'total_hands': stats['total_hands']
                })
        
        return sorted(suspicious, key=lambda x: x['correlation'], reverse=True)


