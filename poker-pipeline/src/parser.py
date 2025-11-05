"""
Parser for poker hand history files.
Reads and parses hand history lines into event dictionaries.
"""


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
