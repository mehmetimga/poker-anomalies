"""
File utility functions for discovering and processing table files.
"""

import os
import glob
import sys
from pathlib import Path
from src.config import DEFAULT_DATA_DIR, TABLE_FILE_PATTERN


def find_table_files(data_dir=None, input_file=None):
    """
    Find table files to process based on input parameters.

    Parameters:
        data_dir: Directory containing table_*.txt files (default: None)
        input_file: Path to single hand history file (default: None, overrides data_dir)

    Returns:
        tuple: (files_to_process, info_message)
            - files_to_process: List of file paths
            - info_message: Information about what was found

    Raises:
        SystemExit: If no files are found or input is invalid
    """
    files_to_process = []
    info_message = ""

    if input_file:
        # Single file mode
        if os.path.exists(input_file):
            files_to_process = [input_file]
            info_message = f"Processing single file: {input_file}"
        else:
            print(f"✗ Error: Input file not found: {input_file}")
            sys.exit(1)
    elif data_dir:
        # Directory mode - find all table_*.txt files
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"✗ Error: Data directory not found: {data_dir}")
            sys.exit(1)

        pattern = str(data_path / TABLE_FILE_PATTERN)
        files_to_process = sorted(glob.glob(pattern))

        if not files_to_process:
            print(f"⚠️  No {TABLE_FILE_PATTERN} files found in {data_dir}")
            print(f"   Looking for files matching: {pattern}")
            sys.exit(1)

        info_message = f"Found {len(files_to_process)} table file(s):"
        for f in files_to_process:
            info_message += f"\n  - {os.path.basename(f)}"
    else:
        # Default: look in data/ directory relative to script location
        script_dir = Path(__file__).parent.parent
        default_data_dir = script_dir / DEFAULT_DATA_DIR
        pattern = str(default_data_dir / TABLE_FILE_PATTERN)
        files_to_process = sorted(glob.glob(pattern))

        if not files_to_process:
            print(f"✗ Error: No {TABLE_FILE_PATTERN} files found in {default_data_dir}")
            print(
                f"   Please provide --input or --data-dir, or add {TABLE_FILE_PATTERN} files to {DEFAULT_DATA_DIR}/"
            )
            sys.exit(1)

        info_message = (
            f"Found {len(files_to_process)} table file(s) in default data directory:"
        )
        for f in files_to_process:
            info_message += f"\n  - {os.path.basename(f)}"

    return files_to_process, info_message


def get_effective_workers(max_workers, max_limit):
    """
    Calculate effective number of workers, capped at maximum limit.

    Parameters:
        max_workers: Requested number of workers
        max_limit: Maximum allowed workers

    Returns:
        int: Effective number of workers (1 to max_limit)
    """
    if not max_workers or max_workers < 1:
        return 1
    return min(max_workers, max_limit)
