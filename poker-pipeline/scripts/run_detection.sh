#!/bin/bash

# Runner for detection analysis.
# Ensures the script executes from the project root and invokes the analyzer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ANALYZER_PATH="${PROJECT_DIR}/src/analyze_detection_rate.py"

if [[ ! -f "${ANALYZER_PATH}" ]]; then
    echo "Error: analyzer not found at ${ANALYZER_PATH}" >&2
    exit 1
fi

cd "${PROJECT_DIR}"
python3 "${ANALYZER_PATH}"
