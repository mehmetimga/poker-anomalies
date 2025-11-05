#!/bin/bash

# Poker Anomaly Detection Pipeline - Local Runner
# This script orchestrates the entire pipeline: Kafka, Producer, Consumer
#
# Usage: ./run_local.sh [--threads N] [--delay D]
#   --threads N: Number of threads for parallel file processing (default: auto-detect from number of files)
#   --delay D:   Delay between events in seconds (default: 0.3)

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Default values
THREADS=""
DELAY="0.3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--threads N] [--delay D]"
            echo "  --threads N: Number of threads for parallel file processing (default: auto-detect)"
            echo "  --delay D:   Delay between events in seconds (default: 0.3)"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Poker Anomaly Detection Pipeline"
echo "=========================================="
echo "Project directory: $PROJECT_DIR"
echo ""

# Step 1: Check Docker
echo "Step 1: Checking Docker..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker."
    exit 1
fi
print_status "Docker is running"
echo ""

# Step 2: Start Kafka with Docker Compose
echo "Step 2: Starting Kafka..."
cd "$PROJECT_DIR"

if [ ! -f "docker-compose.yml" ]; then
    print_error "docker-compose.yml not found!"
    exit 1
fi

# Start Kafka in detached mode
docker compose up -d

print_status "Waiting for Kafka to be ready..."

# Wait for Kafka container to be running
MAX_WAIT=60
WAIT_INTERVAL=2
elapsed=0

while [ $elapsed -lt $MAX_WAIT ]; do
    # Check if Kafka container is running
    if ! docker compose ps | grep -q "kafka.*Up"; then
        print_warning "Kafka container not running yet... (${elapsed}s)"
        sleep $WAIT_INTERVAL
        elapsed=$((elapsed + WAIT_INTERVAL))
        continue
    fi

    # Try to connect to Kafka broker using Kafka CLI tools
    # This checks if Kafka is actually accepting connections
    if docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092 > /dev/null 2>&1; then
        print_status "Kafka is ready and accepting connections"
        break
    fi

    # Also try checking if port is accessible from host
    if command -v nc > /dev/null 2>&1; then
        if nc -z localhost 9092 2>/dev/null; then
            # Port is open, but Kafka might still be initializing
            print_warning "Kafka port is open but not ready yet... (${elapsed}s)"
        fi
    fi

    sleep $WAIT_INTERVAL
    elapsed=$((elapsed + WAIT_INTERVAL))
done

# Final check
if [ $elapsed -ge $MAX_WAIT ]; then
    print_error "Kafka failed to become ready within ${MAX_WAIT}s"
    print_error "Container status:"
    docker compose ps
    print_error "Kafka logs:"
    docker compose logs kafka | tail -20
    exit 1
fi

# Verify Kafka is actually working by trying to list topics
if docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
    print_status "Kafka broker is fully operational"
else
    print_warning "Kafka is running but broker check failed - continuing anyway"
fi
echo ""

# Step 3: Check Python dependencies
echo "Step 3: Checking Python dependencies..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    exit 1
fi

print_status "Python 3 found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# Install dependencies
if [ -f "requirements.txt" ]; then
    print_status "Installing dependencies..."
    pip install -q -r requirements.txt
    print_status "Dependencies installed"
else
    print_warning "requirements.txt not found"
fi
echo ""

# Step 4: Clean up old logs and create logs directory
echo "Step 4: Setting up logs directory..."
cd "$PROJECT_DIR"

# Remove old log files
if [ -d "logs" ]; then
    LOG_COUNT=$(find logs -name "table_*.log" 2>/dev/null | wc -l)
    if [ "$LOG_COUNT" -gt 0 ]; then
        print_status "Removing $LOG_COUNT old log file(s)..."
        rm -f logs/table_*.log
        print_status "Old logs removed"
    else
        print_status "No old logs to remove"
    fi
else
    print_status "Logs directory doesn't exist yet"
fi

# Ensure logs directory exists
mkdir -p logs
print_status "Logs directory ready"
echo ""

# Step 5: Run Producer and Consumer
echo "Step 5: Starting Pipeline..."
echo "=========================================="
echo ""

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo "=========================================="
    echo "Cleaning up..."
    
    # Kill background processes
    if [ ! -z "$PRODUCER_PID" ]; then
        kill $PRODUCER_PID 2>/dev/null || true
    fi
    if [ ! -z "$CONSUMER_PID" ]; then
        kill $CONSUMER_PID 2>/dev/null || true
    fi
    
    # Deactivate virtual environment
    deactivate 2>/dev/null || true
    
    echo "=========================================="
    echo "Pipeline stopped"
    echo "To stop Kafka: docker compose down"
    echo "=========================================="
}

trap cleanup EXIT INT TERM

# Start Consumer in background
print_status "Starting Consumer..."
cd "$PROJECT_DIR"
python3 -m src.consumer --topic poker-actions --kafka localhost:9092 &
CONSUMER_PID=$!
sleep 3

# Maximum number of threads allowed (matches MAX_WORKERS_LIMIT in config.py)
MAX_THREADS=4

# Auto-detect number of files if threads not specified
if [ -z "$THREADS" ]; then
    DATA_DIR="$PROJECT_DIR/data"
    if [ -d "$DATA_DIR" ]; then
        FILE_COUNT=$(find "$DATA_DIR" -name "table_*.txt" | wc -l)
        if [ "$FILE_COUNT" -gt 0 ]; then
            # Use minimum of file count and max threads
            THREADS=$((FILE_COUNT < MAX_THREADS ? FILE_COUNT : MAX_THREADS))
            if [ "$FILE_COUNT" -gt "$MAX_THREADS" ]; then
                print_status "Auto-detected $FILE_COUNT file(s), using $THREADS thread(s) (max limit: $MAX_THREADS)"
                print_status "Remaining files will be processed as threads become available"
            else
                print_status "Auto-detected $FILE_COUNT file(s), using $THREADS thread(s)"
            fi
        else
            THREADS="1"
            print_warning "No table files found, using sequential processing"
        fi
    else
        THREADS="1"
        print_warning "Data directory not found, using sequential processing"
    fi
else
    # Cap user-specified threads at MAX_THREADS
    if [ "$THREADS" -gt "$MAX_THREADS" ]; then
        print_warning "Requested $THREADS threads, but maximum is $MAX_THREADS. Using $MAX_THREADS threads."
        THREADS="$MAX_THREADS"
    fi
fi

# Start Producer (processes all table_*.txt files in data/ directory)
print_status "Starting Producer with $THREADS thread(s) and ${DELAY}s delay..."
PRODUCER_CMD="python3 -m src.producer --topic poker-actions --delay $DELAY --kafka localhost:9092 --threads $THREADS"
eval $PRODUCER_CMD

# Wait for consumer to finish processing
echo ""
print_status "Waiting for consumer to complete..."
wait $CONSUMER_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Pipeline Execution Complete!"
echo "=========================================="
echo "Check logs/ directory for detected anomalies:"
echo "  - Per-table logs: logs/table_*.log (one file per table)"
echo ""
echo "To stop Kafka, run: docker compose down"
echo "=========================================="
