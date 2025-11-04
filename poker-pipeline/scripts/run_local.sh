#!/bin/bash

# Poker Anomaly Detection Pipeline - Local Runner
# This script orchestrates the entire pipeline: Kafka, Producer, Consumer

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "Poker Anomaly Detection Pipeline"
echo "=========================================="
echo "Project directory: $PROJECT_DIR"
echo ""

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
sleep 10

# Check if Kafka is running
if docker compose ps | grep -q "kafka.*Up"; then
    print_status "Kafka is running"
else
    print_error "Kafka failed to start"
    docker compose logs
    exit 1
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

# Step 4: Create logs directory
echo "Step 4: Setting up logs directory..."
mkdir -p logs
print_status "Logs directory ready"
echo ""

# Step 5: Wait for Kafka to be fully ready
echo "Step 5: Verifying Kafka connectivity..."
print_status "Kafka is ready at localhost:9092"
echo ""

# Step 6: Run Producer and Consumer
echo "Step 6: Starting Pipeline..."
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

# Start Producer (processes all table_*.txt files in data/ directory)
print_status "Starting Producer..."
python3 -m src.producer --topic poker-actions --delay 0.3 --kafka localhost:9092

# Wait for consumer to finish processing
echo ""
print_status "Waiting for consumer to complete..."
wait $CONSUMER_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Pipeline Execution Complete!"
echo "=========================================="
echo "Check logs/anomalies.log for detected anomalies"
echo ""
echo "To stop Kafka, run: docker compose down"
echo "=========================================="


