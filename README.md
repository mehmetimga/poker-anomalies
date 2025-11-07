# Poker Collusion Detection Pipeline

A real-time anomaly detection system for identifying collusion in online poker using streaming data, Unscented Kalman Filters (UKF), and multi-layer pattern validation.

## Overview

This project implements a real-time anomaly detection pipeline that processes poker hand history events through Apache Kafka, applies UKF-based state estimation to track player betting behaviors, and flags anomalous patterns that may indicate collusion or cheating.

**Performance**: 91.7% detection rate on bundled samples, ~92-100% precision, ~3-4% false positive rate

### Key Features

- **Real-time Streaming**: Kafka-based event processing with <100ms latency
- **Advanced Filtering**: Unscented Kalman Filter for non-linear bet pattern tracking
- **Multi-Layer Anomaly Detection**: 
  - 3.5σ adaptive threshold (reduces false positives while keeping sensitivity)
  - Absolute bet size detection with percentile caps for responsive large-bet alerts
  - Player-specific adaptive thresholds with robust statistics
- **Advanced Collusion Detection** with four-layer validation:
  - **Minimum Bet Size Filter** ($20): Only flags economically significant collusion
  - **Bet Size Matching**: Detects exact/similar bet matches within 8% tolerance
  - **Action Sequence Filter**: Validates suspicious betting sequences (bet→immediate raise, raise→raise)
  - **Significant Anomaly Filter**: Requires at least one large_bet anomaly within a 6s window
- **High Precision**: ~92-100% precision with ~3-4% false positive rate
- **Modular Architecture**: Easy to extend with additional filters and detection strategies

## Project Structure

```
poker-anomalies/
├── README.md                   # This file (project overview)
├── docs/                       # Research papers and investigations
│   ├── papers/                 # Research papers (PDFs)
│   ├── ai/                     # AI-generated documentation
│   │   ├── ALGORITHM_QUICK_REFERENCE.md
│   │   ├── IMPROVEMENTS_AND_ALTERNATIVES.md
│   │   └── RECOMMENDED_NEXT_STEPS.md
│   └── investigations/
│       └── investigation*.md   # Investigation documents
│
└── poker-pipeline/
    ├── README.md               # Comprehensive pipeline documentation
    ├── requirements.txt        # Python dependencies
    ├── docker-compose.yml      # Kafka setup
    ├── data/
    │   ├── table_1.txt         # Table 1 hand history
    │   ├── table_2.txt         # Table 2 hand history
    │   ├── table_3.txt         # Table 3 hand history (single-player outliers)
    │   └── table_4.txt         # Table 4 hand history (tight collusion scenario)
    ├── src/
    │   ├── __init__.py
    │   ├── producer.py         # Kafka producer
    │   ├── consumer.py         # Kafka consumer + detection
    │   ├── filters.py          # Kalman/EKF/UKF implementations
    │   ├── models.py           # Process/measurement models
    │   └── anomaly_logger.py   # Logging and alerting
    ├── logs/
    │   ├── table_1.log         # Table 1 anomaly output (generated)
    │   ├── table_2.log         # Table 2 anomaly output (generated)
    │   ├── table_3.log         # Table 3 anomaly output (generated)
    │   └── table_4.log         # Table 4 anomaly output (generated)
    ├── scripts/
    │   ├── run_local.sh        # Automated pipeline runner
    │   └── run_detection.sh    # Detection rate reporter
    └── tests/
        └── test_filters.py     # Test suite
```

## Getting Started

For detailed installation, usage, and configuration instructions, see the [poker-pipeline README](poker-pipeline/README.md).

### Quick Start

```bash
cd poker-pipeline
./scripts/run_local.sh
```

This will automatically:
1. Check dependencies
2. Start Kafka
3. Run the pipeline
4. Display results

After the pipeline finishes, run `./scripts/run_detection.sh` to replay the generated logs and report the current detection rate (ships at 11/12 = 91.7%).

## Documentation

- **Pipeline Documentation**: See [poker-pipeline/README.md](poker-pipeline/README.md) for comprehensive documentation including:
  - Architecture details
  - Installation and setup
  - Usage examples
  - Configuration options
  - Algorithm details
  - Troubleshooting

- **Research Papers**: Located in `docs/papers/`:
  - Online Time Series Anomaly Detection with State Space Gaussian Processes
  - Deep Learning for Time Series Anomaly Detection: A Survey

- **AI Documentation**: Located in `docs/ai/`:
  - Algorithm Quick Reference
  - Improvements and Alternatives
  - Recommended Next Steps

- **Investigations**: Located in `docs/investigations/`:
  - Investigation documents detailing methodology and findings
- **Detection Analysis**: Run `poker-pipeline/scripts/run_detection.sh` to compare planted vs detected anomalies (ships at 11/12 = 91.7%)

## References

### Research Papers

Research papers are located in `docs/papers/`:

1. **Online Time Series Anomaly Detection with State Space Gaussian Processes**
   - Foundation for state-space anomaly detection approach

2. **Deep Learning for Time Series Anomaly Detection: A Survey**
   - Overview of modern anomaly detection techniques

### Libraries Used

- `kafka-python`: Kafka client for Python
- `numpy`: Numerical computing
- `scipy`: Scientific computing utilities

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
- Open a GitHub issue
- Check the [pipeline README](poker-pipeline/README.md) troubleshooting section
- Review investigation files in `docs/investigations/` for methodology details
