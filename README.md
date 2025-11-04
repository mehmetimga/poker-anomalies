# Poker Collusion Detection Pipeline

A real-time anomaly detection system for identifying collusion in online poker using streaming data, Unscented Kalman Filters (UKF), and multi-layer pattern validation.

## Overview

This project implements a real-time anomaly detection pipeline that processes poker hand history events through Apache Kafka, applies UKF-based state estimation to track player betting behaviors, and flags anomalous patterns that may indicate collusion or cheating.

**Performance**: ~92-100% precision, ~3-4% false positive rate, 92% recall

### Key Features

- **Real-time Streaming**: Kafka-based event processing with <100ms latency
- **Advanced Filtering**: Unscented Kalman Filter for non-linear bet pattern tracking
- **Multi-Layer Anomaly Detection**: 
  - 5σ adaptive threshold (reduced false positives by 27%)
  - Absolute bet size detection for unusually large bets
  - Player-specific adaptive thresholds with robust statistics
- **Advanced Collusion Detection** with four-layer validation:
  - **Minimum Bet Size Filter** ($30): Only flags economically significant collusion
  - **Bet Size Matching**: Detects exact/similar bet matches (strong coordination indicator)
  - **Action Sequence Filter**: Validates suspicious betting sequences (bet→immediate raise, raise→raise)
  - **Significant Anomaly Filter**: Requires at least one large_bet anomaly (reduces false positives by 20-30%)
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
    │   └── ...                 # Additional table_N.txt files
    ├── src/
    │   ├── __init__.py
    │   ├── producer.py         # Kafka producer
    │   ├── consumer.py         # Kafka consumer + detection
    │   ├── filters.py          # Kalman/EKF/UKF implementations
    │   ├── models.py           # Process/measurement models
    │   └── anomaly_logger.py   # Logging and alerting
    ├── logs/
    │   └── anomalies.log       # Anomaly output (generated)
    ├── scripts/
    │   └── run_local.sh        # Automated runner
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

---

**Built with ❤️ for fraud detection in online gaming**
