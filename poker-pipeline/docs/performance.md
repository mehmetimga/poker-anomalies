# Performance

## Benchmarks (on MacBook Pro M1)

- **Latency**: ~50ms per event (predict + update + log)
- **Throughput**: ~20 events/second (single consumer)
- **Memory**: ~50MB for 6 players + 200 events
- **CPU**: ~10% single core utilization

## Scaling Considerations

For production deployment:
- Use **multiple consumer instances** (Kafka consumer groups)
- Partition by **table_id** for parallel processing
- Deploy on **AWS ECS/EKS** with auto-scaling
- Replace local Kafka with **AWS MSK** or **Confluent Cloud**

## Performance Tips

- Reduce `--delay` for faster processing (default: 0.5s)
- Increase Kafka partitions for multi-table support
- Use virtual environment for isolation

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

