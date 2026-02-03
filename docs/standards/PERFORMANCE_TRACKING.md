# Performance Regression Tracking System

This document describes the system for tracking performance metrics over time, detecting regressions, and providing alerts.

## Components

### Performance Tracker

The core logic resides in `src/inference_pio/core/tools/scripts/benchmarking/performance_tracker.py`.

It handles:
- Metric tracking (Time, Throughput, Memory).
- Regression detection based on thresholds.
- Alerting (Info, Warning, Critical).
- Data storage (JSON history) and reporting (Markdown, CSV).

## Usage

### In Python Scripts

```python
from src.inference_pio.core.tools.scripts.benchmarking.performance_tracker import record_performance_metric, save_regression_data

# Record a metric
record_performance_metric(
    name="inference_tps",
    value=15.5,
    unit="tokens/sec",
    model_name="Qwen3-VL-2B",
    category="performance"
)

# Save data at the end of the run
save_regression_data()
```

### Configuration

The tracker uses default settings but can be configured programmatically or via `performance_regression_config.ini` (if implemented).

- **Threshold**: Default 5% change triggers a regression alert.
- **Storage**: Defaults to `performance_history/`.

## Metric Categories

1.  **Performance** (e.g., TPS): Higher is better. Regression if value decreases.
2.  **Time/Latency** (e.g., ms): Lower is better. Regression if value increases.
3.  **Memory**: Lower is better. Regression if value increases.

## Reports

Reports are generated in `performance_reports/` and history is stored in `performance_history/`.
