# Inference-PIO Performance Regression Testing System

## Table of Contents

1. [Overview](#overview)
2. [Components](#components)
3. [Usage](#usage)
4. [Configuration](#configuration)
5. [Metrics Categories](#metrics-categories)
6. [Regression Detection](#regression-detection)
7. [Data Storage](#data-storage)
8. [Reports](#reports)
9. [Best Practices](#best-practices)
10. [Integration](#integration)
11. [API Reference](#api-reference)

## Overview

The performance regression testing system tracks performance metrics over time, detects performance regressions, and provides alerts when performance degrades. The system integrates with the existing test infrastructure and CI/CD pipeline to ensure consistent performance monitoring across the Inference-PIO project.

The system is designed to be used alongside the test optimization and unified discovery systems to provide comprehensive performance monitoring capabilities.

## Components

### 1. Performance Regression Tracker

The `PerformanceRegressionTracker` class manages historical performance data and detects regressions:

- Tracks performance metrics over time
- Detects regressions based on configurable thresholds
- Maintains historical statistics
- Generates reports and exports data
- Provides alerting capabilities

### 2. Performance Regression Tests

The `PerformanceRegressionTestCase` provides a framework for writing performance regression tests that integrate with the test utilities framework.

### 3. Pytest Plugin

A pytest plugin that enables performance regression testing within the pytest framework (when used with pytest integration).

### 4. CI/CD Integration

Scripts and configurations for integrating performance regression testing into CI/CD pipelines.

## Usage

### Running Performance Regression Tests

```bash
# Run all performance regression tests
python -m src.inference_pio.common.performance_regression_tests

# Run with pytest (if pytest integration is enabled)
pytest tests/performance/ -v --performance-regression --perf-threshold 5.0
```

### Using in Test Cases

```python
from src.inference_pio.common.performance_regression_tests import PerformanceRegressionTestCase

class MyModelPerformanceTest(PerformanceRegressionTestCase):
    def test_inference_speed(self):
        import time
        
        # Your performance test code here
        start_time = time.time()
        result = self.model.infer("test input")
        end_time = time.time()
        
        execution_time = end_time - start_time
        tokens_per_second = len("test input") / execution_time
        
        # Record the performance metric
        self.record_performance_metric(
            name="inference_speed",
            value=tokens_per_second,
            unit="tokens/sec",
            model_name="my_model",
            category="performance"
        )

        # Assert no regression (will fail if regression detected)
        self.assert_no_performance_regression("my_model", "inference_speed")
```

### Using the Standalone Tracker

```python
from src.inference_pio.common.performance_regression_tests import (
    PerformanceRegressionTracker, record_performance_metric, get_regression_alerts
)

# Using the default tracker
record_performance_metric(
    name="inference_speed",
    value=150.0,
    unit="tokens/sec",
    model_name="my_model",
    category="performance"
)

# Check for alerts
alerts = get_regression_alerts()
for alert in alerts:
    print(f"Alert: {alert.message}")
```

### CI/CD Integration

```bash
# Run performance tests in CI with failure threshold
python scripts/ci_performance_tests.py --threshold 5.0 --fail-on-regression

# Or using the unified discovery system
python -c "
from inference_pio.unified_test_discovery import discover_and_run_all_items
from src.inference_pio.common.performance_regression_tests import get_regression_alerts, save_regression_data

results = discover_and_run_all_items()

# Check for performance regressions
alerts = get_regression_alerts()
if any(alert.severity.value == 'critical' for alert in alerts):
    print('Critical performance regressions detected!')
    exit(1)

# Save regression data
save_regression_data()
"
```

## Configuration

The system can be configured using the `performance_regression_config.ini` file:

```ini
[performance_regression]
regression_threshold = 5.0
storage_dir = performance_history
reports_dir = performance_reports
fail_on_regression = true
```

### Configuration Options

- `regression_threshold`: Percentage threshold for detecting regressions (default: 5.0)
- `storage_dir`: Directory to store historical performance data (default: performance_history)
- `reports_dir`: Directory to store generated reports (default: performance_reports)
- `fail_on_regression`: Whether to fail when regressions are detected (default: true)

### Programmatic Configuration

```python
from src.inference_pio.common.performance_regression_tests import PerformanceRegressionTracker

# Create tracker with custom configuration
tracker = PerformanceRegressionTracker(
    storage_dir="custom_performance_history",
    regression_threshold=10.0  # 10% threshold
)
```

## Metrics Categories

The system recognizes different categories of metrics, each with different regression detection logic:

### Performance Metrics
- **Higher is better**: Metrics where higher values indicate better performance
- Examples: tokens/sec, requests/sec, throughput
- Regression: Value decreases by more than threshold percentage

### Time Metrics
- **Lower is better**: Metrics where lower values indicate better performance
- Examples: seconds, milliseconds, execution time
- Regression: Value increases by more than threshold percentage

### Memory Metrics
- **Lower is better**: Metrics where lower values indicate better performance
- Examples: MB, GB, memory usage
- Regression: Value increases by more than threshold percentage

### Custom Categories
You can define custom categories with specific regression logic:

```python
from src.inference_pio.common.performance_regression_tests import PerformanceRegressionTracker

class CustomTracker(PerformanceRegressionTracker):
    def _determine_regression_direction(self, category: str, unit: str) -> str:
        """
        Determine if higher values are better ('higher') or lower values are better ('lower').
        
        Args:
            category: Metric category
            unit: Metric unit
            
        Returns:
            'higher' if higher values are better, 'lower' if lower values are better
        """
        if category == "efficiency":
            return "higher"  # Higher efficiency is better
        elif category == "latency":
            return "lower"   # Lower latency is better
        else:
            # Use default logic based on category
            return super()._determine_regression_direction(category, unit)
```

## Regression Detection

### Detection Algorithm

The system detects regressions based on:

1. **Percentage Change**: Compare current value to previous measurements
2. **Metric Category**: Apply appropriate logic based on whether higher/lower is better
3. **Configurable Thresholds**: Use configured percentage thresholds
4. **Severity Levels**: Assign severity based on magnitude of regression

### Severity Levels

- **INFO**: Minor regressions (below critical threshold)
- **WARNING**: Moderate regressions (above warning threshold)
- **CRITICAL**: Major regressions (above critical threshold)

### Example Detection Logic

```python
def check_regression(self, metric_key: str, current_metric: PerformanceMetric) -> None:
    # Get previous value
    previous_metric = self.get_previous_metric(metric_key)
    current_value = current_metric.value
    previous_value = previous_metric.value
    
    # Calculate percentage change
    if previous_value != 0:
        percentage_change = ((current_value - previous_value) / abs(previous_value)) * 100
    else:
        percentage_change = 0
    
    # Determine if regression based on category
    is_regression = False
    severity = RegressionSeverity.INFO
    
    if current_metric.category == "performance":
        if current_metric.unit in ["tokens/sec", "requests/sec", "throughput"]:
            # Higher values are better, so negative change is regression
            if percentage_change < -self.regression_threshold:
                is_regression = True
                if percentage_change < -2 * self.regression_threshold:
                    severity = RegressionSeverity.CRITICAL
                else:
                    severity = RegressionSeverity.WARNING
        elif current_metric.unit in ["seconds", "ms", "time"]:
            # Lower values are better, so positive change is regression
            if percentage_change > self.regression_threshold:
                is_regression = True
                if percentage_change > 2 * self.regression_threshold:
                    severity = RegressionSeverity.CRITICAL
                else:
                    severity = RegressionSeverity.WARNING
```

## Data Storage

### Historical Data

Historical performance data is stored in JSON format in the `performance_history` directory:

#### Metrics History
```json
{
  "my_model:inference_speed": [
    {
      "name": "inference_speed",
      "value": 150.0,
      "unit": "tokens/sec",
      "timestamp": 1697385600.0,
      "model_name": "my_model",
      "category": "performance",
      "metadata": {
        "test_environment": "local",
        "hardware": "RTX 3090"
      }
    },
    {
      "name": "inference_speed",
      "value": 145.0,
      "unit": "tokens/sec",
      "timestamp": 1697385700.0,
      "model_name": "my_model",
      "category": "performance",
      "metadata": {
        "test_environment": "local",
        "hardware": "RTX 3090"
      }
    }
  ]
}
```

#### Regression Alerts
```json
{
  "metric_name": "inference_speed",
  "model_name": "my_model",
  "previous_value": 150.0,
  "current_value": 145.0,
  "threshold_percentage": 5.0,
  "severity": "warning",
  "timestamp": 1697385700.0,
  "message": "Performance regression detected for my_model:inference_speed. Previous: 150.00tokens/sec, Current: 145.00tokens/sec, Change: -3.33%"
}
```

### Storage Management

The system automatically manages storage by:

- Rotating old data files periodically
- Compressing historical data
- Maintaining only recent history for active metrics
- Providing cleanup utilities

### Backup and Recovery

```python
from src.inference_pio.common.performance_regression_tests import PerformanceRegressionTracker

tracker = PerformanceRegressionTracker()

# Backup current data
backup_data = tracker.export_data()

# Restore from backup
tracker.import_data(backup_data)

# Or load from specific files
tracker.load_history_from_file("backup_metrics.json")
```

## Reports

### Performance Reports

The system generates comprehensive performance reports in Markdown format:

```markdown
# Performance Report

Generated on: 2023-10-15T10:30:45.123456

## Summary

- Total metrics tracked: 150
- Unique metrics: 25
- Regression alerts: 3
- Critical alerts: 1

## Recent Regression Alerts

- **CRITICAL**: Performance regression detected for my_model:inference_speed. Previous: 150.00tokens/sec, Current: 140.00tokens/sec, Change: -6.67%
- **WARNING**: Performance regression detected for another_model:memory_usage. Previous: 500.00MB, Current: 550.00MB, Change: +10.00%

## Metrics History

### my_model:inference_speed
- Latest value: 140.00 tokens/sec
- Historical mean: 148.50 tokens/sec
- Historical stdev: 2.34 tokens/sec
- Min: 140.00, Max: 152.00
- Total measurements: 25

### another_model:memory_usage
- Latest value: 550.00 MB
- Historical mean: 510.00 MB
- Historical stdev: 15.20 MB
- Min: 490.00, Max: 550.00
- Total measurements: 30
```

### CSV Exports

For data analysis, the system provides CSV exports:

```csv
Model Name,Metric Name,Value,Unit,Category,Timestamp,Date Time
my_model,inference_speed,140.0,tokens/sec,performance,1697385600.0,2023-10-15T10:30:00
another_model,memory_usage,550.0,MB,memory,1697385660.0,2023-10-15T10:31:00
```

### Report Generation

```python
from src.inference_pio.common.performance_regression_tests import PerformanceRegressionTracker

tracker = PerformanceRegressionTracker()

# Generate comprehensive report
report_path = tracker.generate_report(output_dir="custom_reports")

# Export to CSV
csv_path = tracker.export_csv(output_dir="custom_exports")

# Get statistical summary
stats = tracker.get_historical_stats("my_model:inference_speed")
print(f"Mean: {stats['mean']}, Std Dev: {stats['stdev']}")
```

## Best Practices

### 1. Establish Baselines Before Changes

Always establish performance baselines before making changes:

```python
def test_performance_baseline():
    """Establish performance baseline before making changes."""
    import time
    
    # Run multiple iterations for stable baseline
    times = []
    for _ in range(10):
        start = time.time()
        result = model.infer("test input")
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    
    # Record baseline
    record_performance_metric(
        name="baseline_inference_time",
        value=avg_time,
        unit="seconds",
        model_name="my_model",
        category="performance",
        metadata={"baseline": True}
    )
```

### 2. Run Performance Tests Regularly

Schedule regular performance testing:

```python
# In CI/CD pipeline
def run_regular_performance_tests():
    """Run performance tests regularly."""
    # Run on schedule (daily, weekly, etc.)
    results = discover_and_run_all_items()
    
    # Check for regressions
    alerts = get_regression_alerts()
    if alerts:
        print(f"Performance alerts: {len(alerts)}")
        for alert in alerts:
            print(f"  {alert.severity.value.upper()}: {alert.message}")
    
    # Save data
    save_regression_data()
```

### 3. Monitor Trends Over Time

Focus on trends rather than individual measurements:

```python
def analyze_performance_trends():
    """Analyze performance trends over time."""
    tracker = PerformanceRegressionTracker()
    
    # Get historical data for trend analysis
    history = tracker.get_metric_history("my_model:inference_speed")
    
    # Calculate trend (moving average, regression, etc.)
    if len(history) >= 10:  # Need sufficient data points
        recent_avg = sum(h.value for h in history[-5:]) / 5
        older_avg = sum(h.value for h in history[-10:-5]) / 5
        
        trend = (recent_avg - older_avg) / older_avg * 100
        print(f"Performance trend: {trend:+.2f}%")
```

### 4. Use Appropriate Thresholds

Set appropriate thresholds for different metrics:

```python
# Different thresholds for different metrics
THRESHOLD_CONFIG = {
    "inference_speed": 5.0,    # 5% threshold for speed
    "memory_usage": 10.0,      # 10% threshold for memory
    "startup_time": 15.0,      # 15% threshold for startup time
    "accuracy": 1.0            # 1% threshold for accuracy (critical)
}

def record_with_appropriate_threshold(name, value, unit, model_name, category):
    """Record metric with appropriate threshold."""
    threshold = THRESHOLD_CONFIG.get(name, 5.0)  # Default to 5%
    
    record_performance_metric(
        name=name,
        value=value,
        unit=unit,
        model_name=model_name,
        category=category
    )
    
    # Check regression with custom threshold
    tracker = PerformanceRegressionTracker()
    # Custom logic to check with specific threshold
```

### 5. Investigate Regressions Promptly

Address performance regressions quickly:

```python
def handle_performance_regressions():
    """Handle performance regressions."""
    alerts = get_regression_alerts()
    
    for alert in alerts:
        if alert.severity == RegressionSeverity.CRITICAL:
            print(f"CRITICAL REGRESSION: {alert.message}")
            print("Investigation required immediately!")
            # Trigger notification, issue creation, etc.
        elif alert.severity == RegressionSeverity.WARNING:
            print(f"WARNING: {alert.message}")
            print("Consider investigating")
        else:
            print(f"INFO: {alert.message}")
```

## Integration

### With Test Framework

The system integrates seamlessly with the test utilities framework:

```python
from src.inference_pio.test_utils import assert_true
from src.inference_pio.common.performance_regression_tests import PerformanceRegressionTestCase

class MyPerformanceTest(PerformanceRegressionTestCase):
    def test_model_performance(self):
        """Test model performance with regression tracking."""
        import time
        
        start_time = time.time()
        result = self.model.infer("test input")
        execution_time = time.time() - start_time
        
        # Record performance metric
        self.record_performance_metric(
            name="inference_time",
            value=execution_time,
            unit="seconds",
            model_name="test_model",
            category="performance"
        )
        
        # Verify result is valid (functional test)
        assert_true(result is not None, "Model should return valid result")
        
        # Check for performance regression
        self.assert_no_performance_regression("test_model", "inference_time")
```

### With Benchmark System

Integrates with the benchmark system:

```python
from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult
from src.inference_pio.common.performance_regression_tests import PerformanceRegressionTracker

class PerformanceBenchmarkWithRegressionTracking(BaseBenchmark):
    def __init__(self, model_plugin, model_name: str):
        super().__init__(model_plugin, model_name)
        self.tracker = PerformanceRegressionTracker()
    
    def run(self) -> BenchmarkResult:
        import time
        
        # Run benchmark
        start_time = time.time()
        for _ in range(100):
            self.model_plugin.infer("test input")
        execution_time = time.time() - start_time
        
        tokens_per_second = 100 / execution_time
        
        # Record with regression tracking
        self.tracker.record_metric(
            name="benchmark_inference_speed",
            value=tokens_per_second,
            unit="tokens/sec",
            model_name=self.model_name,
            category="performance"
        )
        
        return BenchmarkResult(
            name="benchmark_inference_speed",
            value=tokens_per_second,
            unit="tokens/sec",
            metadata={"iterations": 100, "total_time": execution_time},
            model_name=self.model_name,
            category="performance"
        )
```

### With CI/CD Pipeline

Example CI/CD integration:

```yaml
# .github/workflows/performance.yml
name: Performance Tests
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  performance-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install -e ".[dev]"
    
    - name: Run performance tests
      run: |
        python -c "
        from inference_pio.unified_test_discovery import discover_and_run_all_items
        from src.inference_pio.common.performance_regression_tests import get_regression_alerts, save_regression_data
        
        results = discover_and_run_all_items()
        
        # Check for critical regressions
        alerts = get_regression_alerts()
        critical_alerts = [a for a in alerts if a.severity.value == 'critical']
        
        if critical_alerts:
            print('Critical performance regressions detected:')
            for alert in critical_alerts:
                print(f'  {alert.message}')
            exit(1)
        
        # Save regression data
        save_regression_data()
        "
    
    - name: Upload performance reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: performance-reports
        path: performance_reports/
```

### With Monitoring Systems

Integrate with external monitoring:

```python
def send_performance_metrics_to_monitoring():
    """Send performance metrics to external monitoring system."""
    from src.inference_pio.common.performance_regression_tests import get_regression_alerts
    
    alerts = get_regression_alerts()
    
    for alert in alerts:
        # Send to monitoring system (Prometheus, DataDog, etc.)
        send_metric(
            name=f"performance_regression_{alert.severity.value}",
            value=1,
            labels={
                "model": alert.model_name,
                "metric": alert.metric_name,
                "previous_value": alert.previous_value,
                "current_value": alert.current_value
            }
        )
```

## API Reference

### PerformanceRegressionTracker Class

#### Constructor
```python
def __init__(self, storage_dir: str = "performance_history", regression_threshold: float = 5.0):
    """
    Initialize the performance regression tracker.

    Args:
        storage_dir: Directory to store historical performance data
        regression_threshold: Threshold percentage for detecting regressions
    """
```

#### Core Methods
```python
def add_metric(self, metric: PerformanceMetric) -> None:
    """
    Add a new performance metric to the tracker.

    Args:
        metric: PerformanceMetric object to add
    """

def check_regression(self, metric_key: str, current_metric: PerformanceMetric) -> None:
    """
    Check if the current metric represents a regression compared to historical data.

    Args:
        metric_key: Key identifying the metric (model_name:name)
        current_metric: Current metric value to check
    """

def get_historical_stats(self, metric_key: str, window_size: int = 10) -> Dict[str, float]:
    """
    Get statistical information about a metric's historical performance.

    Args:
        metric_key: Key identifying the metric (model_name:name)
        window_size: Number of recent data points to consider

    Returns:
        Dictionary with statistical information
    """
```

#### Storage Methods
```python
def save_history(self) -> None:
    """Save the current metrics history to storage."""

def load_history(self) -> None:
    """Load metrics history from storage."""

def export_data(self) -> Dict[str, Any]:
    """
    Export all data as a dictionary.

    Returns:
        Dictionary containing all tracker data
    """

def import_data(self, data: Dict[str, Any]) -> None:
    """
    Import data from a dictionary.

    Args:
        data: Dictionary containing tracker data to import
    """
```

#### Report Methods
```python
def generate_report(self, output_dir: str = "performance_reports") -> str:
    """
    Generate a comprehensive performance report.

    Args:
        output_dir: Directory to save the report

    Returns:
        Path to the generated report
    """

def export_csv(self, output_dir: str = "performance_exports") -> str:
    """
    Export all metrics to a CSV file.

    Args:
        output_dir: Directory to save the CSV export

    Returns:
        Path to the exported CSV file
    """
```

#### Utility Methods
```python
def reset_alerts(self) -> None:
    """Reset the alerts list."""
```

### PerformanceRegressionTestMixin Class

#### Core Methods
```python
def record_performance_metric(self, name: str, value: float, unit: str,
                              model_name: str, category: str,
                              metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Record a performance metric for regression tracking.

    Args:
        name: Name of the metric
        value: Value of the metric
        unit: Unit of measurement
        model_name: Name of the model being tested
        category: Category of the metric (e.g., 'performance', 'memory')
        metadata: Additional metadata about the measurement
    """

def get_regression_alerts(self) -> List[RegressionAlert]:
    """Get any regression alerts that have been triggered."""

def save_regression_data(self) -> None:
    """Save the regression tracking data."""
```

### Standalone Functions

#### Default Tracker Access
```python
def get_default_tracker() -> PerformanceRegressionTracker:
    """Get the default global performance regression tracker."""

def record_performance_metric(name: str, value: float, unit: str,
                             model_name: str, category: str,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Record a performance metric using the default tracker.

    Args:
        name: Name of the metric
        value: Value of the metric
        unit: Unit of measurement
        model_name: Name of the model being tested
        category: Category of the metric (e.g., 'performance', 'memory')
        metadata: Additional metadata about the measurement
    """

def get_regression_alerts() -> List[RegressionAlert]:
    """Get regression alerts from the default tracker."""

def save_regression_data() -> None:
    """Save regression data from the default tracker."""
```

### Data Classes

#### PerformanceMetric
```python
@dataclass
class PerformanceMetric:
    """Data class to represent a single performance metric."""
    name: str
    value: float
    unit: str
    timestamp: float
    model_name: str
    category: str
    metadata: Optional[Dict[str, Any]] = None
```

#### RegressionAlert
```python
@dataclass
class RegressionAlert:
    """Data class to represent a performance regression alert."""
    metric_name: str
    model_name: str
    previous_value: float
    current_value: float
    threshold_percentage: float
    severity: RegressionSeverity
    timestamp: float
    message: str
```

#### RegressionSeverity
```python
from enum import Enum

class RegressionSeverity(Enum):
    """Enumeration for regression severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
```

---

The Performance Regression Testing System provides comprehensive monitoring and alerting capabilities to ensure the Inference-PIO project maintains optimal performance over time. By tracking metrics, detecting regressions, and generating reports, it helps maintain high-quality standards across all models and components.