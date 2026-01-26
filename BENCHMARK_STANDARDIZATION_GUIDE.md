# Inference-PIO Benchmark Standardization Guide

## Table of Contents

1. [Overview](#overview)
2. [Benchmark Architecture](#benchmark-architecture)
3. [Benchmark Categories](#benchmark-categories)
4. [Implementing Benchmarks](#implementing-benchmarks)
5. [Running Benchmarks](#running-benchmarks)
6. [Results Format](#results-format)
7. [Best Practices](#best-practices)
8. [Extending the System](#extending-the-system)
9. [Integration with Testing](#integration-with-testing)
10. [Performance Regression Tracking](#performance-regression-tracking)

## Overview

The benchmark system provides a consistent interface for evaluating model performance across multiple dimensions:

- Performance (speed, memory usage, throughput)
- Accuracy (correctness of outputs)
- Resource utilization (CPU, memory, loading time)
- Scalability (performance under varying loads)

The system is designed to be standardized across all models in the project, ensuring fair and consistent comparisons.

## Benchmark Architecture

### Core Components

1. **Benchmark Interface (`benchmark_interface.py`)**: Defines the standard interface that all benchmarks must implement
2. **Benchmark Runner (`standardized_runner.py`)**: Orchestrates benchmark execution across all models
3. **Benchmark Templates (`benchmark_template.py`)**: Provides templates for implementing model-specific benchmarks
4. **Results Aggregator**: Collects and analyzes benchmark results across models

### Benchmark Interface

All benchmarks must implement the following interface:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BenchmarkResult:
    """Represents the result of a benchmark run."""
    name: str
    value: float
    unit: str
    metadata: Dict[str, Any]
    model_name: str
    category: str

class BaseBenchmark(ABC):
    """Abstract base class for all benchmarks."""
    
    def __init__(self, model_plugin, model_name: str):
        self.model_plugin = model_plugin
        self.model_name = model_name
    
    @abstractmethod
    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return results."""
        pass
    
    def setup(self) -> None:
        """Setup method called before benchmark execution."""
        pass
    
    def teardown(self) -> None:
        """Teardown method called after benchmark execution."""
        pass
```

## Benchmark Categories

### Performance Benchmarks

Performance benchmarks measure operational characteristics:

#### InferenceSpeedBenchmark
Measures tokens per second for various input lengths:

```python
from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult

class InferenceSpeedBenchmark(BaseBenchmark):
    def run(self) -> BenchmarkResult:
        import time
        
        # Warmup
        self.model_plugin.infer("warmup text")
        
        # Benchmark
        start_time = time.time()
        for _ in range(100):
            self.model_plugin.infer("test input for speed measurement")
        end_time = time.time()
        
        total_time = end_time - start_time
        tokens_per_second = 100 / total_time  # Assuming 1 token per inference
        
        return BenchmarkResult(
            name="inference_speed",
            value=tokens_per_second,
            unit="tokens/sec",
            metadata={
                "iterations": 100,
                "total_time": total_time
            },
            model_name=self.model_name,
            category="performance"
        )
```

#### MemoryUsageBenchmark
Measures memory consumption during model operations:

```python
import psutil
import gc
from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult

class MemoryUsageBenchmark(BaseBenchmark):
    def run(self) -> BenchmarkResult:
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operation
        result = self.model_plugin.infer("long input text" * 100)
        
        # Get peak memory usage
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Cleanup
        del result
        gc.collect()
        
        return BenchmarkResult(
            name="memory_usage",
            value=memory_increase,
            unit="MB",
            metadata={
                "initial_memory": initial_memory,
                "peak_memory": peak_memory
            },
            model_name=self.model_name,
            category="memory"
        )
```

#### BatchProcessingBenchmark
Evaluates throughput with different batch sizes:

```python
import time
from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult

class BatchProcessingBenchmark(BaseBenchmark):
    def __init__(self, model_plugin, model_name: str, batch_size: int = 32):
        super().__init__(model_plugin, model_name)
        self.batch_size = batch_size
    
    def run(self) -> BenchmarkResult:
        import torch
        
        # Create batch of inputs
        inputs = [f"test input {_}" for _ in range(self.batch_size)]
        
        # Warmup
        for inp in inputs[:5]:
            self.model_plugin.infer(inp)
        
        # Benchmark
        start_time = time.time()
        for inp in inputs:
            self.model_plugin.infer(inp)
        end_time = time.time()
        
        total_time = end_time - start_time
        requests_per_second = len(inputs) / total_time
        
        return BenchmarkResult(
            name=f"batch_processing_{self.batch_size}",
            value=requests_per_second,
            unit="requests/sec",
            metadata={
                "batch_size": self.batch_size,
                "total_requests": len(inputs),
                "total_time": total_time
            },
            model_name=self.model_name,
            category="performance"
        )
```

#### ModelLoadingTimeBenchmark
Times how long it takes to load a model:

```python
import time
from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult

class ModelLoadingTimeBenchmark(BaseBenchmark):
    def run(self) -> BenchmarkResult:
        start_time = time.time()
        
        # Reload the model to measure loading time
        self.model_plugin.load_model()
        
        loading_time = time.time() - start_time
        
        return BenchmarkResult(
            name="loading_time",
            value=loading_time,
            unit="seconds",
            metadata={
                "model_path": getattr(self.model_plugin, 'model_path', 'unknown')
            },
            model_name=self.model_name,
            category="performance"
        )
```

### Accuracy Benchmarks

Accuracy benchmarks evaluate correctness of model outputs:

#### AccuracyBenchmark
Evaluates correctness of model outputs using known facts:

```python
from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult

class AccuracyBenchmark(BaseBenchmark):
    def run(self) -> BenchmarkResult:
        test_cases = [
            {"input": "What is 2+2?", "expected": "4"},
            {"input": "Capital of France?", "expected": "Paris"},
            {"input": "How many continents?", "expected": "7"},
        ]
        
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            response = self.model_plugin.infer(case["input"])
            
            # Simple string matching (could be enhanced with semantic similarity)
            if case["expected"].lower() in response.lower():
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return BenchmarkResult(
            name="accuracy",
            value=accuracy,
            unit="ratio",
            metadata={
                "correct": correct,
                "total": total,
                "test_cases": len(test_cases)
            },
            model_name=self.model_name,
            category="accuracy"
        )
```

## Implementing Benchmarks

### For New Models

To implement benchmarks for a new model:

1. Ensure your model plugin implements the `ModelPluginInterface`
2. Create benchmark files in the standard directory structure:

```
src/inference_pio/models/{model_name}/benchmarks/
├── unit/
│   └── benchmark_accuracy.py
├── integration/
│   └── benchmark_comparison.py
└── performance/
    └── benchmark_inference_speed.py
```

3. Use the standardized benchmark classes from `benchmark_interface.py`

### Standard Benchmark Class Structure

```python
from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult

class MyModelBenchmark(BaseBenchmark):
    def __init__(self, model_plugin, model_name: str, custom_param: str = None):
        super().__init__(model_plugin, model_name)
        self.custom_param = custom_param
    
    def setup(self) -> None:
        """Setup method called before benchmark execution."""
        # Initialize any resources needed for the benchmark
        print(f"Setting up benchmark for {self.model_name}")
    
    def run(self) -> BenchmarkResult:
        """Execute the benchmark and return results."""
        # Implementation here
        value = self.execute_benchmark_logic()
        return BenchmarkResult(
            name="my_metric",
            value=value,
            unit="units",
            metadata={"param_used": self.custom_param},
            model_name=self.model_name,
            category="performance"
        )
    
    def teardown(self) -> None:
        """Teardown method called after benchmark execution."""
        # Clean up any resources used by the benchmark
        print(f"Tearing down benchmark for {self.model_name}")
    
    def execute_benchmark_logic(self) -> float:
        """Execute the actual benchmark logic."""
        # Your benchmark implementation here
        return 123.45
```

### Example: Complete Benchmark Implementation

```python
import time
import torch
import psutil
from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult

class ComprehensiveModelBenchmark(BaseBenchmark):
    def __init__(self, model_plugin, model_name: str, input_size: int = 100):
        super().__init__(model_plugin, model_name)
        self.input_size = input_size
    
    def setup(self) -> None:
        """Prepare for benchmark execution."""
        # Pre-generate test inputs
        self.test_inputs = [f"test input {_}" for _ in range(self.input_size)]
        
        # Get initial system stats
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    def run(self) -> BenchmarkResult:
        """Execute comprehensive benchmark."""
        import gc
        
        # Warmup phase
        for _ in range(5):
            self.model_plugin.infer("warmup")
        
        # Performance measurement
        start_time = time.time()
        for inp in self.test_inputs:
            result = self.model_plugin.infer(inp)
        end_time = time.time()
        
        execution_time = end_time - start_time
        requests_per_second = len(self.test_inputs) / execution_time
        
        # Memory measurement
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - self.initial_memory
        
        # Cleanup
        del result
        gc.collect()
        
        # Return comprehensive results
        return BenchmarkResult(
            name="comprehensive_benchmark",
            value=requests_per_second,
            unit="requests/sec",
            metadata={
                "requests_per_second": requests_per_second,
                "memory_used_mb": memory_used,
                "total_time": execution_time,
                "input_count": len(self.test_inputs),
                "avg_time_per_request": execution_time / len(self.test_inputs)
            },
            model_name=self.model_name,
            category="performance"
        )
    
    def teardown(self) -> None:
        """Clean up after benchmark execution."""
        # Clean up any temporary resources
        if hasattr(self, 'test_inputs'):
            del self.test_inputs
```

## Running Benchmarks

### Single Model

```bash
python -m pytest src/inference_pio/models/{model_name}/benchmarks/ -v
```

### All Models

```python
from benchmarks.standardized_runner import run_standardized_benchmarks

# Run only performance benchmarks
results = run_standardized_benchmarks(benchmark_suite='performance')

# Run only accuracy benchmarks
results = run_standardized_benchmarks(benchmark_suite='accuracy')

# Run all benchmarks (default)
results = run_standardized_benchmarks(benchmark_suite='full')
```

### Using the Unified Discovery System

```python
from inference_pio.unified_test_discovery import discover_and_run_benchmarks_only

# Discover and run only benchmarks
results = discover_and_run_benchmarks_only()
```

### Model-Specific Benchmark Execution

```python
from inference_pio.unified_test_discovery import run_benchmarks_for_model

# Run benchmarks for a specific model
results = run_benchmarks_for_model('qwen3_vl_2b')
```

### Programmatic Benchmark Execution

```python
from inference_pio.common.benchmark_interface import BenchmarkResult

def run_specific_benchmark(benchmark_class, model_plugin, model_name):
    """Run a specific benchmark class."""
    benchmark = benchmark_class(model_plugin, model_name)
    
    try:
        benchmark.setup()
        result = benchmark.run()
        return result
    finally:
        benchmark.teardown()

# Example usage
model_plugin = create_model_plugin()
benchmark_result = run_specific_benchmark(
    InferenceSpeedBenchmark, 
    model_plugin, 
    "my_model"
)
print(f"Benchmark result: {benchmark_result.value} {benchmark_result.unit}")
```

## Results Format

Benchmark results are saved in both JSON and CSV formats:

### JSON Format
```json
{
  "timestamp": "2023-10-15T10:30:45.123456",
  "model_name": "qwen3_vl_2b",
  "benchmark_name": "inference_speed",
  "value": 150.5,
  "unit": "tokens/sec",
  "category": "performance",
  "metadata": {
    "iterations": 100,
    "total_time": 0.664,
    "warmup_iterations": 5
  }
}
```

### CSV Format
```
model_name,benchmark_name,value,unit,category,timestamp
qwen3_vl_2b,inference_speed,150.5,tokens/sec,performance,2023-10-15T10:30:45.123456
```

### Results Storage

Files are saved to the `benchmark_results/` directory with timestamps:
- JSON: Detailed results with metadata
- CSV: Summary results for easy analysis

## Best Practices

### 1. Use Standardized Benchmark Classes
Always use the standardized benchmark classes to ensure consistency:

```python
# Good
from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult

class MyBenchmark(BaseBenchmark):
    def run(self) -> BenchmarkResult:
        # Implementation using standard interface
        pass

# Avoid
class MyBenchmark:
    def execute(self):
        # Custom interface not compatible with system
        pass
```

### 2. Include Warmup Runs
Stabilize timing measurements with warmup runs:

```python
def run(self) -> BenchmarkResult:
    # Warmup to stabilize measurements
    for _ in range(5):
        self.model_plugin.infer("warmup input")
    
    # Actual benchmark
    start_time = time.time()
    for _ in range(100):
        self.model_plugin.infer("test input")
    end_time = time.time()
    
    # Calculate results
    execution_time = end_time - start_time
    requests_per_second = 100 / execution_time
    
    return BenchmarkResult(...)
```

### 3. Test Various Input Sizes
Get comprehensive performance profiles:

```python
def run_multiple_input_sizes(self):
    """Run benchmark with various input sizes."""
    input_sizes = [10, 50, 100, 500, 1000]
    results = []
    
    for size in input_sizes:
        benchmark = BatchProcessingBenchmark(
            self.model_plugin, 
            self.model_name, 
            batch_size=size
        )
        result = benchmark.run()
        results.append(result)
    
    return results
```

### 4. Use CPU for Consistency
Ensure consistent benchmarks across different hardware:

```python
def setup(self) -> None:
    """Ensure consistent hardware usage."""
    # Force CPU usage for consistent benchmarks
    if torch.cuda.is_available():
        torch.cuda.set_device(-1)  # Use CPU
```

### 5. Implement Proper Error Handling
Prevent benchmark failures:

```python
def run(self) -> BenchmarkResult:
    try:
        # Benchmark implementation
        result_value = self.perform_measurement()
        return BenchmarkResult(
            name="my_benchmark",
            value=result_value,
            unit="units",
            metadata={},
            model_name=self.model_name,
            category="performance"
        )
    except Exception as e:
        # Log error and return appropriate result
        print(f"Benchmark failed: {e}")
        return BenchmarkResult(
            name="my_benchmark",
            value=0.0,
            unit="units",
            metadata={"error": str(e)},
            model_name=self.model_name,
            category="performance"
        )
```

### 6. Clean Up Resources
Properly clean up after benchmark completion:

```python
def teardown(self) -> None:
    """Clean up resources after benchmark."""
    # Free any allocated resources
    if hasattr(self, 'large_tensor'):
        del self.large_tensor
    if hasattr(self, 'temporary_file'):
        import os
        if os.path.exists(self.temporary_file):
            os.remove(self.temporary_file)
    
    # Force garbage collection
    import gc
    gc.collect()
```

## Extending the System

### Creating New Benchmark Types

New benchmark types can be added by extending the `BaseBenchmark` class:

```python
from inference_pio.common.benchmark_interface import BaseBenchmark, BenchmarkResult

class CustomBenchmark(BaseBenchmark):
    def __init__(self, model_plugin, model_name: str, custom_param: str = None):
        super().__init__(model_plugin, model_name)
        self.custom_param = custom_param

    def run(self) -> BenchmarkResult:
        # Custom benchmark logic here
        value = self.execute_custom_logic()
        return BenchmarkResult(
            name="custom_metric",
            value=value,
            unit="custom_unit",
            metadata={"param_used": self.custom_param},
            model_name=self.model_name,
            category="custom"
        )
    
    def execute_custom_logic(self) -> float:
        """Execute the custom benchmark logic."""
        # Implementation of custom measurement
        return 42.0
```

### Registering New Benchmark Types

Register new benchmark types with the appropriate suite in `benchmark_interface.py`:

```python
# In benchmark_interface.py
BENCHMARK_REGISTRY = {
    'inference_speed': InferenceSpeedBenchmark,
    'memory_usage': MemoryUsageBenchmark,
    'accuracy': AccuracyBenchmark,
    'custom_metric': CustomBenchmark,  # Add your custom benchmark
}

def get_benchmark_class(benchmark_name: str):
    """Get benchmark class by name."""
    return BENCHMARK_REGISTRY.get(benchmark_name)
```

### Custom Benchmark Categories

Define custom categories for specialized benchmarks:

```python
from enum import Enum

class BenchmarkCategory(Enum):
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    MEMORY = "memory"
    CUSTOM = "custom"  # Add custom category
```

## Integration with Testing

### Combined Test and Benchmark Execution

The unified discovery system allows running tests and benchmarks together:

```python
from inference_pio.unified_test_discovery import discover_and_run_all_items

# Discover and run both tests and benchmarks
results = discover_and_run_all_items()
```

### Performance Regression Testing

Benchmarks integrate with performance regression tracking:

```python
from src.inference_pio.common.performance_regression_tests import PerformanceRegressionTestCase

class ModelBenchmarkRegressionTest(PerformanceRegressionTestCase):
    def test_inference_speed_regression(self):
        """Test for inference speed regressions."""
        benchmark = InferenceSpeedBenchmark(self.model_plugin, self.model_name)
        result = benchmark.run()
        
        # Record the performance metric
        self.record_performance_metric(
            name=result.name,
            value=result.value,
            unit=result.unit,
            model_name=result.model_name,
            category=result.category,
            metadata=result.metadata
        )
        
        # Assert no regression
        self.assert_no_performance_regression(
            result.model_name, 
            result.name,
            threshold=0.05  # 5% threshold
        )
```

### CI/CD Integration

Benchmarks can be integrated into CI/CD pipelines:

```bash
# Run benchmarks in CI with failure threshold
python scripts/ci_benchmark_runner.py --threshold 5.0 --fail-on-regression
```

## Performance Regression Tracking

### Automatic Regression Detection

The system automatically detects performance regressions:

```python
from src.inference_pio.common.performance_regression_tests import PerformanceRegressionTracker

tracker = PerformanceRegressionTracker()

# Record benchmark results
benchmark_result = my_benchmark.run()
tracker.record_metric(
    name=benchmark_result.name,
    value=benchmark_result.value,
    unit=benchmark_result.unit,
    model_name=benchmark_result.model_name,
    category=benchmark_result.category
)

# Check for regressions
regressions = tracker.detect_regressions(threshold=0.05)
if regressions:
    print("Performance regressions detected:", regressions)
```

### Historical Data

The system maintains historical performance data:

- Stores previous benchmark results
- Calculates statistical measures (mean, std dev)
- Identifies trends over time
- Generates performance reports

### Alerting System

Automatic alerts for performance degradation:

```python
from src.inference_pio.common.performance_regression_tests import RegressionSeverity

alerts = tracker.get_recent_alerts()
for alert in alerts:
    if alert.severity == RegressionSeverity.CRITICAL:
        print(f"CRITICAL: {alert.message}")
        # Trigger notification or fail build
```

---

This benchmark standardization system provides a consistent, extensible framework for evaluating model performance across the Inference-PIO project. Following these guidelines ensures fair comparisons and reliable performance tracking.