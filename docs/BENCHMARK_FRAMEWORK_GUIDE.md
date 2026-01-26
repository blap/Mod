# Benchmark Framework

## Overview

The Inference-PIO project includes a comprehensive benchmark framework designed to measure performance, resource usage, and scalability across different models and components. The framework provides automated discovery, execution, and reporting of benchmarks.

## Architecture

The benchmark framework consists of three main components:

1. **Discovery System**: Automatically discovers benchmark functions across the project
2. **Execution Engine**: Runs benchmarks with configurable parameters
3. **Reporting System**: Generates detailed reports and metrics

## Standardized Structure

Benchmarks follow the same standardized directory structure as tests:

```
benchmarks/
├── unit/
├── integration/
└── performance/

src/inference_pio/models/{model_name}/benchmarks/
├── unit/
├── integration/
└── performance/
```

### Unit Benchmarks
- Measure performance of individual functions or methods
- Focus on micro-benchmarks and algorithmic efficiency
- Typically run quickly with minimal resource usage

### Integration Benchmarks
- Measure performance of component interactions
- Evaluate end-to-end workflow performance
- Include realistic data sizes and usage patterns

### Performance Benchmarks
- Measure system-wide performance characteristics
- Evaluate scalability under various loads
- Monitor resource usage and throughput

## Benchmark Discovery

The benchmark discovery system automatically finds benchmark functions using the following criteria:

- Function names must start with `run_` or `benchmark_`
- Functions must be defined in benchmark directories
- Functions must be properly decorated or follow naming conventions

### Discovery Algorithm

1. **Path Scanning**: Scans predefined benchmark directories
2. **File Analysis**: Identifies Python files containing benchmark functions
3. **Function Extraction**: Finds functions matching benchmark patterns
4. **Metadata Collection**: Gathers information about benchmark categories and models
5. **Validation**: Ensures benchmark functions are properly defined

## Writing Benchmarks

### Basic Benchmark Function

```python
import time
from src.inference_pio.test_utils import assert_true

def run_basic_performance_benchmark():
    """A basic performance benchmark example."""
    start_time = time.time()
    
    # Perform the operation to benchmark
    result = perform_operation()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Verify the operation completed successfully
    assert_true(result is not None, "Operation should return a result")
    
    # Report performance metrics
    print(f"Execution time: {execution_time:.4f} seconds")
    
    return {
        'execution_time': execution_time,
        'result_size': len(result) if hasattr(result, '__len__') else 1
    }
```

### Memory Usage Benchmark

```python
import psutil
import os

def run_memory_usage_benchmark():
    """Benchmark memory usage of a specific operation."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Perform operation to benchmark
    result = perform_memory_intensive_operation()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = final_memory - initial_memory
    
    print(f"Memory used: {memory_used:.2f} MB")
    
    return {
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
        'memory_used_mb': memory_used
    }
```

### Scalability Benchmark

```python
def run_scalability_benchmark():
    """Benchmark performance with varying input sizes."""
    results = {}
    
    for size in [100, 1000, 10000]:
        start_time = time.time()
        
        # Process data of specified size
        result = process_data_of_size(size)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        results[f'size_{size}'] = {
            'time': execution_time,
            'throughput': size / execution_time if execution_time > 0 else float('inf')
        }
        
        print(f"Size {size}: {execution_time:.4f}s, Throughput: {results[f'size_{size}']['throughput']:.2f} items/s")
    
    return results
```

## Running Benchmarks

### All Project Benchmarks

```python
from benchmarks.discovery import discover_and_run_all_benchmarks

# Discover and run all benchmarks in the project
results = discover_and_run_all_benchmarks()
```

### Model-Specific Benchmarks

```python
from benchmarks.discovery import BenchmarkDiscovery

# Initialize discovery
discovery = BenchmarkDiscovery()

# Run benchmarks for a specific model
model_results = discovery.run_model_benchmarks('qwen3_vl_2b')
```

### Category-Specific Benchmarks

```python
from benchmarks.discovery import BenchmarkDiscovery

# Initialize discovery
discovery = BenchmarkDiscovery()

# Run only performance benchmarks
perf_results = discovery.run_all_benchmarks(include_categories=['performance'])
```

## Benchmark Results

The framework generates comprehensive results including:

- **Execution Time**: Time taken to execute benchmarks
- **Resource Usage**: Memory, CPU, and other resource metrics
- **Throughput**: Operations per second or similar metrics
- **Success Rates**: Percentage of successful benchmark runs
- **Statistical Measures**: Mean, median, percentiles for performance metrics

### Result Format

Benchmark results follow a standardized format:

```python
{
    'timestamp': '2023-10-01T12:00:00Z',
    'model': 'qwen3_vl_2b',  # Optional, for model-specific benchmarks
    'results': {
        'benchmark_function_name': {
            'execution_time': 0.123,
            'memory_used_mb': 45.67,
            'throughput': 800.0,
            # ... other metrics specific to the benchmark
        },
        # ... more benchmark results
    },
    'summary': {
        'total_benchmarks': 5,
        'successful_runs': 5,
        'failed_runs': 0,
        'success_rate': 1.0
    }
}
```

## Reporting and Storage

### Automatic Saving

Benchmark results are automatically saved to:

- **JSON Files**: Detailed results in JSON format
- **CSV Files**: Summary statistics in CSV format
- **Timestamped Directories**: Organized by execution time

### File Locations

- Results are stored in `benchmark_results/` directory
- Each execution creates a timestamped subdirectory
- Both raw results and summaries are preserved

## Performance Monitoring

The framework supports continuous performance monitoring:

- **Baseline Comparisons**: Compare results against known baselines
- **Regression Detection**: Identify performance regressions
- **Trend Analysis**: Track performance over time
- **Alerting**: Notify when performance thresholds are exceeded

## Best Practices

### Writing Effective Benchmarks

- **Realistic Workloads**: Use data and operations that reflect real usage
- **Consistent Conditions**: Ensure benchmarks run under consistent conditions
- **Statistical Significance**: Run benchmarks multiple times for statistical validity
- **Resource Cleanup**: Clean up resources after benchmark completion
- **Clear Metrics**: Define clear, measurable performance metrics
- **Documentation**: Document what each benchmark measures and why

### Benchmark Organization

- **Logical Grouping**: Group related benchmarks together
- **Clear Naming**: Use descriptive names that indicate what is being measured
- **Appropriate Categories**: Place benchmarks in the correct category (unit, integration, performance)
- **Version Tracking**: Track benchmark versions alongside code changes

### Performance Considerations

- **Warm-up Periods**: Allow systems to warm up before measuring performance
- **Isolation**: Run benchmarks in isolated environments when possible
- **Resource Monitoring**: Monitor system resources during benchmark execution
- **Repeat Measurements**: Take multiple measurements to account for variance

## Integration with CI/CD

The benchmark framework integrates with CI/CD pipelines:

- **Automated Execution**: Run benchmarks automatically during builds
- **Threshold Checking**: Fail builds if performance thresholds are not met
- **Historical Comparison**: Compare results with historical data
- **Reporting**: Generate reports for performance dashboards
