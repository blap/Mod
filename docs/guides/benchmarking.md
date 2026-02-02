# Benchmarking Guide

This guide details the benchmarking framework used to measure performance, resource usage, and scalability of models in Inference-PIO.

## 1. Overview

The framework provides automated discovery, execution, and reporting of benchmarks. It is designed to track key metrics like Time To First Token (TTFT), Tokens Per Second (TPS), and Peak Memory Usage.

### Key Components
*   **Discovery System:** Automatically finds `benchmark_*.py` files.
*   **Execution Engine:** Runs benchmarks with configurable parameters (batch size, sequence length).
*   **Hardware Analyzer:** Records system context (CPU, GPU, RAM) to make results comparable.

## 2. Running Benchmarks

Benchmarks are Python scripts located in the `benchmarks/` directory.

### Core Benchmarks
```bash
# Run optimization benchmark (pruning, quantization effects)
python benchmarks/core/benchmark_optimization.py

# Run resize benchmark (image processing)
python benchmarks/core/benchmark_resize.py
```

### Discovery & Execution
To run all available benchmarks or specific subsets:
```bash
# Run all benchmarks using the discovery mechanism
python -c "from benchmarks.core.benchmark_discovery import discover_and_run_all_benchmarks; discover_and_run_all_benchmarks()"
```

## 3. Writing Benchmarks

Benchmarks should be placed in `benchmarks/` or model-specific `benchmarks/` directories.

### Structure
```python
import time
from src.inference_pio.common.hardware_analyzer import get_system_profile

def benchmark_inference():
    # 1. Setup
    model = load_model()

    # 2. Warmup
    model.infer(dummy_input)

    # 3. Measurement
    start = time.perf_counter()
    model.infer(real_input)
    end = time.perf_counter()

    # 4. Reporting
    print(f"Latency: {end - start:.4f}s")
```

### Best Practices
*   **Warmup:** Always run a few iterations before measuring to allow caches and JIT to settle.
*   **Isolation:** Ensure no other heavy processes are running.
*   **Hardware Context:** Use `get_system_profile()` to log *where* the benchmark ran.

## 4. Results & Reporting

Results are automatically saved to `benchmark_results/` in JSON and CSV formats.
Model-specific benchmarks are saved to `src/models/<model>/benchmarks/results/`.
These results are used by the CI/CD pipeline to detect performance regressions.

### Self-Contained Architecture
Each model plugin is completely independent with its own benchmarks located in:
*   `src/models/<model>/benchmarks/unit/` - Unit benchmarks for the specific model
*   `src/models/<model>/benchmarks/integration/` - Integration benchmarks for the specific model
*   `src/models/<model>/benchmarks/performance/` - Performance benchmarks for the specific model
*   `src/models/<model>/benchmarks/results/` - Benchmark results for the specific model

### Output Format
```json
{
  "timestamp": "2023-10-27T10:00:00",
  "model": "qwen3_vl_2b",
  "metric": "throughput",
  "value": 45.2,
  "unit": "tokens/sec",
  "hardware": { ... }
}
```
