# Benchmarking Standards

This document outlines the standardized benchmarking framework and best practices for the Inference-PIO project.

## Directory Structure

Benchmarks are consolidated in the `benchmarks/` directory at the project root.

```
benchmarks/
├── core/                 # Core benchmarking utilities and scripts
│   ├── benchmark_optimization.py
│   └── benchmark_resize.py
├── results/              # Output directory for results (ignored by git)
└── scripts/              # Runner scripts (e.g., standardized_runner.py)
```

## Running Benchmarks

Benchmarks are executed via Python scripts located in the `benchmarks/` directory.

```bash
# Run optimization benchmark
python benchmarks/core/benchmark_optimization.py

# Run resize benchmark
python benchmarks/core/benchmark_resize.py
```

## Benchmark Implementation Guidelines

1.  **Isolation**: Ensure benchmarks isolate the component being measured.
2.  **Repetition**: Run operations multiple times to get statistically significant results.
3.  **Metrics**: Measure Time To First Token (TTFT), Throughput (TPS), and Memory Usage.
4.  **Hardware Awareness**: Use `HardwareAnalyzer` to record the environment context alongside results.
5.  **Output**: Save results to `benchmark_results/` in structured formats (JSON/CSV) for analysis.

## Naming Conventions

- **Files**: `benchmark_{metric_name}.py` (e.g., `benchmark_inference_speed.py`).
- **Classes**: `Benchmark{Component}` (e.g., `BenchmarkQwenAttention`).
- **Functions**: `benchmark_{scenario}` (e.g., `benchmark_large_batch_inference`).
