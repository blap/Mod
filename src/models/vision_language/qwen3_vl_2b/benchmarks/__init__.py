"""
Qwen3-VL-2B Benchmarks Package - Self-Contained Version

This module provides the initialization for the Qwen3-VL-2B model benchmarks package
for the Inference-PIO system. This package includes all benchmark categories for the
Qwen3-VL-2B model with specific optimizations and evaluation metrics.
"""

from .integration.benchmark_async_multimodal_processing import (
    BenchmarkAsyncMultimodalProcessing,
)
from .integration.benchmark_comparison import BenchmarkQwen3VL2BComparison
from .integration.benchmark_intelligent_multimodal_caching import (
    BenchmarkQwen3VL2BIntelligentCaching,
)

# Import from subdirectories and expose at package level
from .performance.benchmark_inference_speed import BenchmarkQwen3VL2BInferenceSpeed
from .performance.benchmark_inference_speed_comparison import (
    Qwen3VL2BInferenceSpeedComparisonBenchmark,
)
from .performance.benchmark_memory_usage import BenchmarkQwen3VL2BMemoryUsage
from .performance.benchmark_optimization_impact import (
    BenchmarkQwen3VL2BOptimizationImpact,
)
from .performance.benchmark_power_efficiency import BenchmarkQwen3VL2BPowerEfficiency
from .performance.benchmark_throughput import BenchmarkQwen3VL2BThroughput

__all__ = [
    "BenchmarkQwen3VL2BComparison",
    "BenchmarkQwen3VL2BInferenceSpeed",
    "Qwen3VL2BInferenceSpeedComparisonBenchmark",
    "BenchmarkQwen3VL2BMemoryUsage",
    "BenchmarkQwen3VL2BOptimizationImpact",
    "BenchmarkQwen3VL2BPowerEfficiency",
    "BenchmarkQwen3VL2BThroughput",
    "BenchmarkQwen3VL2BIntelligentCaching",
    "BenchmarkAsyncMultimodalProcessing",
]
