"""
Core benchmarking utilities and functions for the Inference-PIO project.

This module provides essential benchmarking capabilities including:
- Performance measurement tools
- Optimization impact assessment
- Resource utilization tracking
- Cross-model comparison utilities

The core module contains standardized benchmarking functions that can be
used across different models and optimization strategies.
"""

# Import core benchmarking functionality
from .benchmark_discovery import BenchmarkDiscovery, discover_and_run_all_benchmarks

__all__ = ["BenchmarkDiscovery", "discover_and_run_all_benchmarks"]
