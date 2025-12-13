"""
System components for Qwen3-VL model.

This package provides system-level components including
dependency injection, interfaces, and optimization utilities.
"""
from .di_container import DIContainer, create_default_container
from .interfaces import ConfigurableComponent, MemoryManager, Optimizer, Preprocessor, Pipeline, AttentionMechanism, MLP, Layer
from .pipeline import IntelOptimizedPipeline
from .metrics_collector import MetricsCollector
from .hardware_detection_fallbacks import HardwareDetector
from .system_level_optimizations import SystemLevelOptimizer

__all__ = [
    "DIContainer",
    "create_default_container",
    "ConfigurableComponent",
    "MemoryManager",
    "Optimizer",
    "Preprocessor",
    "Pipeline",
    "AttentionMechanism",
    "MLP",
    "Layer",
    "IntelOptimizedPipeline",
    "MetricsCollector",
    "HardwareDetector",
    "SystemLevelOptimizer"
]