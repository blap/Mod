"""
Qwen3-4B-Instruct-2507 Intelligent Cache System - Init Module

This module initializes the intelligent cache system for the Qwen3-4B-Instruct-2507 model.
"""

from .intelligent_cache_manager import (
    CachePolicy,
    IntelligentCacheConfig,
    IntelligentCacheManager,
    AccessPatternPredictor,
    PerformanceMonitor,
    apply_intelligent_caching_to_model,
    create_intelligent_cache_for_qwen3_4b
)

__all__ = [
    "CachePolicy",
    "IntelligentCacheConfig",
    "IntelligentCacheManager",
    "AccessPatternPredictor",
    "PerformanceMonitor",
    "apply_intelligent_caching_to_model",
    "create_intelligent_cache_for_qwen3_4b",
]