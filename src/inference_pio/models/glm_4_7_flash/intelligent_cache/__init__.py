"""
Intelligent Cache System for GLM-4.7-Flash Model

This package provides advanced caching mechanisms with predictive and intelligent policies
for the GLM-4.7-Flash model.
"""

from .intelligent_cache_manager import (
    IntelligentCacheConfig,
    IntelligentCacheManager,
    CachePolicy,
    apply_intelligent_caching_to_model,
    create_intelligent_cache_for_glm47
)

__all__ = [
    "IntelligentCacheConfig",
    "IntelligentCacheManager",
    "CachePolicy",
    "apply_intelligent_caching_to_model",
    "create_intelligent_cache_for_glm47"
]