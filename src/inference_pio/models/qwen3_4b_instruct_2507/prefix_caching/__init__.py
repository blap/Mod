"""
Qwen3-4B-Instruct-2507 Prefix Caching Module

This module provides prefix caching functionality for the Qwen3-4B-Instruct-2507 model.
"""

from .prefix_cache_manager import apply_prefix_cache_to_model

__all__ = ["apply_prefix_cache_to_model"]
