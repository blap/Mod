"""
Configuration package for Qwen3-VL model.

This module exports all the necessary configuration classes for the Qwen3-VL model.
The configuration is unified in a single class to eliminate dependencies on multiple
configuration modules.
"""
from .config import Qwen3VLConfig, get_default_config, get_hardware_optimized_config


__all__ = [
    "Qwen3VLConfig",
    "get_default_config",
    "get_hardware_optimized_config"
]