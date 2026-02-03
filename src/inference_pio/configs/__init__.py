"""
Configs Package for Inference-PIO

This module provides access to all configuration components in the Inference-PIO system.
"""

from .global_config import GlobalConfig, get_global_config, global_config

__all__ = ["GlobalConfig", "get_global_config", "global_config"]
