"""
Qwen3-0.6B Model Package

This module provides the entry point for the Qwen3-0.6B model plugin.
Each model plugin is completely independent with its own configuration, tests, and benchmarks.
"""

from .config import Qwen3_0_6B_Config, Qwen3_0_6B_DynamicConfig
from .config_integration import (
    Qwen3_0_6BConfigurablePlugin,
)
from .config_integration import (
    create_qwen3_0_6b_plugin as create_qwen3_0_6b_configurable_plugin,
)
from .model import Qwen3_0_6B_Model, create_qwen3_0_6b_model
from .plugin import Qwen3_0_6B_Plugin, create_qwen3_0_6b_plugin

__all__ = [
    "Qwen3_0_6B_Config",
    "Qwen3_0_6B_DynamicConfig",
    "Qwen3_0_6B_Model",
    "create_qwen3_0_6b_model",
    "Qwen3_0_6B_Plugin",
    "create_qwen3_0_6b_plugin",
    "Qwen3_0_6BConfigurablePlugin",
    "create_qwen3_0_6b_configurable_plugin",
]
