"""
Qwen3-Coder-30B Package - Self-Contained Version

This module provides the initialization for the Qwen3-Coder-30B model package
in the self-contained plugin architecture for the Inference-PIO system.
Each model plugin is completely independent with its own configuration, tests, and benchmarks.
"""

from .config import Qwen3Coder30BConfig, Qwen3CoderDynamicConfig
from .config_integration import (
    Qwen3Coder30BConfigurablePlugin,
)
from .config_integration import (
    create_qwen3_coder_30b_plugin as create_qwen3_coder_30b_configurable_plugin,
)
from .model import Qwen3Coder30BModel
from .plugin import Qwen3_Coder_30B_Plugin, create_qwen3_coder_30b_plugin

__all__ = [
    "Qwen3_Coder_30B_Plugin",
    "create_qwen3_coder_30b_plugin",
    "Qwen3Coder30BModel",
    "Qwen3Coder30BConfig",
    "Qwen3CoderDynamicConfig",
    "Qwen3Coder30BConfigurablePlugin",
    "create_qwen3_coder_30b_configurable_plugin",
]
