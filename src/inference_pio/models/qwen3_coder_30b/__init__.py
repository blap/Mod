"""
Qwen3-Coder-30B Package - Self-Contained Version

This module provides the initialization for the Qwen3-Coder-30B model package
in the self-contained plugin architecture for the Inference-PIO system.
"""

from .plugin import Qwen3_Coder_30B_Plugin, create_qwen3_coder_30b_plugin
from .model import Qwen3Coder30BModel
from .config import Qwen3Coder30BConfig


__all__ = [
    "Qwen3_Coder_30B_Plugin",
    "create_qwen3_coder_30b_plugin",
    "Qwen3Coder30BModel",
    "Qwen3Coder30BConfig"
]