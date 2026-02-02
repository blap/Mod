"""
Inference-PIO Package - Main Entry Point

This module provides the main entry point for the Inference-PIO system with self-contained plugins.
Each model plugin is completely independent with its own configuration, tests, and benchmarks.
"""

from .common.base_attention import BaseAttention

# Import common components
from .common.base_model import BaseModel

# Import model factory
from .model_factory import create_model, get_model_class, register_model

# Import plugin system components
from .plugins.manager import PluginManager, get_plugin_manager

# Import utility modules
from . import testing_utils
from . import benchmarking_utils

# Define what gets imported with "from inference_pio import *"
__all__ = [
    # Common Components
    "BaseModel",
    "BaseAttention",
    # Plugin System
    "PluginManager",
    "get_plugin_manager",
    # Model Factory
    "create_model",
    "get_model_class",
    "register_model",
    # Utility Modules
    "testing_utils",
    "benchmarking_utils",
]

# Package version
__version__ = "1.0.0"
__author__ = "Inference-PIO Team"
