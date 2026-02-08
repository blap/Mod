"""
Hardware Processor Plugin Interface
"""
from typing import Any, Dict, List, Optional, Union
from .improved_base_plugin_interface import BasePluginInterface, PluginMetadata

class HardwareProcessorPluginInterface(BasePluginInterface):
    """
    Interface for hardware processor plugins.
    """
    def matmul(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def apply_activation(self, x: Any, activation_type: str) -> Any:
        raise NotImplementedError
