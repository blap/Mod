"""
Plugins Package for Inference-PIO

This module provides access to all plugin components in the Inference-PIO system.
"""

from ..common.standard_plugin_interface import StandardPluginInterface
from .cpu.cpu_plugin import GenericCPUPlugin
from .intel.intel_comet_lake_plugin import IntelCometLakePlugin
from .manager import (
    PluginManager,
    activate_plugin,
    discover_and_load_plugins,
    execute_plugin,
    get_plugin_manager,
    load_plugin_from_path,
    load_plugins_from_directory,
    register_plugin,
)

__all__ = [
    "PluginManager",
    "get_plugin_manager",
    "register_plugin",
    "load_plugin_from_path",
    "load_plugins_from_directory",
    "activate_plugin",
    "execute_plugin",
    "discover_and_load_plugins",
    "StandardPluginInterface",
    "GenericCPUPlugin",
    "IntelCometLakePlugin",
]
