"""
Plugin System Package for Inference-PIO

This module provides the plugin system package for the Inference-PIO system.
"""

from .plugin_manager import (
    PluginManager,
    get_plugin_manager,
    register_plugin,
    load_plugin_from_path,
    load_plugins_from_directory,
    activate_plugin,
    execute_plugin
)

__all__ = [
    "PluginManager",
    "get_plugin_manager",
    "register_plugin",
    "load_plugin_from_path",
    "load_plugins_from_directory",
    "activate_plugin",
    "execute_plugin"
]