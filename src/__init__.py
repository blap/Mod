"""
Inference-PIO Main Package - Self-Contained Plugin Architecture

This package provides the main entry point for the Inference-PIO system with self-contained plugins
for various models including GLM-4.7, Qwen3-Coder-30B, Qwen3-VL-2B, and Qwen3-4B-Instruct-2507.
"""

# Import the main inference_pio package functionality
from . import inference_pio

# Import all model plugins from inference_pio
from .inference_pio import (
    GLM_4_7_Flash_Plugin,
    create_glm_4_7_flash_plugin,
    Qwen3_Coder_30B_Plugin,
    create_qwen3_coder_30b_plugin,
    Qwen3_VL_2B_Instruct_Plugin,
    create_qwen3_vl_2b_instruct_plugin,
    Qwen3_4B_Instruct_2507_Plugin,
    create_qwen3_4b_instruct_2507_plugin,
    PluginManager,
    get_plugin_manager,
    register_plugin,
    load_plugin_from_path,
    load_plugins_from_directory,
    activate_plugin,
    execute_plugin
)

# Import common components
from .inference_pio.common.base_plugin_interface import (
    ModelPluginInterface,
    TextModelPluginInterface,
    ModelPluginMetadata,
    PluginType
)

# Import configuration manager if it exists
try:
    from .inference_pio.common.config_manager import ConfigManager
except ImportError:
    # Define ConfigManager as a placeholder if not available
    class ConfigManager:
        """Placeholder ConfigManager class."""

        def __init__(self):
            pass


__version__ = "1.0.0"
__author__ = "Inference-PIO Team"

# Export all public components
__all__ = [
    # Model plugins
    "GLM_4_7_Plugin",
    "create_glm_4_7_flash_plugin",
    "Qwen3_Coder_30B_Plugin",
    "create_qwen3_coder_30b_plugin",
    "Qwen3_VL_2B_Instruct_Plugin",
    "create_qwen3_vl_2b_instruct_plugin",
    "Qwen3_4B_Instruct_2507_Plugin",
    "create_qwen3_4b_instruct_2507_plugin",

    # Plugin system
    "PluginManager",
    "get_plugin_manager",
    "register_plugin",
    "load_plugin_from_path",
    "load_plugins_from_directory",
    "activate_plugin",
    "execute_plugin",

    # Common interfaces
    "ModelPluginInterface",
    "TextModelPluginInterface",
    "ModelPluginMetadata",
    "PluginType",
    "ConfigManager",
]
