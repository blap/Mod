"""
GLM-4.7-Flash Package - Self-Contained Version

This module provides the initialization for the GLM-4.7-Flash model package
in the self-contained plugin architecture for the Inference-PIO system.
Each model plugin is completely independent with its own configuration, tests, and benchmarks.
"""

from .config import GLM47DynamicConfig, GLM47FlashConfig
from .config_integration import (
    GLM47ConfigurablePlugin,
)
from .config_integration import (
    create_glm_4_7_flash_plugin as create_glm_4_7_configurable_plugin,
)
from .model import GLM47FlashModel
from .plugin import GLM_4_7_Flash_Plugin, create_glm_4_7_flash_plugin

__all__ = [
    "GLM_4_7_Flash_Plugin",
    "create_glm_4_7_flash_plugin",
    "GLM47FlashModel",
    "GLM47FlashConfig",
    "GLM47DynamicConfig",
    "GLM47ConfigurablePlugin",
    "create_glm_4_7_configurable_plugin",
]
