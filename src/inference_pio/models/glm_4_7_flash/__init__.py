"""
GLM-4.7-Flash Package - Self-Contained Version

This module provides the initialization for the GLM-4.7-Flash model package
in the self-contained plugin architecture for the Inference-PIO system.
"""

from .plugin import GLM_4_7_Flash_Plugin, create_glm_4_7_flash_plugin
from .model import GLM47FlashModel
from .config import GLM47FlashConfig


__all__ = [
    "GLM_4_7_Flash_Plugin",
    "create_glm_4_7_flash_plugin",
    "GLM47FlashModel",
    "GLM47FlashConfig"
]