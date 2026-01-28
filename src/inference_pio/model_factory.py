"""
Model Factory

This module provides a unified factory to create model plugins based on model names.
It abstracts the specific locations and instantiation details of each model plugin.
"""

import logging
from typing import Optional, Dict, Any, Type, Union

from .common.standard_plugin_interface import ModelPluginInterface
from .plugin_system.factory import get_processor_plugin
from .common.hardware_analyzer import SystemProfile, get_system_profile

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating model plugins.
    """

    @staticmethod
    def create_model(model_name: str, config: Optional[Dict[str, Any]] = None) -> ModelPluginInterface:
        """
        Create a model plugin instance based on the model name.

        Args:
            model_name: The name of the model to create.
            config: Optional configuration dictionary.

        Returns:
            An instance of the requested model plugin.

        Raises:
            ValueError: If the model name is not supported.
            ImportError: If the model plugin module cannot be imported.
        """
        model_name = model_name.lower().replace("-", "_").replace(" ", "_")

        logger.info(f"Creating model plugin for: {model_name}")

        try:
            if "qwen3_vl" in model_name or "qwen3_vl_2b" in model_name:
                from .models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
                plugin = create_qwen3_vl_2b_instruct_plugin()

            elif "glm_4_7" in model_name or "glm_4" in model_name:
                from .models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
                plugin = create_glm_4_7_flash_plugin()

            elif "qwen3_4b" in model_name:
                from .models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
                plugin = create_qwen3_4b_instruct_2507_plugin()

            elif "qwen3_coder" in model_name or "coder" in model_name:
                from .models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
                plugin = create_qwen3_coder_30b_plugin()

            else:
                # Try generic creation or fuzzy matching if needed in future
                raise ValueError(f"Unsupported model name: {model_name}. Available models: qwen3-vl-2b, glm-4-7-flash, qwen3-4b, qwen3-coder")

            # Initialize the plugin if config is provided
            if config:
                logger.debug(f"Initializing {model_name} with provided config")
                plugin.initialize(**config)

            return plugin

        except ImportError as e:
            logger.error(f"Failed to import model plugin for {model_name}: {e}")
            raise ImportError(f"Model plugin for {model_name} could not be loaded. Ensure dependencies are installed.") from e
        except Exception as e:
            logger.error(f"Error creating model plugin for {model_name}: {e}")
            raise e

    @staticmethod
    def list_supported_models() -> list[str]:
        """
        Returns a list of supported model names.
        """
        return [
            "qwen3-vl-2b",
            "glm-4-7-flash",
            "qwen3-4b",
            "qwen3-coder-30b"
        ]
