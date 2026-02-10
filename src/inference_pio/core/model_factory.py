"""
Model Factory

This module provides a unified factory to create model plugins based on model names.
It abstracts the specific locations and instantiation details of each model plugin.
Each model plugin is completely independent with its own configuration, tests,
and benchmarks.
"""

import logging
from typing import Any, Dict, Optional, Type

from inference_pio.common.hardware.hardware_analyzer import (
    SystemProfile,
    get_system_profile
)
from inference_pio.common.interfaces.improved_base_plugin_interface import (
    ModelPluginInterface
)
from inference_pio.plugins.manager import get_plugin_manager
from inference_pio.core.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating model plugins.
    """

    @staticmethod
    def create_model(
        model_name: str, config: Optional[Dict[str, Any]] = None
    ) -> ModelPluginInterface:
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

        # Define mapping of model names to HF Hub IDs for download logic
        hf_map = {
            "qwen3_0_6b": "Qwen/Qwen1.5-0.5B",  # Fallback/Simulated
            "qwen3_vl_2b": "Qwen/Qwen-VL-Chat",
            "glm_4_7_flash": "THUDM/glm-4-9b-chat",  # Approximate
            "qwen3_coder_30b": "Qwen/Qwen2.5-Coder-32B-Instruct",
            "qwen3_coder_next": "Qwen/Qwen3-Coder-Next"
        }

        # Determine HF ID
        hf_id = None
        for key, val in hf_map.items():
            if key in model_name:
                hf_id = val
                break

        # Resolve model path using the H: priority logic
        resolved_path = ModelLoader.resolve_model_path(
            model_name, hf_repo_id=hf_id
        )

        # Inject resolved path into config
        if config is None:
            config = {}

        # We only override if it looks like a path or if we want to force the HF ID
        if resolved_path:
            config['model_path'] = resolved_path

        try:
            # First, check if the model is registered
            if model_name in _REGISTERED_MODELS:
                model_class = _REGISTERED_MODELS[model_name]
                plugin = model_class()
            elif "qwen3_0_6b" in model_name or "qwen3_0.6b" in model_name:
                from inference_pio.models.qwen3_0_6b.plugin import (
                    create_qwen3_0_6b_plugin
                )

                plugin = create_qwen3_0_6b_plugin()

            elif "qwen3_vl" in model_name or "qwen3_vl_2b" in model_name:
                from inference_pio.models.qwen3_vl_2b.plugin import (
                    create_qwen3_vl_2b_instruct_plugin,
                )

                plugin = create_qwen3_vl_2b_instruct_plugin()

            elif "glm_4_7" in model_name or "glm_4" in model_name:
                from inference_pio.models.glm_4_7_flash.plugin import (
                    create_glm_4_7_flash_plugin
                )

                plugin = create_glm_4_7_flash_plugin()

            elif "qwen3_coder_next" in model_name:
                from inference_pio.models.qwen3_coder_next.plugin import (
                    create_qwen3_coder_next_plugin
                )

                plugin = create_qwen3_coder_next_plugin()

            elif "qwen3_coder" in model_name:
                from inference_pio.models.qwen3_coder_30b.plugin import (
                    create_qwen3_coder_30b_plugin
                )

                plugin = create_qwen3_coder_30b_plugin()

            else:
                # Try generic creation or fuzzy matching if needed in future
                raise ValueError(
                    f"Unsupported model name: {model_name}. "
                    "Available models: qwen3-0.6b, qwen3-vl-2b, glm-4-7-flash, "
                    "qwen3-coder-30b, qwen3-coder-next"
                )

            # Initialize the plugin if config is provided
            if config:
                logger.debug(
                    f"Initializing {model_name} with provided config"
                )
                plugin.initialize(**config)

            return plugin

        except ImportError as e:
            logger.error(
                f"Failed to import model plugin for {model_name}: {e}"
            )
            raise ImportError(
                f"Model plugin for {model_name} could not be loaded. "
                "Ensure dependencies are installed."
            ) from e
        except Exception as e:
            logger.error(f"Error creating model plugin for {model_name}: {e}")
            raise e

    @staticmethod
    def list_supported_models() -> list[str]:
        """
        Returns a list of supported model names.
        """
        base_models = [
            "qwen3-0.6b",
            "qwen3-vl-2b",
            "glm-4-7-flash",
            "qwen3-coder-30b",
            "qwen3-coder-next",
        ]
        # Add any registered models
        registered_models = list(_REGISTERED_MODELS.keys())
        return base_models + registered_models


def create_model(
    model_name: str, config: Optional[Dict[str, Any]] = None
) -> ModelPluginInterface:
    """
    Create a model plugin instance based on the model name.

    Args:
        model_name: The name of the model to create.
        config: Optional configuration dictionary.

    Returns:
        An instance of the requested model plugin.
    """
    return ModelFactory.create_model(model_name, config)


def get_model_class(model_name: str) -> Type[ModelPluginInterface]:
    """
    Get the model plugin class without instantiating it.

    Args:
        model_name: The name of the model class to get.

    Returns:
        The class of the requested model plugin.
    """
    model_name = model_name.lower().replace("-", "_").replace(" ", "_")

    if "qwen3_0_6b" in model_name or "qwen3_0.6b" in model_name:
        from inference_pio.models.qwen3_0_6b.plugin import Qwen3_0_6bPlugin
        return Qwen3_0_6bPlugin

    elif "qwen3_vl" in model_name or "qwen3_vl_2b" in model_name:
        from inference_pio.models.qwen3_vl_2b.plugin import Qwen3Vl2bInstructPlugin
        return Qwen3Vl2bInstructPlugin

    elif "glm_4_7" in model_name or "glm_4" in model_name:
        from inference_pio.models.glm_4_7_flash.plugin import Glm47FlashPlugin
        return Glm47FlashPlugin

    elif "qwen3_coder_next" in model_name:
        from inference_pio.models.qwen3_coder_next.plugin import Qwen3CoderNextPlugin
        return Qwen3CoderNextPlugin

    elif "qwen3_coder" in model_name:
        from inference_pio.models.qwen3_coder_30b.plugin import Qwen3Coder30bPlugin
        return Qwen3Coder30bPlugin

    else:
        # Return a generic plugin class if model name not recognized
        from inference_pio.common.interfaces.improved_base_plugin_interface import ModelPluginInterface
        raise ValueError(
            f"Unsupported model name: {model_name}. "
            "Available models: qwen3-0.6b, qwen3-vl-2b, glm-4-7-flash, "
            "qwen3-coder-30b, qwen3-coder-next"
        )


# Dictionary to store registered model classes
_REGISTERED_MODELS = {}

def register_model(model_name: str, model_class: Type[ModelPluginInterface]):
    """
    Register a new model plugin class with the factory.

    Args:
        model_name: The name to register the model under.
        model_class: The model plugin class to register.

    Returns:
        True if registration was successful, False otherwise
    """
    try:
        _REGISTERED_MODELS[model_name.lower()] = model_class
        logger.info(f"Successfully registered model: {model_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to register model {model_name}: {e}")
        return False


__all__ = [
    "ModelFactory",
    "create_model",
    "get_model_class",
    "register_model",
    "SystemProfile",
    "get_system_profile",
    "get_plugin_manager"
]
