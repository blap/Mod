"""
Inference-PIO Main Entry Point

This module provides the main entry point for the Inference-PIO system.
"""

import os
import sys

# Add the src directory to the path to allow imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from .common.base_attention import BaseAttention

# Import the main components
from .common.base_model import BaseModel
from .model_factory import ModelFactory, create_model

# Import all model plugins
from .models.glm_4_7_flash.plugin import (
    GLM_4_7_Flash_Plugin,
    create_glm_4_7_flash_plugin,
)
from .models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin, create_qwen3_0_6b_plugin
from .models.qwen3_4b_instruct_2507.plugin import (
    Qwen3_4B_Instruct_2507_Plugin,
    create_qwen3_4b_instruct_2507_plugin,
)
from .models.qwen3_coder_30b.plugin import (
    Qwen3_Coder_30B_Plugin,
    create_qwen3_coder_30b_plugin,
)
from .models.qwen3_vl_2b.plugin import (
    Qwen3_VL_2B_Instruct_Plugin,
    create_qwen3_vl_2b_instruct_plugin,
)
from .plugins.manager import PluginManager, get_plugin_manager

__version__ = "1.0.0"
__author__ = "Inference-PIO Team"


def main():
    """Main entry point for the Inference-PIO system."""
    print("Inference-PIO System Initialized")
    print(f"Version: {__version__}")

    # Example usage
    print("\nAvailable models:")
    for model in ModelFactory.list_supported_models():
        print(f"- {model}")

    print("\nExample usage:")
    print("from inference_pio import create_model")
    print("model = create_model('qwen3-0.6b')")
    print("result = model.generate_text('Hello, world!')")


if __name__ == "__main__":
    main()
