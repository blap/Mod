"""
Inference-PIO Plugin System - Main Package

This is the main package for the Inference-PIO system with self-contained plugins.
"""

import os
import sys

# Add the src directory to the path to allow imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import common components using absolute imports from src level
from common.base_attention import BaseAttention
from common.base_model import BaseModel

# Import model factory
from model_factory import create_model, get_model_class, register_model

# Import configurable model plugins
from models.specialized.glm_4_7_flash.config_integration import (
    GLM47ConfigurablePlugin,
)
from models.specialized.glm_4_7_flash.config_integration import (
    create_glm_4_7_flash_plugin as create_glm_4_7_configurable_plugin,
)

# Import model plugins
from models.specialized.glm_4_7_flash.plugin import (
    GLM_4_7_Flash_Plugin,
    create_glm_4_7_flash_plugin,
)
from models.language.qwen3_0_6b.config_integration import (
    Qwen3_0_6BConfigurablePlugin,
)
from models.language.qwen3_0_6b.config_integration import (
    create_qwen3_0_6b_plugin as create_qwen3_0_6b_configurable_plugin,
)
from models.language.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin, create_qwen3_0_6b_plugin
from models.language.qwen3_4b_instruct_2507.config_integration import (
    Qwen34BInstruct2507ConfigurablePlugin,
)
from models.language.qwen3_4b_instruct_2507.config_integration import (
    create_qwen3_4b_instruct_2507_plugin as create_qwen3_4b_instruct_2507_configurable_plugin,
)
from models.language.qwen3_4b_instruct_2507.plugin import (
    Qwen3_4B_Instruct_2507_Plugin,
    create_qwen3_4b_instruct_2507_plugin,
)
from models.coding.qwen3_coder_30b.config_integration import (
    Qwen3Coder30BConfigurablePlugin,
)
from models.coding.qwen3_coder_30b.config_integration import (
    create_qwen3_coder_30b_plugin as create_qwen3_coder_30b_configurable_plugin,
)
from models.coding.qwen3_coder_30b.plugin import (
    Qwen3_Coder_30B_Plugin,
    create_qwen3_coder_30b_plugin,
)
from models.vision_language.qwen3_vl_2b.config_integration import (
    Qwen3VL2BConfigurablePlugin,
)
from models.vision_language.qwen3_vl_2b.config_integration import (
    create_qwen3_vl_2b_instruct_plugin as create_qwen3_vl_2b_configurable_plugin,
)
from models.vision_language.qwen3_vl_2b.plugin import (
    Qwen3_VL_2B_Instruct_Plugin,
    create_qwen3_vl_2b_instruct_plugin,
)

# Import plugin system components
from plugins.manager import PluginManager, get_plugin_manager

# Define what gets imported with "from inference_pio import *"
__all__ = [
    # Common Components
    "BaseModel",
    "BaseAttention",
    # Plugin System
    "PluginManager",
    "get_plugin_manager",
    # Model Plugins
    "GLM_4_7_Flash_Plugin",
    "create_glm_4_7_flash_plugin",
    "Qwen3_4B_Instruct_2507_Plugin",
    "create_qwen3_4b_instruct_2507_plugin",
    "Qwen3_Coder_30B_Plugin",
    "create_qwen3_coder_30b_plugin",
    "Qwen3_VL_2B_Instruct_Plugin",
    "create_qwen3_vl_2b_instruct_plugin",
    "Qwen3_0_6B_Plugin",
    "create_qwen3_0_6b_plugin",
    # Configurable Model Plugins
    "GLM47ConfigurablePlugin",
    "create_glm_4_7_configurable_plugin",
    "Qwen34BInstruct2507ConfigurablePlugin",
    "create_qwen3_4b_instruct_2507_configurable_plugin",
    "Qwen3Coder30BConfigurablePlugin",
    "create_qwen3_coder_30b_configurable_plugin",
    "Qwen3VL2BConfigurablePlugin",
    "create_qwen3_vl_2b_configurable_plugin",
    "Qwen3_0_6BConfigurablePlugin",
    "create_qwen3_0_6b_configurable_plugin",
    # Model Factory
    "create_model",
    "get_model_class",
    "register_model",
]

# Package version
__version__ = "1.0.0"
__author__ = "Inference-PIO Team"
