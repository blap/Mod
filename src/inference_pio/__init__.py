"""
Inference-PIO Package - Main Entry Point

This module provides the main entry point for the Inference-PIO system with self-contained plugins.
"""

# Import common components
from .common import (
    AdaptiveBatchManager,
    BatchMetrics,
    BatchSizeAdjustmentReason,
    get_adaptive_batch_manager,
    StructuredPruningSystem,
    PruningMethod,
    PruningResult,
    get_structured_pruning_system,
    apply_structured_pruning
)

# Import plugin system components
from .plugin_system.plugin_manager import (
    PluginManager,
    get_plugin_manager,
    register_plugin,
    load_plugin_from_path,
    load_plugins_from_directory,
    activate_plugin,
    execute_plugin,
    discover_and_load_plugins
)

# Import model plugins
from .models import (
    GLM_4_7_Flash_Plugin,
    create_glm_4_7_flash_plugin,
    Qwen3_Coder_30B_Plugin,
    create_qwen3_coder_30b_plugin,
    Qwen3_VL_2B_Instruct_Plugin,
    create_qwen3_vl_2b_instruct_plugin,
    Qwen3_4B_Instruct_2507_Plugin,
    create_qwen3_4b_instruct_2507_plugin
)

# Import configurable model plugins
from .models.glm_4_7_flash.config_integration import (
    GLM47ConfigurablePlugin,
    create_glm_4_7_flash_plugin as create_glm_4_7_configurable_plugin
)
from .models.qwen3_4b_instruct_2507.config_integration import (
    Qwen34BInstruct2507ConfigurablePlugin,
    create_qwen3_4b_instruct_2507_plugin as create_qwen3_4b_instruct_2507_configurable_plugin
)
from .models.qwen3_coder_30b.config_integration import (
    Qwen3Coder30BConfigurablePlugin,
    create_qwen3_coder_30b_plugin as create_qwen3_coder_30b_configurable_plugin
)
from .models.qwen3_vl_2b.config_integration import (
    Qwen3VL2BConfigurablePlugin,
    create_qwen3_vl_2b_instruct_plugin as create_qwen3_vl_2b_configurable_plugin
)

# Import design patterns for easy access
from .design_patterns.integration import (
    create_optimized_plugin,
    create_adapted_model,
    create_optimized_adapted_plugin,
    create_glm_4_7_optimized_plugin,
    create_qwen3_4b_instruct_2507_optimized_plugin,
    create_qwen3_coder_30b_optimized_plugin,
    create_qwen3_vl_2b_optimized_plugin
)

# Define what gets imported with "from inference_pio import *"
__all__ = [
    # Common Components
    "AdaptiveBatchManager",
    "BatchMetrics",
    "BatchSizeAdjustmentReason",
    "get_adaptive_batch_manager",
    "StructuredPruningSystem",
    "PruningMethod",
    "PruningResult",
    "get_structured_pruning_system",
    "apply_structured_pruning",

    # Plugin System
    "PluginManager",
    "get_plugin_manager",
    "register_plugin",
    "load_plugin_from_path",
    "load_plugins_from_directory",
    "activate_plugin",
    "execute_plugin",
    "discover_and_load_plugins",

    # Model Plugins
    "GLM_4_7_Flash_Plugin",
    "create_glm_4_7_flash_plugin",
    "Qwen3_Coder_30B_Plugin",
    "create_qwen3_coder_30b_plugin",
    "Qwen3_VL_2B_Instruct_Plugin",
    "create_qwen3_vl_2b_instruct_plugin",
    "Qwen3_4B_Instruct_2507_Plugin",
    "create_qwen3_4b_instruct_2507_plugin",

    # Configurable Model Plugins
    "GLM47ConfigurablePlugin",
    "create_glm_4_7_configurable_plugin",
    "Qwen34BInstruct2507ConfigurablePlugin",
    "create_qwen3_4b_instruct_2507_configurable_plugin",
    "Qwen3Coder30BConfigurablePlugin",
    "create_qwen3_coder_30b_configurable_plugin",
    "Qwen3VL2BConfigurablePlugin",
    "create_qwen3_vl_2b_configurable_plugin",

    # Design Pattern Functions
    "create_optimized_plugin",
    "create_adapted_model",
    "create_optimized_adapted_plugin",
    "create_glm_4_7_optimized_plugin",
    "create_qwen3_4b_instruct_2507_optimized_plugin",
    "create_qwen3_coder_30b_optimized_plugin",
    "create_qwen3_vl_2b_optimized_plugin"
]

# Package version
__version__ = "1.0.0"
__author__ = "Inference-PIO Team"