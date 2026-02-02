"""
Common Components Package for Inference-PIO

This module provides access to all common components in the Inference-PIO system.
"""

from transformers.file_utils import ModelOutput

# Import utility functions from unified utils module
from utils.tensor_utils import (
    apply_chunking_to_forward,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_with_position_ids,
    gelu_new,
    masked_fill_with_broadcast,
    normalize_with_l2,
    pad_sequence_to_length,
    repeat_kv,
    rotate_half,
    safe_tensor_operation,
    silu,
    softmax_with_temperature,
    swish,
    truncate_sequence_to_length,
    validate_tensor_shape,
)
from .base_attention import BaseAttention

# Base components
from .base_model import BaseModel
from .config_factory import ConfigFactory, create_model_config, register_model_config
from .config_loader import ConfigLoader, load_config_from_path, validate_config_data
from .config_validator import (
    ConfigValidationError,
    ConfigValidator,
    get_config_validator,
)

# Hardware analyzer
from .hardware_analyzer import get_system_profile
from .improved_base_plugin_interface import ModelPluginInterface as ModelInterface
from .improved_base_plugin_interface import PluginMetadata as ModelPluginMetadata
from .improved_base_plugin_interface import (
    TextModelPluginInterface,
)

# Configuration management
from .model_config_base import BaseConfig, ConfigurableModelMixin, ModelConfigError
from .model_optimization_mixin import ModelOptimizationMixin
from .model_sharding import ModelShardingMixin
from .model_surgery_component import ModelSurgeryMixin

# Optimization management
from .optimization_manager import OptimizationInterface, get_optimization_manager
from .security_management import SecurityManagementMixin

# Interfaces
from .standard_plugin_interface import (
    PluginMetadata,
    PluginType,
    StandardPluginInterface,
)
from .unified_config_manager import UnifiedConfigManager, get_unified_config_manager
from .secondary_plugin_manager import SecondaryPluginManager, SecondaryPluginType

# All exports
__all__ = [
    # Interfaces
    "StandardPluginInterface",
    "ModelInterface",
    "OptimizationInterface",
    "ModelPluginMetadata",
    "PluginType",
    "TextModelPluginInterface",
    # Base components
    "BaseModel",
    "ModelOutput",
    "BaseAttention",
    # Configuration management
    "BaseConfig",
    "ConfigurableModelMixin",
    "ModelConfigError",
    "ConfigFactory",
    "register_model_config",
    "create_model_config",
    "ConfigValidator",
    "ConfigValidationError",
    "get_config_validator",
    "ConfigLoader",
    "load_config_from_path",
    "validate_config_data",
    "UnifiedConfigManager",
    "get_unified_config_manager",
    "ModelOptimizationMixin",
    "ModelShardingMixin",
    "SecurityManagementMixin",
    "ModelSurgeryMixin",
    # Optimization management
    "get_optimization_manager",
    # Hardware analyzer
    "get_system_profile",
    # Secondary plugin management
    "SecondaryPluginManager",
    "SecondaryPluginType",
]
