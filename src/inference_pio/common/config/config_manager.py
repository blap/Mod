"""
Configuration Manager for Model Plugins

This module provides configuration management for different model plugins
in the Inference-PIO system.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .model_config_base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class GLM47DynamicConfig(BaseConfig):
    """
    Dynamic configuration for GLM-4.7-Flash model.

    This configuration class extends BaseConfig with GLM-specific parameters
    and settings for the GLM-4.7-Flash model.
    """

    # Model identification
    model_path: str = "ZhipuAI/glm-4-9b-chat"  # Using 4.9B as closest to 4.7B
    model_name: str = "glm_4_7_flash"

    # GLM-specific parameters
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    num_hidden_layers: int = 32
    intermediate_size: int = 13696
    vocab_size: int = 134528
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0

    # GLM-specific settings
    pre_seq_len: int = 8
    prefix_projection: bool = False

    # Virtual execution settings
    enable_virtual_execution: bool = False
    num_virtual_partitions: int = 2
    memory_per_partition_gb: float = 4.0

    def get_model_specific_params(self) -> Dict[str, Any]:
        """
        Return GLM-4.7-Flash specific parameters.

        Returns:
            A dictionary containing GLM-4.7-Flash specific configuration parameters.
        """
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "pre_seq_len": self.pre_seq_len,
            "prefix_projection": self.prefix_projection,
        }


@dataclass
class Qwen34BInstruct2507DynamicConfig(BaseConfig):
    """
    Dynamic configuration for Qwen3-4B-Instruct-2507 model.

    This configuration class extends BaseConfig with Qwen3-specific parameters
    and settings for the Qwen3-4B-Instruct-2507 model.
    """

    # Model identification
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    model_name: str = "qwen3_4b_instruct_2507"

    # Qwen3-specific parameters
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: int = 8192
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Qwen3-specific settings
    use_sliding_window: bool = True
    sliding_window: int = 4096

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return Qwen3-4B-Instruct-2507 specific parameters."""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "use_sliding_window": self.use_sliding_window,
            "sliding_window": self.sliding_window,
        }


@dataclass
class Qwen3Coder30BDynamicConfig(BaseConfig):
    """
    Dynamic configuration for Qwen3-Coder-30B model.

    This configuration class extends BaseConfig with Qwen3-specific parameters
    and settings for the Qwen3-Coder-30B model.
    """

    # Model identification
    model_path: str = "Qwen/Qwen3-Coder-30B"
    model_name: str = "qwen3_coder_30b"

    # Qwen3-Coder specific parameters
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    num_hidden_layers: int = 64
    intermediate_size: int = 16384
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Coder-specific settings
    use_sliding_window: bool = True
    sliding_window: int = 4096

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return Qwen3-Coder-30B specific parameters."""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "use_sliding_window": self.use_sliding_window,
            "sliding_window": self.sliding_window,
        }


@dataclass
class Qwen3VL2BDynamicConfig(BaseConfig):
    """
    Dynamic configuration for Qwen3-VL-2B model.

    This configuration class extends BaseConfig with Qwen3-VL specific parameters
    and settings for the Qwen3-VL-2B model.
    """

    # Model identification
    model_path: str = "Qwen/Qwen3-VL-2B"
    model_name: str = "qwen3_vl_2b"

    # Qwen3-VL specific parameters
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    num_hidden_layers: int = 24
    intermediate_size: int = 6144
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Vision-specific settings
    vision_config: Dict[str, Any] = field(default_factory=dict)
    visual_dim: int = 1024
    patch_size: int = 14

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return Qwen3-VL-2B specific parameters."""
        params = {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "visual_dim": self.visual_dim,
            "patch_size": self.patch_size,
        }
        if self.vision_config:
            params["vision_config"] = self.vision_config
        return params


@dataclass
class Qwen3_0_6BDynamicConfig(BaseConfig):
    """
    Dynamic configuration for Qwen3-0.6B model.

    This configuration class extends BaseConfig with Qwen3-specific parameters
    and settings for the Qwen3-0.6B model.
    """

    # Model identification
    model_path: str = "Qwen/Qwen3-0.6B"
    model_name: str = "qwen3_0_6b"

    # Qwen3-0.6B specific parameters
    hidden_size: int = 896
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    num_hidden_layers: int = 24
    intermediate_size: int = 4864
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Qwen3-0.6B specific settings
    use_sliding_window: bool = True
    sliding_window: int = 4096

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return Qwen3-0.6B specific parameters."""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "use_sliding_window": self.use_sliding_window,
            "sliding_window": self.sliding_window,
        }


@dataclass
class Qwen34BDynamicConfig(BaseConfig):
    """
    Dynamic configuration for Qwen3-4B-Instruct model.

    This configuration class extends BaseConfig with Qwen3-specific parameters
    and settings for the Qwen3-4B-Instruct model.
    """

    # Model identification
    model_path: str = "Qwen/Qwen3-4B-Instruct-2507"
    model_name: str = "qwen3_4b_instruct_2507"

    # Qwen3-4B specific parameters
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: int = 8192
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Qwen3-4B specific settings
    use_sliding_window: bool = True
    sliding_window: int = 4096

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return Qwen3-4B-Instruct-2507 specific parameters."""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "use_sliding_window": self.use_sliding_window,
            "sliding_window": self.sliding_window,
        }


@dataclass
class Qwen3CoderDynamicConfig(BaseConfig):
    """
    Dynamic configuration for Qwen3-Coder model.

    This configuration class extends BaseConfig with Qwen3-specific parameters
    and settings for the Qwen3-Coder model.
    """

    # Model identification
    model_path: str = "Qwen/Qwen3-Coder-30B"
    model_name: str = "qwen3_coder_30b"

    # Qwen3-Coder specific parameters
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    num_hidden_layers: int = 64
    intermediate_size: int = 16384
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Coder-specific settings
    use_sliding_window: bool = True
    sliding_window: int = 4096

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return Qwen3-Coder-30B specific parameters."""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "use_sliding_window": self.use_sliding_window,
            "sliding_window": self.sliding_window,
        }


@dataclass
class Qwen3VLDynamicConfig(BaseConfig):
    """
    Dynamic configuration for Qwen3-VL model.

    This configuration class extends BaseConfig with Qwen3-VL specific parameters
    and settings for the Qwen3-VL model.
    """

    # Model identification
    model_path: str = "Qwen/Qwen3-VL-2B"
    model_name: str = "qwen3_vl_2b"

    # Qwen3-VL specific parameters
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_key_value_heads: int = 12
    num_hidden_layers: int = 24
    intermediate_size: int = 6144
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    # Vision-specific settings
    visual_dim: int = 1024
    patch_size: int = 14

    def get_model_specific_params(self) -> Dict[str, Any]:
        """Return Qwen3-VL-2B specific parameters."""
        return {
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "visual_dim": self.visual_dim,
            "patch_size": self.patch_size,
        }


def get_dynamic_config_for_model(model_name: str) -> BaseConfig:
    """
    Get the appropriate dynamic configuration for a given model name.

    Args:
        model_name: Name of the model

    Returns:
        Appropriate configuration object for the specified model
    """
    model_name_lower = model_name.lower().replace("-", "_").replace(" ", "_")

    if "glm_4_7" in model_name_lower or "glm_4" in model_name_lower:
        return GLM47DynamicConfig()
    elif "qwen3_4b_instruct" in model_name_lower:
        return Qwen34BInstruct2507DynamicConfig()
    elif "qwen3_4b" in model_name_lower:
        return Qwen34BDynamicConfig()
    elif "qwen3_coder_30b" in model_name_lower:
        return Qwen3Coder30BDynamicConfig()
    elif "qwen3_coder" in model_name_lower or "coder" in model_name_lower:
        return Qwen3CoderDynamicConfig()
    elif "qwen3_vl_2b" in model_name_lower:
        return Qwen3VL2BDynamicConfig()
    elif "qwen3_vl" in model_name_lower:
        return Qwen3VLDynamicConfig()
    elif "qwen3_0_6b" in model_name_lower:
        return Qwen3_0_6BDynamicConfig()
    else:
        # Default to base config if model not recognized
        return BaseConfig()


__all__ = [
    "GLM47DynamicConfig",
    "Qwen34BInstruct2507DynamicConfig",
    "Qwen3Coder30BDynamicConfig",
    "Qwen3VL2BDynamicConfig",
    "Qwen3_0_6BDynamicConfig",
    "Qwen34BDynamicConfig",
    "Qwen3CoderDynamicConfig",
    "Qwen3VLDynamicConfig",
    "get_dynamic_config_for_model",
]
