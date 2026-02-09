"""
Qwen3-Coder-30B Configuration - Self-Contained Version
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ...common.config.model_config_base import BaseConfig, get_default_model_path

@dataclass
class Qwen3Coder30BConfig(BaseConfig):
    model_path: str = ""
    model_name: str = "qwen3_coder_30b"

    hidden_size: int = 7680
    num_attention_heads: int = 60
    num_hidden_layers: int = 60
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0
    intermediate_size: int = 20480
    vocab_size: int = 152064
    layer_norm_eps: float = 1e-06

    def __post_init__(self):
        if not self.model_path:
            self.model_path = get_default_model_path(self.model_name)
        # BaseConfig does not implement __post_init__, so we don't call super() if unnecessary,
        # or we ensure BaseConfig has it.
        # Checking BaseConfig implementation: It's a dataclass.
        pass

__all__ = ["Qwen3Coder30BConfig"]
