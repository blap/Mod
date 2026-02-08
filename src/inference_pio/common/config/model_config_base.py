from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Any
import json
import os

class ModelConfigError(Exception):
    pass

@dataclass
class PretrainedConfig:
    """
    Base configuration class for models (Dependency-Free).
    """
    model_type: str = ""
    architectures: List[str] = field(default_factory=list)
    torch_dtype: str = "float32"
    transformers_version: str = "4.30.0"

    # Common attributes
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> 'PretrainedConfig':
        # Load from JSON file
        config_file = os.path.join(pretrained_model_name_or_path, "config.json")
        if not os.path.exists(config_file):
            # Fallback or create default
            return cls(**kwargs)

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        # Merge kwargs
        config_dict.update(kwargs)

        # Filter valid keys for this class instance
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_dict)

# Alias for compatibility if needed elsewhere
BaseConfig = PretrainedConfig

def get_default_model_path(model_name: str) -> str:
    return os.path.join("models", model_name)

def get_optimal_config_for_hardware() -> Dict[str, Any]:
    return {"device": "cpu"} # Simplification
