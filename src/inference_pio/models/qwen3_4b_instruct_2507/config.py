"""
Qwen3-4B-Instruct-2507 Configuration
"""
from dataclasses import dataclass, field
from typing import Optional, List, Union

@dataclass
class Qwen3_4B_Instruct_2507_Config:
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    vocab_size: int = 151936
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    rope_theta: float = 10000.0
    num_key_value_heads: int = 32
    use_cache: bool = True

    # Defaults
    model_type: str = "qwen2"
    architectures: List[str] = field(default_factory=lambda: ["Qwen2ForCausalLM"])
    torch_dtype: str = "float16" # Just a string now

class Qwen3_4B_Instruct_2507_DynamicConfig(Qwen3_4B_Instruct_2507_Config):
    pass