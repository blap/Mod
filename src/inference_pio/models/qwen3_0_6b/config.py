from typing import Optional, List
from ...common.config.model_config_base import BaseConfig

class Qwen3_0_6B_Config(BaseConfig):
    def __init__(self, **kwargs):
        # Filter kwargs for BaseConfig fields
        valid_keys = BaseConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        super().__init__(**filtered)

        self.model_type = "qwen3"
        self.hidden_size = 1024
        self.num_attention_heads = 16
        self.num_key_value_heads = 8 # GQA 16/8
        self.num_hidden_layers = 28 # Updated from 24 to 28 per specs
        self.vocab_size = 151936
        self.max_position_embeddings = 32768
        self.tie_word_embeddings = True # Enable tying

        # Override with any specific kwargs provided
        for k, v in kwargs.items():
            setattr(self, k, v)

class Qwen3_0_6B_DynamicConfig(Qwen3_0_6B_Config):
    pass
