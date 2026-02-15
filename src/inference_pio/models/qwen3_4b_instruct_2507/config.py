from typing import Optional, List
from ...common.config.model_config_base import BaseConfig

class Qwen3_4B_Instruct_2507_Config(BaseConfig):
    def __init__(self, **kwargs):
        # Filter kwargs for BaseConfig fields
        valid_keys = BaseConfig.__dataclass_fields__.keys()
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        super().__init__(**filtered)

        self.model_type = "qwen3"
        # 4B Params: 36 Layers, 32 Heads (GQA 32/8)
        # Hidden Size: ~3072
        self.hidden_size = 3072
        self.num_attention_heads = 32
        self.num_key_value_heads = 8 # GQA 32/8
        self.num_hidden_layers = 36
        self.vocab_size = 151936
        self.max_position_embeddings = 131072 # 128K context
        self.tie_word_embeddings = True

        for k, v in kwargs.items():
            setattr(self, k, v)

class Qwen3_4B_Instruct_2507_DynamicConfig(Qwen3_4B_Instruct_2507_Config):
    pass
