from typing import Optional, List
from ...common.base_config import BaseConfig

class Qwen3_4B_Instruct_2507_Config(BaseConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "qwen3"
        self.hidden_size = 2048
        self.num_attention_heads = 16
        self.num_hidden_layers = 24
        self.vocab_size = 151936

class Qwen3_4B_Instruct_2507_DynamicConfig(Qwen3_4B_Instruct_2507_Config):
    pass
