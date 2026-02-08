from dataclasses import dataclass
from typing import Optional, List, Union

@dataclass
class PretrainedConfig:
    """
    Base configuration class for models.
    """
    model_type: str = ""
    architectures: List[str] = None
    torch_dtype: str = "float32" # Changed to string
    transformers_version: str = "4.30.0" # Mock version if needed

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
