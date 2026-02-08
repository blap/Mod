"""
Refactored Model Adapter Interface
"""
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    Module = torch.nn.Module
else:
    Module = Any

class ModelAdapterInterface:
    def adapt_model(self, model: Module, config: Any) -> Module:
        raise NotImplementedError
