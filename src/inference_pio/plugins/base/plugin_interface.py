"""
Hardware-Specific Processor Plugin Interface
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.engine.backend import Tensor
else:
    Tensor = Any

class HardwareProcessorPluginInterface(ABC):
    @property
    @abstractmethod
    def plugin_name(self) -> str: raise NotImplementedError
    @property
    @abstractmethod
    def supported_hardware_architectures(self) -> list[str]: raise NotImplementedError
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool: raise NotImplementedError
    @abstractmethod
    def optimized_matmul(self, a: Tensor, b: Tensor) -> Tensor: raise NotImplementedError
    @abstractmethod
    def optimized_scaled_dot_product_attention(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0, is_causal: bool = False) -> Tensor: raise NotImplementedError
    @abstractmethod
    def optimized_apply_activation(self, x: Tensor, activation_type: str) -> Tensor: raise NotImplementedError
    @abstractmethod
    def configure_thread_management(self, num_threads: int): raise NotImplementedError

__all__ = ["HardwareProcessorPluginInterface"]
