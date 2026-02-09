"""
Native CPU Processor Plugin (C-Engine Backend)

This plugin implements hardware operations using the custom C-based Tensor Engine.
It replaces the legacy PyTorch-based GenericCPUPlugin.
"""

import logging
from typing import Any, Dict, Optional, List

from ...common.interfaces.improved_base_plugin_interface import BasePluginInterface, PluginMetadata, PluginType
from ...core.engine.backend import Tensor, scaled_dot_product_attention

logger = logging.getLogger(__name__)

class NativeCPUPlugin(BasePluginInterface):
    """
    Native CPU implementation using custom C-based tensor operations.
    Zero external dependencies (no Torch, no Numpy).
    """

    def __init__(self):
        metadata = PluginMetadata(
            name="NativeCPU",
            version="2.0.0",
            author="System",
            description="High-performance native C tensor engine for CPU.",
            plugin_type=PluginType.HARDWARE,
            dependencies=[],
            compatibility={"os": ["linux", "windows"], "arch": ["x86_64", "arm64"]},
        )
        super().__init__(metadata)
        self._config = {}
        self._num_threads = 1

    def initialize(self, config: Dict[str, Any] = None) -> bool:
        self._config = config or {}
        # In a real C-engine, thread management would be via OMP_NUM_THREADS env var or C-api
        # For now, we assume environment setup
        import os
        if "num_threads" in self._config:
            os.environ["OMP_NUM_THREADS"] = str(self._config["num_threads"])

        logger.info("NativeCPUPlugin initialized (C-Engine Backend).")
        return True

    def cleanup(self) -> bool:
        return True

    # --- Hardware Operations Interface ---

    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        return a.matmul(b)

    def optimized_matmul(self, a: Tensor, b: Tensor) -> Tensor:
        return self.matmul(a, b)

    def scaled_dot_product_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> Tensor:
        # Native C implementation of SDPA (Fused)
        return scaled_dot_product_attention(query, key, value)

    def apply_activation(self, x: Tensor, activation_type: str) -> Tensor:
        if activation_type == "silu" or activation_type == "swish":
            return x.silu()
        elif activation_type == "gelu":
            return x.gelu()
        return x

    def manage_threads(self, num_threads: int):
        # Set env var for OpenMP
        import os
        os.environ["OMP_NUM_THREADS"] = str(num_threads)

def create_generic_cpu_plugin() -> NativeCPUPlugin:
    """
    Factory that replaces the old GenericCPUPlugin with NativeCPUPlugin.
    Keeps function name for compatibility.
    """
    return NativeCPUPlugin()
