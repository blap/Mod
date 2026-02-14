from .base import NativeCPUPlugin as BaseNativeCPUPlugin
from ...core.engine.backend import Tensor
import logging
import ctypes

logger = logging.getLogger(__name__)

class NativeCPUPlugin(BaseNativeCPUPlugin):
    """
    Concrete implementation of Native CPU Plugin.
    Exposes high-level operations like Fused Ops and Thread Management.
    """
    def initialize(self, config: dict = None) -> bool:
        if not super().initialize(config): return False

        # Check for Fused Ops support in lib
        if hasattr(self.lib, 'tensor_fused_add_bias_activation'):
            logger.info("NativeCPU: Fused operations enabled.")

        # Init memory pool if huge pages requested
        if config and config.get("huge_pages", False):
            if hasattr(self.lib, 'init_memory_pool'):
                # 1GB pool for testing
                self.lib.init_memory_pool(ctypes.c_size_t(1024*1024*1024))
                logger.info("NativeCPU: Huge Pages Memory Pool initialized (1GB).")
        return True

    def fused_add_bias_activation(self, out: Tensor, a: Tensor, b: Tensor, bias: Tensor, act: str):
        if hasattr(self.lib, 'tensor_fused_add_bias_activation'):
            b_act = act.encode('utf-8')
            self.lib.tensor_fused_add_bias_activation(out, a, b, bias, b_act)
        else:
            # Fallback (Slow Python Path, should rely on individual C ops if available)
            pass
