from .base import NativeCPUPlugin
from ...core.engine.backend import Tensor
import logging

class NativeCPUPlugin(NativeCPUPlugin):
    def initialize(self, config: dict = None) -> bool:
        if not super().initialize(config): return False

        # Check for Fused Ops support in lib
        if hasattr(self.lib, 'tensor_fused_add_bias_activation'):
            logging.info("NativeCPU: Fused operations enabled.")

        # Init memory pool if huge pages requested
        if config.get("huge_pages", False):
            if hasattr(self.lib, 'init_memory_pool'):
                # 1GB pool for testing
                self.lib.init_memory_pool(1024*1024*1024)
                logging.info("NativeCPU: Huge Pages Memory Pool initialized (1GB).")
        return True

    def fused_add_bias_activation(self, out: Tensor, a: Tensor, b: Tensor, bias: Tensor, act: str):
        if hasattr(self.lib, 'tensor_fused_add_bias_activation'):
            # Convert act string to C compatible if needed, currently passing str is risky via ctypes without conversion
            # Assuming API takes char*
            b_act = act.encode('utf-8')
            self.lib.tensor_fused_add_bias_activation(out, a, b, bias, b_act)
        else:
            # Fallback
            out = a + b
            if bias: out = out + bias
            if act == "silu": out = out.silu()
            elif act == "gelu": out = out.gelu()
