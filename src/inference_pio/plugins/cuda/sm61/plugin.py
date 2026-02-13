from ..base import CUDABasePlugin
from ...common.utils.lib_loader import load_backend_lib
import logging

logger = logging.getLogger(__name__)

class CUDASM61Plugin(CUDABasePlugin):
    """
    Optimized CUDA Plugin for Pascal (GTX 10-series, SM 6.1).
    Attempts to load specialized binary if available.
    """
    def _load_library(self):
        try:
            # Try specific target first
            self.lib = load_backend_lib("cuda", "sm61")
            logger.info("Loaded optimized CUDA SM61 backend.")
        except Exception:
            logger.info("SM61 specific backend not found, falling back to generic CUDA.")
            try:
                self.lib = load_backend_lib("cuda")
            except Exception as e:
                logger.warning(f"Failed to load generic CUDA backend: {e}")

        if self.lib:
            # Standard definitions
            import ctypes
            self.lib.create_tensor.restype = ctypes.c_void_p
            self.lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
            self.lib.free_tensor.argtypes = [ctypes.c_void_p]

    def get_device_info(self):
        return {"vendor": "NVIDIA", "backend": "CUDA", "arch": "Pascal (SM6.1)"}
