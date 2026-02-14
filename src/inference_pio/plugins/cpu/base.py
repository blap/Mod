import ctypes
import os
import logging
from ..base.cpu_interface import CPUHardwareInterface
from ...common.utils.lib_loader import load_backend_lib
from ...core.engine.backend import Tensor

logger = logging.getLogger(__name__)

class NativeCPUPlugin(CPUHardwareInterface):
    """
    Base Native CPU Plugin implementation handling core C library interaction.
    """
    def __init__(self):
        self.lib = None
        self._load_library()

    def _load_library(self):
        try:
            # Use standardized loader for CPU backend
            self.lib = load_backend_lib("cpu")
            self._setup_signatures()
        except Exception as e:
            logger.error(f"Failed to load Native CPU backend: {e}")

    def _setup_signatures(self):
        if not self.lib: return
        # Standardize function signatures if needed
        # Assuming int* for shape, int ndim, int device_id -> void*
        self.lib.create_tensor.restype = ctypes.c_void_p
        self.lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        self.lib.free_tensor.argtypes = [ctypes.c_void_p]

    def initialize(self, config: dict = None) -> bool:
        return self.lib is not None

    def get_cpu_info(self) -> dict:
        return {"vendor": "Generic", "backend": "Native C"}

    def configure_environment(self) -> None:
        pass

    def get_library_path(self) -> str:
        # Helper to get loaded path if needed
        return "libtensor_ops"

    # --- Standard Allocator Interface ---
    def allocate(self, size_bytes: int):
        if not self.lib: return None
        elements = size_bytes // 4
        if elements == 0: elements = 1
        shape = (ctypes.c_int * 1)(elements)
        # Device -1 for CPU in C backend logic
        return self.lib.create_tensor(shape, 1, -1)

    def free(self, ptr):
        if self.lib and ptr:
            self.lib.free_tensor(ptr)
