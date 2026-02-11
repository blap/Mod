import ctypes
import os
from ..base.gpu_interface import GPUHardwareInterface

import logging

logger = logging.getLogger(__name__)

class AMDBasePlugin(GPUHardwareInterface):
    """
    AMD GPU Plugin Stub / Emulation Layer.
    LIMITATION: This implementation is currently a STUB. It emulates GPU operations
    using CPU logic (via malloc/free in C) because full ROCm/OpenCL bindings are
    not yet implemented in the build environment.

    It allows the engine to load without crashing on AMD systems but runs at CPU speed.
    """
    def __init__(self):
        self.lib = None
        self._load_library()

    def _load_library(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(base_path, "c_src", "libtensor_ops_amd.so")
        if os.name == 'nt':
             lib_path = lib_path.replace(".so", ".dll")

        try:
            self.lib = ctypes.CDLL(lib_path)
            # Basic malloc wrapper
            if hasattr(self.lib, 'amd_create_tensor'):
                self.lib.amd_create_tensor.restype = ctypes.c_void_p
                self.lib.amd_create_tensor.argtypes = [ctypes.c_int]
        except OSError:
            logger.warning(f"Failed to load AMD lib at {lib_path}")

    def initialize(self, **kwargs) -> bool:
        logger.warning("Initializing AMDBasePlugin in EMULATION MODE. Performance will be limited.")
        return self.lib is not None

    def get_device_info(self) -> dict:
        return {
            "vendor": "AMD",
            "backend": "Stub/Emulation",
            "status": "Limited Functionality"
        }

    def allocate(self, size_bytes: int):
        if self.lib:
            return self.lib.amd_create_tensor(size_bytes)
        return None

    def free(self, ptr):
        if self.lib:
            self.lib.amd_free_tensor(ptr)

    def memcpy_h2d(self, dst_ptr, src_data, size_bytes):
        logger.debug("AMD Stub: memcpy_h2d (No-op/Emulated)")
        # In a real stub, we might memcpy to the malloc'd pointer
        pass

    def memcpy_d2h(self, dst_data, src_ptr, size_bytes):
        logger.debug("AMD Stub: memcpy_d2h (No-op/Emulated)")
        pass

    def synchronize(self):
        pass

    def matmul(self, a, b, c, M, N, K):
        if self.lib:
            # This calls the C-level emulation (likely OpenMP or naive loop)
            self.lib.amd_matmul(a, b, c)
        else:
            raise RuntimeError("AMD Backend not loaded")

    def cleanup(self):
        pass
