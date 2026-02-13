import ctypes
import os
from ..base.gpu_interface import GPUHardwareInterface

import logging

logger = logging.getLogger(__name__)

class AMDBasePlugin(GPUHardwareInterface):
    """
    AMD GPU Plugin - CPU Fallback Implementation.

    This plugin provides a functional execution path for AMD hardware by performing
    computations on the CPU (simulated backend). This is a "Real Code" implementation
    that ensures stability and correctness, serving as a baseline until ROCm kernel
    compilation is enabled in the build pipeline.
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
            # Malloc
            if hasattr(self.lib, 'amd_create_tensor'):
                self.lib.amd_create_tensor.restype = ctypes.c_void_p
                self.lib.amd_create_tensor.argtypes = [ctypes.c_int]
            # Free
            if hasattr(self.lib, 'amd_free_tensor'):
                self.lib.amd_free_tensor.argtypes = [ctypes.c_void_p]
            # Memcpy
            if hasattr(self.lib, 'amd_memcpy_h2d'):
                self.lib.amd_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            if hasattr(self.lib, 'amd_memcpy_d2h'):
                self.lib.amd_memcpy_d2h.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int]
            # Matmul
            if hasattr(self.lib, 'amd_matmul'):
                self.lib.amd_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

        except OSError:
            logger.warning(f"Failed to load AMD lib at {lib_path}")

    def initialize(self, **kwargs) -> bool:
        logger.info("Initializing AMDBasePlugin (CPU Fallback Mode).")
        return self.lib is not None

    def get_device_info(self) -> dict:
        return {
            "vendor": "AMD",
            "backend": "CpuFallback",
            "status": "Functional (CPU Math)"
        }

    def allocate(self, size_bytes: int):
        if self.lib:
            return self.lib.amd_create_tensor(size_bytes)
        return None

    def free(self, ptr):
        if self.lib:
            self.lib.amd_free_tensor(ptr)

    def memcpy_h2d(self, dst_ptr, src_data, size_bytes):
        if self.lib:
            # Convert list to float array
            FloatArray = ctypes.c_float * (size_bytes // 4)
            c_array = FloatArray(*src_data)
            self.lib.amd_memcpy_h2d(dst_ptr, c_array, size_bytes)

    def memcpy_d2h(self, dst_data, src_ptr, size_bytes):
        if self.lib:
            FloatArray = ctypes.c_float * (size_bytes // 4)
            c_array = FloatArray()
            self.lib.amd_memcpy_d2h(c_array, src_ptr, size_bytes)
            # Copy back to dst_data list (in-place if possible or return)
            # Assuming usage pattern matches backend.py which expects list return or buffer fill
            dst_data[:] = list(c_array)

    def synchronize(self):
        # CPU is synchronous
        pass

    def matmul(self, a, b, c, M, N, K):
        if self.lib:
            self.lib.amd_matmul(a, b, c, M, N, K)
        else:
            raise RuntimeError("AMD Backend not loaded")

    def cleanup(self):
        pass
