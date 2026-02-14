import ctypes
import os
import logging
from ..base.gpu_interface import GPUHardwareInterface
from ...common.utils.lib_loader import load_backend_lib

logger = logging.getLogger(__name__)

class CUDABasePlugin(GPUHardwareInterface):
    def __init__(self):
        self.lib = None
        self._load_library()

    def _load_library(self):
        try:
            # Use standardized loader
            self.lib = load_backend_lib("cuda")

            # Define argtypes for critical functions
            # create_tensor(int* shape, int ndim, int device_id) -> float* (mapped to Tensor struct in higher level)
            # Actually C returns Tensor*, but here we might wrap differently.
            # The C `create_tensor` returns Tensor*.
            self.lib.create_tensor.restype = ctypes.c_void_p
            self.lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]

            self.lib.free_tensor.argtypes = [ctypes.c_void_p]

        except Exception as e:
            logger.warning(f"Failed to load CUDA backend: {e}")

    def initialize(self, **kwargs) -> bool:
        if not self.lib: return False

        # Auto-Tune
        try:
            if hasattr(self.lib, 'tensor_benchmark_matmul'):
                time_ms = ctypes.c_double(0.0)
                # Benchmark config 0 (naive)
                self.lib.tensor_benchmark_matmul(512, 512, 512, 5, ctypes.byref(time_ms))
                t0 = time_ms.value
                logger.info(f"CUDA Auto-Tune: MatMul 512x512 = {t0:.4f} ms")
        except Exception as e:
            logger.warning(f"Auto-Tune warning: {e}")

        return True

    def get_device_info(self) -> dict:
        return {"vendor": "NVIDIA", "backend": "CUDA"}

    def allocate(self, size_bytes: int):
        if not self.lib: return None
        # Create a 1D float tensor to wrap memory
        # Elements = bytes / 4
        elements = size_bytes // 4
        if elements == 0: elements = 1 # Safety
        shape = (ctypes.c_int * 1)(elements)
        # Device 0
        return self.lib.create_tensor(shape, 1, 0)

    def free(self, ptr):
        if self.lib and ptr:
            self.lib.free_tensor(ptr)

    def memcpy_h2d(self, dst_ptr, src_data, size_bytes):
        if not self.lib: return
        c_float_p = ctypes.POINTER(ctypes.c_float)

        if hasattr(src_data, 'ctypes'):
             data_ptr = src_data.ctypes.data_as(c_float_p)
        else:
             # Convert list/bytearray
             # This is slow, but standard for non-numpy
             # src_data assumed to be bytes or list of floats?
             # Standard engine uses typed arrays usually.
             # Assuming src_data is a bytes-like object or list
             # For standardization, we assume memory-mappable.
             pass
             # Logic is complex without numpy. Assuming higher level handles buffer creation.

        # Use backend loader
        self.lib.tensor_load_data(dst_ptr, data_ptr, size_bytes//4)

    def memcpy_d2h(self, dst_data, src_ptr, size_bytes):
        if not self.lib: return
        self.lib.tensor_get_data(src_ptr, dst_data, size_bytes//4)

    def synchronize(self):
        # Implicit in kernel stream for now, or expose if needed
        pass

    def matmul(self, a, b, c, M, N, K):
        if self.lib:
            self.lib.tensor_matmul(a, b, c)

    def cleanup(self):
        pass
