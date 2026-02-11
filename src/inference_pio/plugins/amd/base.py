import ctypes
import os
from ..base.gpu_interface import GPUHardwareInterface

class AMDBasePlugin(GPUHardwareInterface):
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
            self.lib.amd_create_tensor.restype = ctypes.c_void_p
        except OSError:
            print(f"Warning: Failed to load AMD lib at {lib_path}")

    def initialize(self, **kwargs) -> bool:
        return self.lib is not None

    def get_device_info(self) -> dict:
        return {"vendor": "AMD", "backend": "OpenCL/ROCm"}

    def allocate(self, size_bytes: int):
        return self.lib.amd_create_tensor(size_bytes)

    def free(self, ptr):
        self.lib.amd_free_tensor(ptr)

    def memcpy_h2d(self, dst_ptr, src_data, size_bytes):
        pass # Emulation

    def memcpy_d2h(self, dst_data, src_ptr, size_bytes):
        pass

    def synchronize(self):
        pass

    def matmul(self, a, b, c, M, N, K):
        self.lib.amd_matmul(a, b, c)

    def cleanup(self):
        pass
