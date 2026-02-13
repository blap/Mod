import ctypes
import logging
from ..base.gpu_interface import GPUHardwareInterface
from ...common.utils.lib_loader import load_backend_lib

logger = logging.getLogger(__name__)

class IntelBasePlugin(GPUHardwareInterface):
    """
    Standard Intel GPU Plugin utilizing OpenCL backend.
    """
    def __init__(self):
        self.lib = None
        self._load_library()

    def _load_library(self):
        try:
            self.lib = load_backend_lib("intel")
            self._setup_signatures()
        except Exception as e:
            logger.warning(f"Failed to load Intel OpenCL backend: {e}")

    def _setup_signatures(self):
        if not self.lib: return
        self.lib.create_tensor.restype = ctypes.c_void_p
        self.lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        self.lib.free_tensor.argtypes = [ctypes.c_void_p]

    def initialize(self, **kwargs) -> bool:
        return self.lib is not None

    def get_device_info(self) -> dict:
        return {"vendor": "Intel", "backend": "OpenCL"}

    def allocate(self, size_bytes: int):
        if not self.lib: return None
        elements = size_bytes // 4
        if elements == 0: elements = 1
        shape = (ctypes.c_int * 1)(elements)
        return self.lib.create_tensor(shape, 1, 0)

    def free(self, ptr):
        if self.lib and ptr:
            self.lib.free_tensor(ptr)

    def memcpy_h2d(self, dst_ptr, src_data, size_bytes):
        pass

    def memcpy_d2h(self, dst_data, src_ptr, size_bytes):
        if not self.lib: return
        self.lib.tensor_get_data(src_ptr, dst_data, size_bytes//4)

    def synchronize(self):
        pass

    def matmul(self, a, b, c, M, N, K):
        if self.lib:
            self.lib.tensor_matmul(a, b, c)

    def cleanup(self):
        pass
