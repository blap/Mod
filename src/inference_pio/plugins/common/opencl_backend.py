import ctypes
import os
import logging
from ..base.gpu_interface import GPUHardwareInterface

logger = logging.getLogger(__name__)

class OpenCLBackend(GPUHardwareInterface):
    """
    Standardized OpenCL Backend for AMD and Intel GPUs.
    Loads a compiled C library (libtensor_ops_amd.so or libtensor_ops_intel.so)
    that implements the same interface as libtensor_ops_cuda.so and libtensor_ops.so (CPU).
    """
    def __init__(self, platform_vendor_filter="AMD"):
        self.lib = None
        self.platform_vendor_filter = platform_vendor_filter
        self._load_library()

    def _load_library(self):
        # Determine library name based on vendor and OS
        vendor_suffix = self.platform_vendor_filter.lower()
        if os.name == 'nt':
            lib_name = f"libtensor_ops_{vendor_suffix}.dll"
        else:
            lib_name = f"libtensor_ops_{vendor_suffix}.so"

        # Path: src/inference_pio/plugins/{vendor}/c_src/{lib_name}
        # Assuming we are in src/inference_pio/plugins/common/ or similar import path
        # Adjust logic to find path relative to THIS file
        base_dir = os.path.dirname(os.path.dirname(__file__)) # src/inference_pio/plugins
        lib_path = os.path.join(base_dir, vendor_suffix, "c_src", lib_name)

        if not os.path.exists(lib_path):
             # Fallback check
             logger.warning(f"Compiled OpenCL library not found at {lib_path}")
             return

        try:
            self.lib = ctypes.CDLL(lib_path)
            self._setup_signatures()
            logger.info(f"Loaded OpenCL backend from {lib_path}")
        except OSError as e:
            logger.error(f"Failed to load OpenCL library {lib_path}: {e}")

    def _setup_signatures(self):
        if not self.lib: return

        # Must match tensor_ops_opencl.c exports
        # create_tensor(int* shape, int ndim, int device_id) -> Tensor*
        self.lib.create_tensor.restype = ctypes.c_void_p
        self.lib.create_tensor.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]

        self.lib.free_tensor.argtypes = [ctypes.c_void_p]

        # tensor_load_data(Tensor* t, float* buffer, int size)
        self.lib.tensor_load_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        # tensor_get_data(Tensor* t, float* buffer, int size)
        self.lib.tensor_get_data.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]

        # Ops
        self.lib.tensor_fill.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.lib.tensor_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.tensor_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.tensor_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.tensor_rms_norm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float]
        self.lib.tensor_silu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.tensor_gelu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        # tensor_rope(q, k, cos, sin, out_q, out_k)
        self.lib.tensor_rope.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.tensor_softmax.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        # tensor_topk(input, k, out_val, out_idx)
        self.lib.tensor_topk.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]


    def initialize(self, **kwargs) -> bool:
        # Check if C lib loaded
        return self.lib is not None

    def get_device_info(self) -> dict:
        return {"backend": "OpenCL C", "vendor": self.platform_vendor_filter, "lib": str(self.lib)}

    # --- Interface Implementation ---
    # The Backend class in backend.py usually handles Tensor creation using .lib directly if device="opencl..."
    # But if "allocate" is called by legacy code:

    def allocate(self, size_bytes: int):
        # This allocates a RAW cl_mem or Tensor*?
        # The CUDA plugin returns a Tensor* pointer (void_p).
        # OpenCL backend C code create_tensor returns Tensor*.
        # But `allocate` usually implies simple memory block.
        # However, the unified backend.py expects create_tensor to be called on self.lib.
        # So this allocate method might be redundant if we fully standardized.
        # Let's support it by creating a 1D tensor as a container.
        shape = (ctypes.c_int * 1)(size_bytes // 4)
        return self.lib.create_tensor(shape, 1, 0)

    def free(self, ptr):
        self.lib.free_tensor(ptr)

    def memcpy_h2d(self, dst_ptr, src_data, size_bytes):
        # Convert src_data list to float array
        if hasattr(src_data, 'ctypes'):
             data_ptr = src_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
             arr = (ctypes.c_float * (size_bytes//4))(*src_data)
             data_ptr = arr
        self.lib.tensor_load_data(dst_ptr, data_ptr, size_bytes//4)

    def memcpy_d2h(self, dst_data, src_ptr, size_bytes):
        # dst_data is a ctypes float array usually
        self.lib.tensor_get_data(src_ptr, dst_data, size_bytes//4)

    def synchronize(self):
        # C backend handles queue finish in ops usually, or we can add explicit export
        pass

    def cleanup(self):
        pass
