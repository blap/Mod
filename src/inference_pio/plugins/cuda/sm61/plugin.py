import os
import ctypes
from ..base import CUDABasePlugin

class CUDASM61Plugin(CUDABasePlugin):
    def _load_library(self):
        # Override to try loading SM61 specific lib first
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        lib_name = "libtensor_ops_cuda_sm61.dll" if os.name == 'nt' else "libtensor_ops_cuda_sm61.so"
        lib_path = os.path.join(base_path, "cuda", "c_src", lib_name)

        try:
            if os.path.exists(lib_path):
                self.lib = ctypes.CDLL(lib_path)
                print(f"Loaded Optimized SM61 Backend: {lib_path}")
            else:
                print(f"SM61 Optimized lib not found at {lib_path}, falling back to generic.")
                super()._load_library()
        except OSError:
            super()._load_library()

    def get_device_info(self):
        return {"vendor": "NVIDIA", "backend": "CUDA", "arch": "sm_61", "model": "GTX 10-series"}

    def matmul(self, a, b, c, M, N, K):
        self.lib.tensor_matmul(a, b, c)
