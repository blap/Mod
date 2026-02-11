from ..base import CUDABasePlugin

class CUDASM61Plugin(CUDABasePlugin):
    def get_device_info(self):
        return {"vendor": "NVIDIA", "backend": "CUDA", "arch": "sm_61", "model": "GTX 10-series"}

    def matmul(self, a, b, c, M, N, K):
        # Override to use sm61 optimized kernel if available
        # For now, falls back to base, but structurally separate
        super().matmul(a, b, c, M, N, K)
