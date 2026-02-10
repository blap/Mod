# Hardware Plugins

## Overview

Inference-PIO utilizes a modular hardware plugin system to optimize execution for different backend targets (CPU, CUDA, Metal, etc.). This system allows seamless switching between execution engines without changing model code.

**Key Design:** Hardware plugins interface directly with `libtensor_ops` (C/C++ backend) via `ctypes` or `pybind11` (simulated).

## 1. Creating a Hardware Plugin

A hardware plugin must implement the `HardwareBackendInterface`.

```python
from ...common.interfaces.backend_interface import HardwareBackendInterface

class MyHardwarePlugin(HardwareBackendInterface):
    def initialize(self):
        # Load shared library
        self.lib = ctypes.CDLL("./libmybackend.so")

    def matmul(self, a, b):
        # Call C function
        return self.lib.matmul(a, b)

    def rope(self, x, freqs_cis):
        # Call C function
        return self.lib.rope(x, freqs_cis)
```

## 2. Supported Backends

### CPU Backend (`libtensor_ops_cpu`)
Default backend for x86_64/ARM.
*   **Features:** SIMD (AVX2/NEON), OpenMP threading.
*   **Kernel Implementation:** C-based `gemm`, `rope`, `softmax`.

### CUDA Backend (`libtensor_ops_cuda`)
Accelerated backend for NVIDIA GPUs.
*   **Features:** Tensor Cores, FlashAttention-2, fused kernels.
*   **Requirement:** CUDA Toolkit 11.8+.

## 3. Extending the Backend

To add a new operation (e.g., a custom activation):
1.  **Implement in C/CUDA:** Add the kernel to `src/backend/kernels/`.
2.  **Expose in Python:** Update `src/inference_pio/core/engine/backend.py` to include the new function signature.
3.  **Compile:** Rebuild the shared library.

## 4. Automatic Selection

The system automatically selects the best available backend at runtime:
1.  Check for CUDA availability.
2.  Check for AVX2 support.
3.  Fallback to generic CPU implementation.

## 5. Performance Tuning

Hardware plugins can be configured via `SystemProfile`:
*   `num_threads`: OMP_NUM_THREADS
*   `precision`: FP32, FP16, INT8 (if supported)
*   `memory_limit`: Max memory usage.
