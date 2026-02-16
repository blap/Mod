# Creating GPU Plugins

GPU plugins integrate hardware accelerators (NVIDIA, AMD, Intel ARC) into the inference engine.

## Base Classes

*   **CUDA:** Inherit from `CUDABasePlugin` (`src/inference_pio/plugins/cuda/base.py`).
*   **OpenCL (AMD/Intel):** Inherit from `AMDBasePlugin` or `IntelBasePlugin`.

## CUDA Example

### 1. `plugin_manifest.json`

```json
{
    "name": "NVIDIATuringPlugin",
    "version": "1.0.0",
    "type": "hardware",
    "description": "Optimized backend for NVIDIA Turing GPUs",
    "entry_point": "plugin:create_plugin",
    "compatibility": {
        "compute_capability": "7.5"
    }
}
```

### 2. `plugin.py`

```python
from ..base import CUDABasePlugin

class NVIDIATuringPlugin(CUDABasePlugin):
    def initialize(self):
        super().initialize()
        # Tune specific kernels for Turing
        self._tune_kernels()

def create_plugin():
    return NVIDIATuringPlugin()
```

### 3. Custom Kernels (`.cu`)

1.  Write optimized CUDA C++ kernels in `c_src/`.
2.  Update `build_ops.py` to compile them using `nvcc` into a shared library (e.g., `libtensor_ops_cuda_turing.so`).
3.  Ensure the functions are exported with `extern "C"`.

## OpenCL Example

OpenCL plugins typically share the `tensor_ops_opencl.c` source but can be compiled with different macros or linked against different vendor SDKs.

1.  Create `src/inference_pio/plugins/amd/rx6000/`.
2.  Implement `plugin.py` inheriting from `AMDBasePlugin`.
3.  Ensure `build_ops.py` compiles the OpenCL backend for this target.
