# Guide: Creating GPU Plugins

This guide explains how to implement custom GPU backends (CUDA and OpenCL) for Inference-PIO.

## Overview

GPU plugins leverage high-performance compute kernels for inference. Inference-PIO supports:
*   **CUDA:** For NVIDIA GPUs.
*   **OpenCL:** For AMD, Intel, and other OpenCL-compatible devices.

## CUDA Backends

CUDA kernels are implemented in `.cu` files and compiled using `nvcc`.

### File Structure
```
src/inference_pio/plugins/cuda/
├── base.py                 # Python wrapper (ctypes)
├── c_src/
│   ├── tensor_ops_cuda.cu  # Main CUDA implementation
│   └── fused_ops.cu        # Optimized fused kernels
└── __init__.py
```

### Compilation

The `build_ops.py` script automatically detects `nvcc` and compiles `.cu` files into shared libraries (`.dll` or `.so`).

**Note:** Cross-compiling CUDA (e.g., building Windows DLLs on Linux) is currently *not supported* by `build_ops.py` due to NVCC complexities. You must build on the target OS or use native compilation.

## OpenCL Backends

OpenCL kernels are implemented in `.c` files that load OpenCL libraries dynamically.

### File Structure
```
src/inference_pio/plugins/common/c_src/
├── tensor_ops_opencl.c    # Shared OpenCL implementation
└── cl_minimal.h           # Minimal OpenCL headers (dependency-free)

src/inference_pio/plugins/amd/
├── base.py                # AMD-specific Python wrapper
└── __init__.py

src/inference_pio/plugins/intel/
├── base.py                # Intel-specific Python wrapper
└── __init__.py
```

### Dynamic Loading
The OpenCL backend uses `dlopen` (Linux) or `LoadLibrary` (Windows) to load vendor-specific OpenCL implementations at runtime. This avoids build-time dependencies on OpenCL SDKs.

To implement a new OpenCL backend:
1.  Implement kernels in `tensor_ops_opencl.c`.
2.  Use the `VENDOR_FILTER` macro to conditionally compile vendor-specific logic if needed.
3.  Ensure your C code is compatible with both Windows (MSVC/MinGW) and Linux (GCC).

### Building OpenCL plugins
Run:
```bash
python3 build_ops.py
```
This will compile separate shared libraries for AMD and Intel backends (`libtensor_ops_amd.so`, `libtensor_ops_intel.so`) using the common source.

### Cross-Compilation
The OpenCL backend supports full cross-compilation:
*   **Linux Host:** Builds Windows DLLs via MinGW.
*   **Windows Host:** Builds Linux `.so` files via WSL.
