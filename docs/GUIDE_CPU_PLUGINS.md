# Guide: Creating CPU Plugins

This guide explains how to implement custom CPU backends for Inference-PIO, including the build process for cross-platform support.

## Architecture

CPU plugins extend the `NativeCPUPlugin` class. The core compute logic is implemented in C and compiled into a shared library (`.so` or `.dll`).

### File Structure
```
src/inference_pio/plugins/cpu/
├── base.py                 # Python wrapper (ctypes)
├── c_src/
│   ├── tensor_ops.c        # Main C implementation
│   ├── safetensors_loader.c # Optimized loader
│   └── tensor_ops.h        # Headers
└── __init__.py
```

## Compilation

Inference-PIO uses a unified build system `build_ops.py` that handles compilation for all plugins.

### 1. Adding New Ops
Implement your C function in `tensor_ops.c` and declare it with the `EXPORT` macro:

```c
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void my_custom_op(Tensor* a, Tensor* out) {
    // ... implementation ...
}
```

### 2. Building
Run the build script from the project root:
```bash
python3 build_ops.py
```

### 3. Cross-Compilation
The build system supports generating binaries for other platforms:
*   **Linux Host:** Install `mingw-w64` to build Windows DLLs.
*   **Windows Host:** Install WSL (`wsl --install`) to build Linux `.so` files.

Ensure your code is platform-agnostic (use standard C99/C11 features). Avoid platform-specific headers like `<windows.h>` unless wrapped in `#ifdef _WIN32`.

## Python Integration

In `base.py`, load the library and define the function signature:

```python
class MyCPUPlugin(NativeCPUPlugin):
    def __init__(self):
        super().__init__()
        # self.lib is automatically loaded by NativeCPUPlugin

        self.lib.my_custom_op.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.my_custom_op.restype = None

    def my_op(self, a, out):
        self.lib.my_custom_op(a.c_ptr, out.c_ptr)
```
