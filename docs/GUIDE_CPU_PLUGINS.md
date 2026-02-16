# Creating CPU Plugins

CPU plugins allow you to customize how the inference engine interacts with the host processor. While the core "CPU Backend" is a shared C library (`libtensor_ops.so`), a CPU *Plugin* can manage initialization, thread affinity, and hardware-specific optimizations.

## Base Class

All CPU plugins should inherit from `NativeCPUPlugin` located in `src/inference_pio/plugins/cpu/base.py`.

## Implementation

Create a new directory `src/inference_pio/plugins/cpu/<arch_name>/` (e.g., `intel_avx512`).

### 1. `plugin_manifest.json`

```json
{
    "name": "IntelAVX512Plugin",
    "version": "1.0.0",
    "type": "hardware",
    "description": "Optimized CPU backend for Intel AVX512",
    "entry_point": "plugin:create_plugin"
}
```

### 2. `plugin.py`

```python
from ..base import NativeCPUPlugin
import logging

class IntelAVX512Plugin(NativeCPUPlugin):
    def initialize(self):
        super().initialize()
        # Set specific OpenMP flags or thread affinity
        self._set_thread_affinity()

    def _set_thread_affinity(self):
        # Implementation using os.sched_setaffinity or ctypes
        pass

def create_plugin():
    return IntelAVX512Plugin()
```

### 3. C-Backend Extensions

If you need custom C kernels for your CPU architecture:
1.  Add source files to `src/inference_pio/plugins/cpu/<arch_name>/c_src/`.
2.  Update `build_ops.py` to compile your specific library (e.g., `libtensor_ops_avx512.so`).
3.  Override `load_backend_lib` in your Python plugin class to load your custom `.so` instead of the default.

```python
    def load_backend_lib(self):
        # Custom loading logic
        pass
```
