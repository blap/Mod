import ctypes
import os
from ..base.gpu_interface import GPUHardwareInterface

class IntelBasePlugin(GPUHardwareInterface):
    def initialize(self, **kwargs): return True
    def get_device_info(self): return {"vendor": "Intel", "backend": "OneAPI"}
    def allocate(self, size): return 1
    def free(self, ptr): pass
    def memcpy_h2d(self, dst, src, size): pass
    def memcpy_d2h(self, dst, src, size): pass
    def synchronize(self): pass
    def matmul(self, a, b, c, m, n, k): pass
    def cleanup(self): pass
