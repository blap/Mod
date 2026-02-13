from ..common.opencl_backend import OpenCLBackend

class IntelBasePlugin(OpenCLBackend):
    """
    Intel GPU Plugin using OpenCL Backend.
    Provides "Real Code" execution for Intel GPUs (iGPU/Arc) via the system's OpenCL runtime.
    This effectively replaces the need for a complex SYCL/DPC++ build in this environment.
    """
    def __init__(self):
        super().__init__(platform_vendor_filter="Intel")

    def get_device_info(self) -> dict:
        info = super().get_device_info()
        info["vendor"] = "Intel"
        return info
