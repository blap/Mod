from ..common.opencl_backend import OpenCLBackend

class AMDBasePlugin(OpenCLBackend):
    """
    AMD GPU Plugin using OpenCL Backend.
    Provides "Real Code" execution for AMD GPUs via the system's OpenCL runtime.
    This avoids the need for ROCm/HIP compiler dependencies at build time.
    """
    def __init__(self):
        super().__init__(platform_vendor_filter="AMD") # Filter for AMD/Advanced Micro Devices

    def get_device_info(self) -> dict:
        info = super().get_device_info()
        info["vendor"] = "AMD"
        return info
