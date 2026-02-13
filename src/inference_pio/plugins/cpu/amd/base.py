import os
from ...base.cpu_interface import CPUHardwareInterface

class AMDCPUBasePlugin(CPUHardwareInterface):
    def initialize(self, **kwargs) -> bool:
        self.configure_environment()
        return True

    def get_cpu_info(self) -> dict:
        return {"vendor": "AMD", "arch": "x86_64"}

    def configure_environment(self) -> None:
        # AMD Zen optimizations
        os.environ["OMP_PROC_BIND"] = "TRUE"
        os.environ["OMP_PLACES"] = "cores"

    def get_library_path(self) -> str:
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if os.name == 'nt':
            return os.path.join(base, "cpu", "c_src", "libtensor_ops.dll")
        return os.path.join(base, "cpu", "c_src", "libtensor_ops.so")
