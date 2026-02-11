import os
from ...base.cpu_interface import CPUHardwareInterface

class IntelCPUBasePlugin(CPUHardwareInterface):
    def initialize(self, **kwargs) -> bool:
        self.configure_environment()
        return True

    def get_cpu_info(self) -> dict:
        return {"vendor": "Intel", "arch": "x86_64"}

    def configure_environment(self) -> None:
        # Intel OpenMP optimizations
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

    def get_library_path(self) -> str:
        # Default to generic libtensor_ops unless overridden
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if os.name == 'nt':
            return os.path.join(base, "cpu", "c_src", "libtensor_ops.dll")
        return os.path.join(base, "cpu", "c_src", "libtensor_ops.so")
