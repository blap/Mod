import os
from ..base import IntelCPUBasePlugin

class Intel_i5_10210U_Plugin(IntelCPUBasePlugin):
    def get_cpu_info(self):
        return {
            "vendor": "Intel",
            "model": "i5-10210U",
            "cores": 4,
            "threads": 8,
            "instructions": ["AVX2", "FMA3"]
        }

    def configure_environment(self) -> None:
        super().configure_environment()
        # Specific tuning for 4 physical cores
        os.environ["OMP_NUM_THREADS"] = "4"
        # i5-10210U is hyperthreaded, but OMP often prefers physical cores
