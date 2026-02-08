"""
Hardware Analyzer - Dependency Free
Analyzes system hardware capabilities.
"""

import logging
import platform
import os
import psutil
from typing import Dict, Any

logger = logging.getLogger(__name__)

class SystemProfile:
    def __init__(self):
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.cpu_threads = psutil.cpu_count(logical=True)
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        self.is_weak_hardware = self.total_memory_gb < 8.0 or self.cpu_cores < 4
        self.has_gpu = False # Default to False for C-Engine (CPU only)

        # Check for GPU via NVIDIA-SMI (Linux/Windows) just for info
        # Avoiding torch.cuda.is_available() dependency
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                self.has_gpu = True
                self.gpu_info = result.stdout.decode('utf-8').strip()
        except:
            pass

def get_system_profile() -> SystemProfile:
    return SystemProfile()

class HardwareAnalyzer:
    """
    Analyzes hardware capabilities to determine optimal configuration.
    """

    def analyze(self) -> Dict[str, Any]:
        profile = get_system_profile()
        return {
            "cpu_cores": profile.cpu_cores,
            "total_memory_gb": profile.total_memory_gb,
            "has_gpu": profile.has_gpu,
            "is_weak": profile.is_weak_hardware
        }
