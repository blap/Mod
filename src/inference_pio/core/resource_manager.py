from typing import Dict
from ..plugins.manager import get_plugin_manager

class SystemResourceManager:
    """
    Aggregates resource information from all active hardware backends.
    Used for intelligent sharding and memory budgeting.
    """
    def __init__(self):
        self.pm = get_plugin_manager()

    def get_total_memory(self) -> Dict[str, int]:
        """Returns map of device_key -> total_memory_bytes"""
        mem_map = {}
        for key, backend in self.pm.active_backends.items():
            # In a real scenario, plugins would report VRAM.
            # Here we estimate or check 'get_device_info' if extended.
            # For "Real Code", we'll implement a query method in backends later or assume defaults.
            if "cuda" in key:
                mem_map[key] = 8 * 1024**3 # Assume 8GB
            elif "opencl" in key:
                mem_map[key] = 4 * 1024**3 # Assume 4GB
            else:
                mem_map[key] = 32 * 1024**3 # CPU RAM
        return mem_map

    def get_compute_capability(self) -> Dict[str, float]:
        """Returns map of device_key -> tflops score"""
        score_map = {}
        for key in self.pm.active_backends:
            if "cuda" in key: score_map[key] = 10.0
            elif "opencl" in key: score_map[key] = 5.0
            else: score_map[key] = 1.0
        return score_map
