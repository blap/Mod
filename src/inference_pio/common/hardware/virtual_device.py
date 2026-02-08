"""
Virtual Device - Dependency Free
"""

from typing import Optional, Dict, Any
import logging

# Use Core Engine Layers
from ...core.engine.layers import Module
from ...core.engine.backend import Tensor

logger = logging.getLogger(__name__)

class VirtualDevice:
    """
    Represents a virtual execution device (e.g., a partition of RAM or a logical core).
    """
    def __init__(self, device_id: int, memory_limit_gb: float):
        self.device_id = device_id
        self.memory_limit_bytes = int(memory_limit_gb * 1024**3)
        self.current_memory_usage = 0
        self.loaded_modules = {}

    def load_module(self, name: str, module: Module):
        # Simulate memory tracking
        # size = sum(p.size * 4 for p in module.parameters())
        # For C-Engine module, we'd iterate parameters
        # module.state_dict() returns tensors
        size = 0
        for k, v in module.state_dict().items():
            size += v.size * 4 # float32

        if self.current_memory_usage + size > self.memory_limit_bytes:
            logger.warning(f"Device {self.device_id} OOM: Cannot load {name}")
            return False

        self.loaded_modules[name] = module
        self.current_memory_usage += size
        return True

    def execute(self, module_name: str, *args, **kwargs):
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name](*args, **kwargs)
        raise RuntimeError(f"Module {module_name} not loaded on device {self.device_id}")

class VirtualExecutionSimulator:
    def __init__(self, num_virtual_devices: int, memory_per_device_gb: float):
        self.devices = [VirtualDevice(i, memory_per_device_gb) for i in range(num_virtual_devices)]

    def execute_partition_on_device(self, partition: Module, input_data: Any, device_id: int):
        device = self.devices[device_id]
        # In a real sim, we'd move weights.
        # Here we assume shared memory (CPU) but enforce logical constraints.
        return partition(input_data)

    def cleanup(self):
        self.devices.clear()

    def get_stats(self):
        return {f"device_{d.device_id}_usage": d.current_memory_usage for d in self.devices}
