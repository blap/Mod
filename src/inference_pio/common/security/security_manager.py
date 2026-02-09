from dataclasses import dataclass
from enum import Enum

@dataclass
class ResourceLimits:
    gpu_memory_gb: float = 0.0
    cpu_percent: float = 80.0
    memory_gb: float = 8.0
    disk_space_gb: float = 10.0

class SecurityLevel(Enum):
    MEDIUM_TRUST = "medium"

def initialize_plugin_isolation(*args, **kwargs): return True
def cleanup_plugin_isolation(*args, **kwargs): pass
