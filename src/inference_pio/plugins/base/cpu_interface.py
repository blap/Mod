from abc import ABC, abstractmethod
from typing import Any, Dict

class CPUHardwareInterface(ABC):
    """
    Abstract Base Class for CPU Hardware Plugins.
    """

    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize CPU context (threading, affinity)."""
        pass

    @abstractmethod
    def get_cpu_info(self) -> Dict[str, Any]:
        """Return metadata about the CPU."""
        pass

    @abstractmethod
    def configure_environment(self) -> None:
        """Apply environment variables for optimization (OMP, KMP)."""
        pass

    @abstractmethod
    def get_library_path(self) -> str:
        """Return path to the optimized C library for this CPU."""
        pass
