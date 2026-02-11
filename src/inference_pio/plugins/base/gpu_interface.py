from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

class GPUHardwareInterface(ABC):
    """
    Abstract Base Class for GPU Hardware Plugins.
    Defines the contract for memory management and kernel execution.
    """

    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """Initialize the hardware context (e.g., CUDA context, OpenCL queue)."""
        pass

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """Return metadata about the device (name, memory, compute capability)."""
        pass

    @abstractmethod
    def allocate(self, size_bytes: int) -> Any:
        """Allocate memory on the device. Returns a handle/pointer."""
        pass

    @abstractmethod
    def free(self, ptr: Any) -> None:
        """Free memory on the device."""
        pass

    @abstractmethod
    def memcpy_h2d(self, dst_ptr: Any, src_data: Any, size_bytes: int) -> None:
        """Copy data from Host to Device."""
        pass

    @abstractmethod
    def memcpy_d2h(self, dst_data: Any, src_ptr: Any, size_bytes: int) -> None:
        """Copy data from Device to Host."""
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """Block until all device operations are complete."""
        pass

    # Kernel Launch Abstraction
    # Ideally, specific operations (matmul) are methods here, or we expose a generic launcher.
    # For a plugin system, we enforce standard ops.

    @abstractmethod
    def matmul(self, a_ptr: Any, b_ptr: Any, c_ptr: Any, M: int, N: int, K: int) -> None:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass
