from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

# Avoid hard torch dependency in interface definitions
if TYPE_CHECKING:
    import torch
    Tensor = torch.Tensor
else:
    # Use Any or mock for runtime if torch missing
    Tensor = Any

class KernelFusionManagerInterface:
    """
    Interface for kernel fusion managers.
    """

    def fuse_operations(self, model: Any) -> Any:
        """
        Apply kernel fusion to the model.

        Args:
            model: The model to optimize

        Returns:
            Optimized model
        """
        raise NotImplementedError

    def register_fusion_pattern(self, pattern: Any, handler: Any) -> None:
        """
        Register a new fusion pattern.

        Args:
            pattern: Pattern to match
            handler: Function to handle the fusion
        """
        raise NotImplementedError

    def get_fusion_stats(self) -> Dict[str, Any]:
        """
        Get statistics about performed fusions.

        Returns:
            Dictionary containing fusion statistics
        """
        raise NotImplementedError
