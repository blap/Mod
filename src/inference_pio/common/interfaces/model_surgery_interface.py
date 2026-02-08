from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    Module = torch.nn.Module
else:
    Module = Any

class ModelSurgeryManagerInterface:
    """
    Interface for model surgery operations.
    """

    def analyze_model(self, model: Module) -> Dict[str, Any]:
        """
        Analyze model structure for potential surgery.

        Args:
            model: The model to analyze

        Returns:
            Analysis results
        """
        raise NotImplementedError

    def perform_surgery(self, model: Module, surgery_config: Dict[str, Any]) -> Module:
        """
        Perform surgery on the model.

        Args:
            model: The model to modify
            surgery_config: Configuration for the surgery

        Returns:
            Modified model
        """
        raise NotImplementedError

    def restore_model(self, model: Module) -> Module:
        """
        Restore model to original state.

        Args:
            model: The modified model

        Returns:
            Restored model
        """
        raise NotImplementedError
