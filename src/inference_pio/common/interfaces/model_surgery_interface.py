"""
Interface for model surgery functionality in the Mod project.

This module defines a clear interface for model surgery operations
that can be implemented by different surgery strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch


class ModelSurgeryManagerInterface(ABC):
    """
    Interface for model surgery operations.
    """

    @abstractmethod
    def setup_model_surgery(self, **kwargs) -> bool:
        """
        Set up model surgery system for identifying and removing non-essential components.

        Args:
            **kwargs: Model surgery configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def enable_model_surgery(self, **kwargs) -> bool:
        """
        Enable model surgery for the plugin to identify and temporarily remove
        non-essential components during inference.

        Args:
            **kwargs: Model surgery configuration parameters

        Returns:
            True if model surgery was enabled successfully, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def perform_model_surgery(
        self,
        model: torch.nn.Module = None,
        components_to_remove: Optional[List[str]] = None,
        preserve_components: Optional[List[str]] = None,
    ) -> torch.nn.Module:
        """
        Perform model surgery by identifying and removing non-essential components.

        Args:
            model: Model to perform surgery on (if None, uses internal model)
            components_to_remove: Specific components to remove (None means auto-detect)
            preserve_components: Components to preserve even if they're removable

        Returns:
            Modified model with surgery applied
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def restore_model_from_surgery(
        self, model: torch.nn.Module = None, surgery_id: Optional[str] = None
    ) -> torch.nn.Module:
        """
        Restore a model from surgery by putting back removed components.

        Args:
            model: Model to restore (if None, uses internal model)
            surgery_id: Specific surgery to reverse (None means reverse latest)

        Returns:
            Restored model
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def analyze_model_for_surgery(self, model: torch.nn.Module = None) -> Dict[str, Any]:
        """
        Analyze a model to identify potential candidates for surgical removal.

        Args:
            model: Model to analyze (if None, uses internal model)

        Returns:
            Dictionary containing analysis results
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_surgery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about performed model surgeries.

        Returns:
            Dictionary containing surgery statistics
        """
        pass