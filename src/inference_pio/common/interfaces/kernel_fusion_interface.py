"""
Interface for kernel fusion functionality in the Mod project.

This module defines a clear interface for kernel fusion operations
that can be implemented by different fusion strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch


class KernelFusionManagerInterface(ABC):
    """
    Interface for kernel fusion operations.
    """

    @abstractmethod
    def setup_kernel_fusion(self, **kwargs) -> bool:
        """
        Set up kernel fusion system for optimizing model operations.

        Args:
            **kwargs: Kernel fusion configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def apply_kernel_fusion(self, model: torch.nn.Module = None) -> bool:
        """
        Apply kernel fusion optimizations to the model.

        Args:
            model: Model to optimize (if None, uses internal model if available)

        Returns:
            True if optimization was successful, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_fusion_manager(self):
        """
        Get the kernel fusion manager instance.

        Returns:
            Kernel fusion manager instance
        """
        pass