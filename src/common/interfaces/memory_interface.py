"""
Interface for memory management functionality in the Mod project.

This module defines a clear interface for memory management operations
that can be implemented by different memory management strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class MemoryManagerInterface(ABC):
    """
    Interface for memory management operations.
    """

    @abstractmethod
    def setup_memory_management(self, **kwargs) -> bool:
        """
        Set up memory management including swap and paging configurations.

        Args:
            **kwargs: Memory management configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        pass

    @abstractmethod
    def enable_tensor_paging(self, **kwargs) -> bool:
        """
        Enable tensor paging for the model to move parts between RAM and disk.

        Args:
            **kwargs: Tensor paging configuration parameters

        Returns:
            True if tensor paging was enabled successfully, False otherwise
        """
        pass

    @abstractmethod
    def enable_smart_swap(self, **kwargs) -> bool:
        """
        Enable smart swap functionality to configure additional swap on OS level.

        Args:
            **kwargs: Smart swap configuration parameters

        Returns:
            True if smart swap was enabled successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for the plugin.

        Returns:
            Dictionary containing memory statistics
        """
        pass

    @abstractmethod
    def force_memory_cleanup(self) -> bool:
        """
        Force cleanup of memory resources including cached tensors and swap files.

        Returns:
            True if cleanup was successful, False otherwise
        """
        pass

    @abstractmethod
    def start_predictive_memory_management(self, **kwargs) -> bool:
        """
        Start predictive memory management using ML algorithms to anticipate memory needs.

        Args:
            **kwargs: Configuration parameters for predictive management

        Returns:
            True if predictive management was started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop_predictive_memory_management(self) -> bool:
        """
        Stop predictive memory management.

        Returns:
            True if predictive management was stopped successfully, False otherwise
        """
        pass

    @abstractmethod
    def clear_cuda_cache(self) -> bool:
        """
        Clear CUDA cache to free up memory.

        Returns:
            True if cache was cleared successfully, False otherwise
        """
        pass

    @abstractmethod
    def setup_activation_offloading(self, **kwargs) -> bool:
        """
        Set up activation offloading system for managing intermediate activations between RAM and disk.

        Args:
            **kwargs: Activation offloading configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        pass

    @abstractmethod
    def enable_activation_offloading(self, **kwargs) -> bool:
        """
        Enable activation offloading for the model to move intermediate activations between RAM and disk.

        Args:
            **kwargs: Activation offloading configuration parameters

        Returns:
            True if activation offloading was enabled successfully, False otherwise
        """
        pass

    @abstractmethod
    def offload_activations(self, **kwargs) -> bool:
        """
        Offload specific activations to disk based on predictive algorithms.

        Args:
            **kwargs: Configuration parameters for offloading

        Returns:
            True if offloading was successful, False otherwise
        """
        pass

    @abstractmethod
    def predict_activation_access(self, **kwargs) -> Dict[str, Any]:
        """
        Predict which activations will be accessed based on input patterns.

        Args:
            **kwargs: Configuration parameters for prediction

        Returns:
            Dictionary mapping activation names to access probabilities
        """
        pass

    @abstractmethod
    def get_activation_offloading_stats(self) -> Dict[str, Any]:
        """
        Get statistics about activation offloading operations.

        Returns:
            Dictionary containing activation offloading statistics
        """
        pass

    @abstractmethod
    def setup_disk_offloading(self, **kwargs) -> bool:
        """
        Set up disk offloading system for managing model components between RAM and disk.

        Args:
            **kwargs: Disk offloading configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        pass

    @abstractmethod
    def enable_disk_offloading(self, **kwargs) -> bool:
        """
        Enable disk offloading for the model to move parts between RAM and disk.

        Args:
            **kwargs: Disk offloading configuration parameters

        Returns:
            True if disk offloading was enabled successfully, False otherwise
        """
        pass

    @abstractmethod
    def offload_model_parts(self, **kwargs) -> bool:
        """
        Offload specific model parts to disk based on predictive algorithms.

        Args:
            **kwargs: Configuration parameters for offloading

        Returns:
            True if offloading was successful, False otherwise
        """
        pass

    @abstractmethod
    def predict_model_part_access(self, **kwargs) -> Dict[str, Any]:
        """
        Predict which model parts will be accessed based on input patterns.

        Args:
            **kwargs: Configuration parameters for prediction

        Returns:
            Dictionary mapping model part names to access probabilities
        """
        pass

    @abstractmethod
    def get_offloading_stats(self) -> Dict[str, Any]:
        """
        Get statistics about disk offloading operations.

        Returns:
            Dictionary containing offloading statistics
        """
        pass