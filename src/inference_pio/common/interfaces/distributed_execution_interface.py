"""
Interface for distributed execution functionality in the Mod project.

This module defines a clear interface for distributed execution operations
that can be implemented by different distributed execution strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class DistributedExecutionManagerInterface(ABC):
    """
    Interface for distributed execution operations.
    """

    @abstractmethod
    def setup_distributed_simulation(self, **kwargs) -> bool:
        """
        Set up distributed simulation system for multi-GPU execution simulation.

        Args:
            **kwargs: Distributed simulation configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        pass

    @abstractmethod
    def enable_distributed_execution(self, **kwargs) -> bool:
        """
        Enable distributed execution simulation on single or multiple GPUs.

        Args:
            **kwargs: Distributed execution configuration parameters

        Returns:
            True if distributed execution was enabled successfully, False otherwise
        """
        pass

    @abstractmethod
    def partition_model_for_distributed(
        self, num_partitions: int = 1, **kwargs
    ) -> bool:
        """
        Partition the model for distributed execution.

        Args:
            num_partitions: Number of partitions to create
            **kwargs: Additional partitioning parameters

        Returns:
            True if partitioning was successful, False otherwise
        """
        pass

    @abstractmethod
    def get_virtual_execution_manager(self):
        """
        Get the virtual execution manager instance.

        Returns:
            Virtual execution manager instance or None
        """
        pass

    @abstractmethod
    def get_virtual_device_simulator(self):
        """
        Get the virtual device simulator instance.

        Returns:
            Virtual device simulator instance or None
        """
        pass

    @abstractmethod
    def execute_with_virtual_execution(self, data: Any) -> Any:
        """
        Execute inference using virtual execution (distributed simulation).

        Args:
            data: Input data for inference

        Returns:
            Inference results
        """
        pass

    @abstractmethod
    def get_virtual_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about virtual execution.

        Returns:
            Dictionary containing virtual execution statistics
        """
        pass

    @abstractmethod
    def synchronize_partitions(self) -> bool:
        """
        Synchronize all model partitions.

        Returns:
            True if synchronization was successful, False otherwise
        """
        pass

    @abstractmethod
    def pipeline_synchronize(self, current_stage: int, num_stages: int) -> bool:
        """
        Synchronize partitions in a pipeline fashion.

        Args:
            current_stage: Current pipeline stage
            num_stages: Total number of pipeline stages

        Returns:
            True if synchronization was successful, False otherwise
        """
        pass

    @abstractmethod
    def get_synchronization_manager(self):
        """
        Get the synchronization manager instance.

        Returns:
            Synchronization manager instance or None
        """
        pass