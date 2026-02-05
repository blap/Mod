"""
Interface for adaptive batching functionality in the Mod project.

This module defines a clear interface for adaptive batching operations
that can be implemented by different batching strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional


class AdaptiveBatchingManagerInterface(ABC):
    """
    Interface for adaptive batching operations.
    """

    @abstractmethod
    def setup_adaptive_batching(self, **kwargs) -> bool:
        """
        Set up adaptive batching system for dynamic batch size adjustment.

        Args:
            **kwargs: Adaptive batching configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_optimal_batch_size(
        self, processing_time_ms: float, tokens_processed: int
    ) -> int:
        """
        Get the optimal batch size based on current performance metrics.

        Args:
            processing_time_ms: Processing time for the current batch in milliseconds
            tokens_processed: Number of tokens processed in the current batch

        Returns:
            Recommended batch size for the next batch
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def adjust_batch_size(self) -> Tuple[int, bool, Optional[str]]:
        """
        Adjust the batch size based on current metrics.

        Returns:
            Tuple of (new_batch_size, was_adjusted, reason_for_adjustment)
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_batching_status(self) -> Dict[str, Any]:
        """
        Get the current status of the adaptive batching system.

        Returns:
            Dictionary containing batching status information
        """
        pass