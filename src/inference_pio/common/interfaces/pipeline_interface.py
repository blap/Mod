"""
Interface for pipeline functionality in the Mod project.

This module defines a clear interface for pipeline operations
that can be implemented by different pipeline strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from .model_surgery_interface import ModelSurgeryManagerInterface


class PipelineManagerInterface(ABC):
    """
    Interface for pipeline operations.
    """

    @abstractmethod
    def setup_pipeline(self, **kwargs) -> bool:
        """
        Set up disk-based inference pipeline system for the plugin.

        Args:
            **kwargs: Pipeline configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def execute_pipeline(
        self, data: Any, pipeline_config: Dict[str, Any] = None
    ) -> Any:
        """
        Execute inference using the disk-based pipeline system.

        Args:
            data: Input data for inference
            pipeline_config: Configuration for the pipeline execution

        Returns:
            Inference results from the pipeline
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def create_pipeline_stages(self, **kwargs) -> List["PipelineStage"]:
        """
        Create pipeline stages for the model.

        Args:
            **kwargs: Stage configuration parameters

        Returns:
            List of PipelineStage objects
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_pipeline_manager(self):
        """
        Get the pipeline manager instance.

        Returns:
            Pipeline manager instance or None
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the pipeline execution.

        Returns:
            Dictionary containing pipeline statistics
        """
        pass