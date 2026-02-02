"""
Model Surgery Component for Inference-PIO System

This module contains the model surgery functionality extracted from the base plugin interface
to reduce the size of the main interface file and improve modularity.
"""

import logging
from typing import Any, Dict, List, Optional

import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelSurgeryMixin:
    """
    Mixin class that provides model surgery functionality to plugin interfaces.
    """

    def __init__(self):
        # Model surgery attributes
        self._model_surgery_system = None
        self._surgery_config = {}

    def setup_model_surgery(self, **kwargs) -> bool:
        """
        Set up model surgery system for identifying and removing non-essential components.

        Args:
            **kwargs: Model surgery configuration parameters

        Returns:
            True if setup was successful, False otherwise
        """
        try:
            # Import the model surgery system
            from .model_surgery import ModelSurgerySystem

            self._model_surgery_system = ModelSurgerySystem()

            # Store surgery configuration
            self._surgery_config = {
                "enabled": kwargs.get("surgery_enabled", True),
                "auto_identify": kwargs.get("auto_identify_components", True),
                "preserve_components": kwargs.get("preserve_components", []),
                "surgery_priority_threshold": kwargs.get(
                    "surgery_priority_threshold", 10
                ),
                "analysis_only": kwargs.get("analysis_only", False),
            }

            logger.info("Model surgery system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to setup model surgery: {e}")
            return False

    def enable_model_surgery(self, **kwargs) -> bool:
        """
        Enable model surgery for the plugin to identify and temporarily remove
        non-essential components during inference.

        Args:
            **kwargs: Model surgery configuration parameters

        Returns:
            True if model surgery was enabled successfully, False otherwise
        """
        try:
            if not self._model_surgery_system:
                if not self.setup_model_surgery(**kwargs):
                    logger.error("Failed to setup model surgery system")
                    return False

            # Update surgery configuration with new parameters
            for key, value in kwargs.items():
                if hasattr(self, "_surgery_config"):
                    self._surgery_config[key] = value

            logger.info("Model surgery enabled")
            return True
        except Exception as e:
            logger.error(f"Failed to enable model surgery: {e}")
            return False

    def perform_model_surgery(
        self,
        model: nn.Module = None,
        components_to_remove: Optional[List[str]] = None,
        preserve_components: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Perform model surgery by identifying and removing non-essential components.

        Args:
            model: Model to perform surgery on (if None, uses self._model if available)
            components_to_remove: Specific components to remove (None means auto-detect)
            preserve_components: Components to preserve even if they're removable

        Returns:
            Modified model with surgery applied
        """
        try:
            if not self._model_surgery_system:
                logger.error("Model surgery system not initialized")
                return model or getattr(self, "_model", None)

            # Determine which model to operate on
            target_model = model
            if target_model is None:
                if hasattr(self, "_model") and self._model is not None:
                    target_model = self._model
                else:
                    logger.warning("No model provided and no internal model found")
                    return model

            # Use config values or fallback to parameters
            if preserve_components is None:
                preserve_components = self._surgery_config.get(
                    "preserve_components", []
                )

            # Perform the surgery
            from .model_surgery import apply_model_surgery

            modified_model = apply_model_surgery(
                target_model,
                components_to_remove=components_to_remove,
                preserve_components=preserve_components,
            )

            logger.info("Model surgery performed successfully")
            return modified_model
        except Exception as e:
            logger.error(f"Failed to perform model surgery: {e}")
            # Return original model on failure
            return model or getattr(self, "_model", None)

    def restore_model_from_surgery(
        self, model: nn.Module = None, surgery_id: Optional[str] = None
    ) -> nn.Module:
        """
        Restore a model from surgery by putting back removed components.

        Args:
            model: Model to restore (if None, uses self._model)
            surgery_id: Specific surgery to reverse (None means reverse latest)

        Returns:
            Restored model
        """
        try:
            if not self._model_surgery_system:
                logger.error("Model surgery system not initialized")
                return model or getattr(self, "_model", None)

            # Determine which model to operate on
            target_model = model
            if target_model is None:
                if hasattr(self, "_model") and self._model is not None:
                    target_model = self._model
                else:
                    logger.warning("No model provided and no internal model found")
                    return model

            # Perform restoration
            from .model_surgery import restore_model_from_surgery

            restored_model = restore_model_from_surgery(target_model, surgery_id)

            logger.info("Model restoration from surgery completed successfully")
            return restored_model
        except Exception as e:
            logger.error(f"Failed to restore model from surgery: {e}")
            # Return original model on failure
            return model or getattr(self, "_model", None)

    def analyze_model_for_surgery(self, model: nn.Module = None) -> Dict[str, Any]:
        """
        Analyze a model to identify potential candidates for surgical removal.

        Args:
            model: Model to analyze (if None, uses self._model)

        Returns:
            Dictionary containing analysis results
        """
        try:
            if not self._model_surgery_system:
                logger.error("Model surgery system not initialized")
                return {}

            # Determine which model to operate on
            target_model = model
            if target_model is None:
                if hasattr(self, "_model") and self._model is not None:
                    target_model = self._model
                else:
                    logger.warning("No model provided and no internal model found")
                    return {}

            # Perform analysis
            analysis = self._model_surgery_system.analyze_model_for_surgery(
                target_model
            )

            logger.info("Model analysis for surgery completed")
            return analysis
        except Exception as e:
            logger.error(f"Failed to analyze model for surgery: {e}")
            return {}

    def get_surgery_stats(self) -> Dict[str, Any]:
        """
        Get statistics about performed model surgeries.

        Returns:
            Dictionary containing surgery statistics
        """
        try:
            if not self._model_surgery_system:
                logger.error("Model surgery system not initialized")
                return {}

            return self._model_surgery_system.get_surgery_stats()
        except Exception as e:
            logger.error(f"Failed to get surgery stats: {e}")
            return {}


__all__ = ["ModelSurgeryMixin"]
