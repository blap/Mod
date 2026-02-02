"""
Security Management Component for Inference-PIO System

This module contains the security-related functionality extracted from the base plugin interface
to reduce the size of the main interface file and improve modularity.
"""

import logging
from typing import Any, Dict, Optional

import torch

from .security_manager import (
    ResourceLimits,
    SecurityLevel,
    cleanup_plugin_isolation,
    initialize_plugin_isolation,
)

logger = logging.getLogger(__name__)


class SecurityManagementMixin:
    """
    Mixin class that provides security management functionality to plugin interfaces.
    """

    def __init__(self):
        # Security and isolation attributes
        self._security_initialized = False
        self._security_level = SecurityLevel.MEDIUM_TRUST
        self._resource_limits = None

    def initialize_security(self, **kwargs) -> bool:
        """
        Initialize security and resource isolation for the plugin.

        Args:
            **kwargs: Security configuration parameters

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Get security parameters from kwargs
            security_level = kwargs.get("security_level", SecurityLevel.MEDIUM_TRUST)
            resource_limits = kwargs.get("resource_limits")

            # Initialize resource isolation
            success = initialize_plugin_isolation(
                plugin_id=(
                    self.metadata.name if hasattr(self, "metadata") else "unknown"
                ),
                security_level=security_level,
                resource_limits=resource_limits,
            )

            if success:
                self._security_initialized = True
                self._security_level = security_level
                self._resource_limits = resource_limits
                logger.info(
                    f"Security initialized for plugin with level {security_level.value}"
                )
            else:
                logger.error("Failed to initialize security for plugin")

            return success
        except Exception as e:
            logger.error(f"Error initializing security: {e}")
            return False

    def validate_file_access(self, file_path: str) -> bool:
        """
        Validate if the plugin is allowed to access a specific file path.

        Args:
            file_path: Path to the file to access

        Returns:
            True if access is allowed, False otherwise
        """
        if not self._security_initialized:
            logger.warning("Security not initialized, allowing access by default")
            return True

        from .security_manager import validate_path_access

        return validate_path_access(
            self.metadata.name if hasattr(self, "metadata") else "unknown", file_path
        )

    def validate_network_access(self, host: str) -> bool:
        """
        Validate if the plugin is allowed to connect to a specific network host.

        Args:
            host: Host to connect to

        Returns:
            True if access is allowed, False otherwise
        """
        if not self._security_initialized:
            logger.warning("Security not initialized, allowing access by default")
            return True

        from .security_manager import validate_network_access

        return validate_network_access(
            self.metadata.name if hasattr(self, "metadata") else "unknown", host
        )

    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get resource usage information for the plugin.

        Returns:
            Dictionary with resource usage information
        """
        if not self._security_initialized:
            logger.warning("Security not initialized")
            return {}

        from .security_manager import get_plugin_resource_usage

        return get_plugin_resource_usage(
            self.metadata.name if hasattr(self, "metadata") else "unknown"
        )

    def cleanup_security(self) -> bool:
        """
        Clean up security and resource isolation for the plugin.

        Returns:
            True if cleanup was successful, False otherwise
        """
        if not self._security_initialized:
            logger.info("Security not initialized, nothing to clean up")
            return True

        success = cleanup_plugin_isolation(
            self.metadata.name if hasattr(self, "metadata") else "unknown"
        )

        if success:
            self._security_initialized = False
            logger.info("Security cleaned up for plugin")
        else:
            logger.error("Failed to clean up security for plugin")

        return success


__all__ = ["SecurityManagementMixin"]
