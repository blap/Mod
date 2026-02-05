"""
Interface for security functionality in the Mod project.

This module defines a clear interface for security operations
that can be implemented by different security strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class SecurityManagerInterface(ABC):
    """
    Interface for security operations.
    """

    @abstractmethod
    def initialize_security(self, **kwargs) -> bool:
        """
        Initialize security and resource isolation for the plugin.

        Args:
            **kwargs: Security configuration parameters

        Returns:
            True if initialization was successful, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def validate_file_access(self, file_path: str) -> bool:
        """
        Validate if the plugin is allowed to access a specific file path.

        Args:
            file_path: Path to the file to access

        Returns:
            True if access is allowed, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def validate_network_access(self, host: str) -> bool:
        """
        Validate if the plugin is allowed to connect to a specific network host.

        Args:
            host: Host to connect to

        Returns:
            True if access is allowed, False otherwise
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def get_resource_usage(self) -> Dict[str, Any]:
        """
        Get resource usage information for the plugin.

        Returns:
            Dictionary with resource usage information
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def cleanup_security(self) -> bool:
        """
        Clean up security and resource isolation for the plugin.

        Returns:
            True if cleanup was successful, False otherwise
        """
        pass