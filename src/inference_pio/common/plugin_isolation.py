"""
Plugin Isolation Decorators for Inference-PIO

This module provides decorators to wrap plugin operations with security and resource isolation.
"""

import functools
import logging
from typing import Any, Callable, Optional
from .security_manager import (
    get_resource_isolation_manager,
    SecurityLevel,
    ResourceLimits,
    validate_path_access,
    validate_network_access
)


logger = logging.getLogger(__name__)


def plugin_isolation(
    plugin_id: str,
    security_level: SecurityLevel = SecurityLevel.MEDIUM_TRUST,
    resource_limits: Optional[ResourceLimits] = None,
    allowed_paths: Optional[list] = None,
    forbidden_paths: Optional[list] = None,
    allowed_network_hosts: Optional[list] = None
):
    """
    Decorator to wrap plugin operations with security and resource isolation.

    Args:
        plugin_id: Unique identifier for the plugin
        security_level: Security level for the operation
        resource_limits: Resource limits to enforce
        allowed_paths: Paths that are allowed to be accessed
        forbidden_paths: Paths that are forbidden to be accessed
        allowed_network_hosts: Network hosts that are allowed to connect to
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize resource isolation if not already done
            manager = get_resource_isolation_manager()
            if plugin_id not in manager.plugin_resources:
                manager.initialize_plugin_isolation(
                    plugin_id=plugin_id,
                    security_level=security_level,
                    resource_limits=resource_limits
                )

            # Begin operation
            operation_token = manager.begin_operation(plugin_id)
            if not operation_token:
                logger.error(f"Failed to begin operation for plugin {plugin_id}")
                raise RuntimeError(f"Resource limits exceeded for plugin {plugin_id}")

            try:
                # Execute the original function
                result = func(*args, **kwargs)
                
                # Validate any path or network access that occurred during execution
                # This is a simplified check - in a real implementation, you'd need
                # to intercept all file and network operations
                
                return result
            except Exception as e:
                logger.error(f"Error in isolated operation for plugin {plugin_id}: {e}")
                raise
            finally:
                # End operation
                manager.end_operation(plugin_id, operation_token)

        return wrapper
    return decorator


def validate_file_access(file_path: str, plugin_id: str) -> bool:
    """
    Validate if a plugin is allowed to access a specific file path.

    Args:
        file_path: Path to the file to access
        plugin_id: ID of the plugin requesting access

    Returns:
        True if access is allowed, False otherwise
    """
    return validate_path_access(plugin_id, file_path)


def validate_network_host(host: str, plugin_id: str) -> bool:
    """
    Validate if a plugin is allowed to connect to a specific network host.

    Args:
        host: Host to connect to
        plugin_id: ID of the plugin requesting access

    Returns:
        True if access is allowed, False otherwise
    """
    return validate_network_access(plugin_id, host)


def secure_file_operation(plugin_id: str):
    """
    Decorator to wrap file operations with path validation.

    Args:
        plugin_id: ID of the plugin performing the operation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if the first argument is a file path
            if args and isinstance(args[0], str):
                file_path = args[0]
                if not validate_file_access(file_path, plugin_id):
                    raise PermissionError(f"Plugin {plugin_id} not allowed to access path: {file_path}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def secure_network_operation(plugin_id: str):
    """
    Decorator to wrap network operations with host validation.

    Args:
        plugin_id: ID of the plugin performing the operation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if the first argument is a host
            if args and isinstance(args[0], str):
                host = args[0]
                if not validate_network_host(host, plugin_id):
                    raise PermissionError(f"Plugin {plugin_id} not allowed to connect to host: {host}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


__all__ = [
    "plugin_isolation",
    "validate_file_access",
    "validate_network_host",
    "secure_file_operation",
    "secure_network_operation"
]