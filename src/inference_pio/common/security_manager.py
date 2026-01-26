"""
Security and Resource Isolation Manager for Inference-PIO

This module implements security and resource isolation mechanisms for the Inference-PIO system,
ensuring that different plugins and models operate safely and independently from each other.
"""

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import psutil
import torch
from contextlib import contextmanager
import tempfile
import secrets
import hashlib
from datetime import datetime


logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different types of operations."""
    UNTRUSTED = "untrusted"
    LOW_TRUST = "low_trust"
    MEDIUM_TRUST = "medium_trust"
    HIGH_TRUST = "high_trust"
    TRUSTED = "trusted"


class ResourceLimitType(Enum):
    """Types of resource limits."""
    CPU_PERCENT = "cpu_percent"
    MEMORY_GB = "memory_gb"
    GPU_MEMORY_GB = "gpu_memory_gb"
    DISK_SPACE_GB = "disk_space_gb"
    NETWORK_BANDWIDTH = "network_bandwidth"
    FILE_HANDLES = "file_handles"
    PROCESSES = "processes"


@dataclass
class ResourceLimits:
    """Resource limits for a plugin or model."""
    cpu_percent: Optional[float] = None
    memory_gb: Optional[float] = None
    gpu_memory_gb: Optional[float] = None
    disk_space_gb: Optional[float] = None
    network_bandwidth: Optional[float] = None  # Mbps
    file_handles: Optional[int] = None
    processes: Optional[int] = None


@dataclass
class SecurityContext:
    """Security context for a plugin or model."""
    plugin_id: str
    security_level: SecurityLevel
    resource_limits: ResourceLimits
    allowed_paths: List[str]
    forbidden_paths: List[str]
    allowed_network_hosts: List[str]
    creation_time: datetime
    sandbox_directory: Optional[str] = None
    access_token: Optional[str] = None


class SecurityManager:
    """
    Main security manager that handles security contexts and resource isolation
    for different plugins and models in the Inference-PIO system.
    """

    def __init__(self):
        self.security_contexts: Dict[str, SecurityContext] = {}
        self.active_monitors: Dict[str, threading.Thread] = {}
        self.monitoring_active = False
        self.lock = threading.Lock()

    def create_security_context(
        self,
        plugin_id: str,
        security_level: SecurityLevel = SecurityLevel.MEDIUM_TRUST,
        resource_limits: Optional[ResourceLimits] = None,
        allowed_paths: Optional[List[str]] = None,
        forbidden_paths: Optional[List[str]] = None,
        allowed_network_hosts: Optional[List[str]] = None
    ) -> SecurityContext:
        """
        Create a security context for a plugin or model.

        Args:
            plugin_id: Unique identifier for the plugin/model
            security_level: Security level for the context
            resource_limits: Resource limits to enforce
            allowed_paths: Paths that are allowed to be accessed
            forbidden_paths: Paths that are forbidden to be accessed
            allowed_network_hosts: Network hosts that are allowed to connect to

        Returns:
            Created security context
        """
        if resource_limits is None:
            resource_limits = ResourceLimits(
                cpu_percent=80.0,
                memory_gb=8.0,
                gpu_memory_gb=4.0 if torch.cuda.is_available() else 0.0,
                disk_space_gb=10.0
            )

        if allowed_paths is None:
            allowed_paths = [str(Path.home()), "/tmp", "/var/tmp"]

        if forbidden_paths is None:
            forbidden_paths = ["/etc", "/root", "/proc", "/sys"]

        if allowed_network_hosts is None:
            allowed_network_hosts = ["localhost", "127.0.0.1"]

        # Create a unique access token for this context
        access_token = secrets.token_urlsafe(32)

        # Create a sandbox directory for this context
        sandbox_dir = tempfile.mkdtemp(prefix=f"pio_sandbox_{plugin_id}_")

        context = SecurityContext(
            plugin_id=plugin_id,
            security_level=security_level,
            resource_limits=resource_limits,
            allowed_paths=allowed_paths,
            forbidden_paths=forbidden_paths,
            allowed_network_hosts=allowed_network_hosts,
            creation_time=datetime.now(),
            sandbox_directory=sandbox_dir,
            access_token=access_token
        )

        with self.lock:
            self.security_contexts[plugin_id] = context

        logger.info(f"Created security context for plugin {plugin_id} with security level {security_level.value}")
        return context

    def validate_path_access(self, plugin_id: str, path: str) -> bool:
        """
        Validate if a plugin is allowed to access a specific path.

        Args:
            plugin_id: ID of the plugin requesting access
            path: Path to validate

        Returns:
            True if access is allowed, False otherwise
        """
        if plugin_id not in self.security_contexts:
            logger.warning(f"No security context found for plugin {plugin_id}")
            return False

        context = self.security_contexts[plugin_id]
        path_obj = Path(path).resolve()

        # Check forbidden paths first
        for forbidden_path in context.forbidden_paths:
            try:
                if path_obj.is_relative_to(Path(forbidden_path)):
                    logger.warning(f"Plugin {plugin_id} attempted to access forbidden path: {path}")
                    return False
            except ValueError:
                # Path is not relative to forbidden path, continue
                continue

        # Check allowed paths
        for allowed_path in context.allowed_paths:
            try:
                if path_obj.is_relative_to(Path(allowed_path)):
                    return True
            except ValueError:
                # Path is not relative to allowed path, continue
                continue

        logger.warning(f"Plugin {plugin_id} attempted to access unauthorized path: {path}")
        return False

    def validate_network_access(self, plugin_id: str, host: str) -> bool:
        """
        Validate if a plugin is allowed to connect to a specific network host.

        Args:
            plugin_id: ID of the plugin requesting access
            host: Host to validate

        Returns:
            True if access is allowed, False otherwise
        """
        if plugin_id not in self.security_contexts:
            logger.warning(f"No security context found for plugin {plugin_id}")
            return False

        context = self.security_contexts[plugin_id]

        # Check if host is in allowed list
        if host in context.allowed_network_hosts:
            return True

        # Check if host is localhost (always allowed)
        if host in ["localhost", "127.0.0.1", "::1"]:
            return True

        logger.warning(f"Plugin {plugin_id} attempted to connect to unauthorized host: {host}")
        return False

    def enforce_resource_limits(self, plugin_id: str) -> bool:
        """
        Enforce resource limits for a plugin.

        Args:
            plugin_id: ID of the plugin to enforce limits for

        Returns:
            True if limits are within bounds, False if exceeded
        """
        if plugin_id not in self.security_contexts:
            logger.warning(f"No security context found for plugin {plugin_id}")
            return False

        context = self.security_contexts[plugin_id]
        limits = context.resource_limits

        # Check CPU usage
        if limits.cpu_percent is not None:
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > limits.cpu_percent:
                logger.warning(f"Plugin {plugin_id} exceeded CPU limit: {cpu_percent}% > {limits.cpu_percent}%")
                return False

        # Check memory usage
        if limits.memory_gb is not None:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            memory_gb = memory_mb / 1024
            if memory_gb > limits.memory_gb:
                logger.warning(f"Plugin {plugin_id} exceeded memory limit: {memory_gb:.2f}GB > {limits.memory_gb}GB")
                return False

        # Check GPU memory usage
        if limits.gpu_memory_gb is not None and torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            if gpu_memory_allocated > limits.gpu_memory_gb:
                logger.warning(f"Plugin {plugin_id} exceeded GPU memory limit: {gpu_memory_allocated:.2f}GB > {limits.gpu_memory_gb}GB")
                return False

        # Check disk space usage in sandbox directory
        if context.sandbox_directory and limits.disk_space_gb is not None:
            try:
                disk_usage = self._get_directory_size(context.sandbox_directory)
                disk_usage_gb = disk_usage / (1024 ** 3)
                if disk_usage_gb > limits.disk_space_gb:
                    logger.warning(f"Plugin {plugin_id} exceeded disk space limit: {disk_usage_gb:.2f}GB > {limits.disk_space_gb}GB")
                    return False
            except Exception as e:
                logger.error(f"Error checking disk usage for plugin {plugin_id}: {e}")
                return False

        return True

    def _get_directory_size(self, directory: str) -> int:
        """Get the total size of a directory in bytes."""
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    # File might have been deleted or inaccessible
                    continue
        return total_size

    def start_resource_monitoring(self, plugin_id: str, interval: float = 1.0) -> bool:
        """
        Start monitoring resource usage for a plugin.

        Args:
            plugin_id: ID of the plugin to monitor
            interval: Monitoring interval in seconds

        Returns:
            True if monitoring started successfully, False otherwise
        """
        if plugin_id not in self.security_contexts:
            logger.warning(f"No security context found for plugin {plugin_id}")
            return False

        if plugin_id in self.active_monitors and self.active_monitors[plugin_id].is_alive():
            logger.warning(f"Monitoring already active for plugin {plugin_id}")
            return False

        def monitor_resources():
            while self.monitoring_active:
                try:
                    if not self.enforce_resource_limits(plugin_id):
                        logger.error(f"Resource limits exceeded for plugin {plugin_id}, taking action...")
                        # TODO: Implement resource limit violation response
                        # For now, just log the violation
                        pass
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Error in resource monitoring for plugin {plugin_id}: {e}")
                    break

        thread = threading.Thread(target=monitor_resources, daemon=True)
        thread.start()
        self.active_monitors[plugin_id] = thread

        logger.info(f"Started resource monitoring for plugin {plugin_id}")
        return True

    def stop_resource_monitoring(self, plugin_id: str) -> bool:
        """
        Stop monitoring resource usage for a plugin.

        Args:
            plugin_id: ID of the plugin to stop monitoring

        Returns:
            True if monitoring stopped successfully, False otherwise
        """
        if plugin_id not in self.active_monitors:
            logger.warning(f"No active monitor found for plugin {plugin_id}")
            return False

        # Thread is daemon, so it will stop when the main program exits
        # We just remove it from our tracking
        del self.active_monitors[plugin_id]

        logger.info(f"Stopped resource monitoring for plugin {plugin_id}")
        return True

    def cleanup_security_context(self, plugin_id: str) -> bool:
        """
        Clean up security context and associated resources.

        Args:
            plugin_id: ID of the plugin to clean up

        Returns:
            True if cleanup was successful, False otherwise
        """
        if plugin_id not in self.security_contexts:
            logger.warning(f"No security context found for plugin {plugin_id}")
            return False

        context = self.security_contexts[plugin_id]

        # Stop monitoring
        self.stop_resource_monitoring(plugin_id)

        # Remove sandbox directory
        if context.sandbox_directory:
            try:
                import shutil
                shutil.rmtree(context.sandbox_directory)
                logger.info(f"Removed sandbox directory for plugin {plugin_id}: {context.sandbox_directory}")
            except Exception as e:
                logger.error(f"Error removing sandbox directory for plugin {plugin_id}: {e}")

        # Remove security context
        del self.security_contexts[plugin_id]

        logger.info(f"Cleaned up security context for plugin {plugin_id}")
        return True

    def get_security_context(self, plugin_id: str) -> Optional[SecurityContext]:
        """
        Get the security context for a plugin.

        Args:
            plugin_id: ID of the plugin

        Returns:
            Security context if found, None otherwise
        """
        return self.security_contexts.get(plugin_id)

    def validate_access_token(self, plugin_id: str, token: str) -> bool:
        """
        Validate an access token for a plugin.

        Args:
            plugin_id: ID of the plugin
            token: Access token to validate

        Returns:
            True if token is valid, False otherwise
        """
        context = self.get_security_context(plugin_id)
        if not context:
            return False
        return context.access_token == token


class ResourceIsolationManager:
    """
    Manager for resource isolation between different plugins and models.
    """

    def __init__(self):
        self.security_manager = SecurityManager()
        self.plugin_resources: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def initialize_plugin_isolation(
        self,
        plugin_id: str,
        security_level: SecurityLevel = SecurityLevel.MEDIUM_TRUST,
        resource_limits: Optional[ResourceLimits] = None
    ) -> bool:
        """
        Initialize resource isolation for a plugin.

        Args:
            plugin_id: ID of the plugin
            security_level: Security level for the plugin
            resource_limits: Resource limits to enforce

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Create security context
            context = self.security_manager.create_security_context(
                plugin_id=plugin_id,
                security_level=security_level,
                resource_limits=resource_limits
            )

            # Initialize resource tracking for this plugin
            with self.lock:
                self.plugin_resources[plugin_id] = {
                    'context': context,
                    'active_operations': 0,
                    'start_time': time.time(),
                    'resources_used': {}
                }

            # Start resource monitoring
            self.security_manager.start_resource_monitoring(plugin_id)

            logger.info(f"Initialized resource isolation for plugin {plugin_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize resource isolation for plugin {plugin_id}: {e}")
            return False

    def begin_operation(self, plugin_id: str) -> Optional[str]:
        """
        Begin an operation for a plugin, returning an operation token.

        Args:
            plugin_id: ID of the plugin

        Returns:
            Operation token if successful, None otherwise
        """
        if plugin_id not in self.plugin_resources:
            logger.warning(f"No resource isolation initialized for plugin {plugin_id}")
            return None

        # Check resource limits before allowing operation
        if not self.security_manager.enforce_resource_limits(plugin_id):
            logger.error(f"Resource limits exceeded for plugin {plugin_id}, denying operation")
            return None

        # Generate operation token
        operation_token = secrets.token_urlsafe(16)

        with self.lock:
            self.plugin_resources[plugin_id]['active_operations'] += 1
            self.plugin_resources[plugin_id]['operation_tokens'] = \
                self.plugin_resources[plugin_id].get('operation_tokens', {})
            self.plugin_resources[plugin_id]['operation_tokens'][operation_token] = time.time()

        logger.debug(f"Started operation for plugin {plugin_id}, active operations: {self.plugin_resources[plugin_id]['active_operations']}")
        return operation_token

    def end_operation(self, plugin_id: str, operation_token: str) -> bool:
        """
        End an operation for a plugin.

        Args:
            plugin_id: ID of the plugin
            operation_token: Operation token returned by begin_operation

        Returns:
            True if operation ended successfully, False otherwise
        """
        if plugin_id not in self.plugin_resources:
            logger.warning(f"No resource isolation initialized for plugin {plugin_id}")
            return False

        with self.lock:
            if 'operation_tokens' in self.plugin_resources[plugin_id]:
                if operation_token in self.plugin_resources[plugin_id]['operation_tokens']:
                    del self.plugin_resources[plugin_id]['operation_tokens'][operation_token]
                    self.plugin_resources[plugin_id]['active_operations'] -= 1
                    return True

        logger.warning(f"Invalid operation token for plugin {plugin_id}")
        return False

    def validate_path_access(self, plugin_id: str, path: str) -> bool:
        """
        Validate if a plugin can access a specific path.

        Args:
            plugin_id: ID of the plugin
            path: Path to validate

        Returns:
            True if access is allowed, False otherwise
        """
        return self.security_manager.validate_path_access(plugin_id, path)

    def validate_network_access(self, plugin_id: str, host: str) -> bool:
        """
        Validate if a plugin can connect to a specific network host.

        Args:
            plugin_id: ID of the plugin
            host: Host to validate

        Returns:
            True if access is allowed, False otherwise
        """
        return self.security_manager.validate_network_access(plugin_id, host)

    def cleanup_plugin_isolation(self, plugin_id: str) -> bool:
        """
        Clean up resource isolation for a plugin.

        Args:
            plugin_id: ID of the plugin

        Returns:
            True if cleanup was successful, False otherwise
        """
        # Clean up active operations
        if plugin_id in self.plugin_resources:
            with self.lock:
                ops = self.plugin_resources[plugin_id].get('operation_tokens', {})
                for token in list(ops.keys()):
                    self.end_operation(plugin_id, token)

        # Clean up security context
        success = self.security_manager.cleanup_security_context(plugin_id)

        # Remove from tracking
        if plugin_id in self.plugin_resources:
            with self.lock:
                del self.plugin_resources[plugin_id]

        logger.info(f"Cleaned up resource isolation for plugin {plugin_id}")
        return success

    def get_plugin_resource_usage(self, plugin_id: str) -> Dict[str, Any]:
        """
        Get resource usage information for a plugin.

        Args:
            plugin_id: ID of the plugin

        Returns:
            Dictionary with resource usage information
        """
        if plugin_id not in self.plugin_resources:
            return {}

        context = self.security_manager.get_security_context(plugin_id)
        if not context:
            return {}

        usage_info = {
            'plugin_id': plugin_id,
            'security_level': context.security_level.value,
            'active_operations': self.plugin_resources[plugin_id]['active_operations'],
            'uptime_seconds': time.time() - self.plugin_resources[plugin_id]['start_time'],
            'sandbox_directory': context.sandbox_directory
        }

        # Add system resource usage
        try:
            process = psutil.Process()
            usage_info['memory_usage_mb'] = process.memory_info().rss / (1024 * 1024)
            usage_info['cpu_percent'] = process.cpu_percent()
            
            if torch.cuda.is_available():
                usage_info['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024 ** 3)
                usage_info['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024 ** 3)
        except Exception as e:
            logger.error(f"Error getting resource usage for plugin {plugin_id}: {e}")

        return usage_info


# Global resource isolation manager instance
_global_resource_isolation_manager: Optional[ResourceIsolationManager] = None


def get_resource_isolation_manager() -> ResourceIsolationManager:
    """
    Get the global resource isolation manager instance.

    Returns:
        ResourceIsolationManager instance
    """
    global _global_resource_isolation_manager
    if _global_resource_isolation_manager is None:
        _global_resource_isolation_manager = ResourceIsolationManager()
    return _global_resource_isolation_manager


def initialize_plugin_isolation(
    plugin_id: str,
    security_level: SecurityLevel = SecurityLevel.MEDIUM_TRUST,
    resource_limits: Optional[ResourceLimits] = None
) -> bool:
    """
    Initialize resource isolation for a plugin using the global manager.

    Args:
        plugin_id: ID of the plugin
        security_level: Security level for the plugin
        resource_limits: Resource limits to enforce

    Returns:
        True if initialization was successful, False otherwise
    """
    manager = get_resource_isolation_manager()
    return manager.initialize_plugin_isolation(plugin_id, security_level, resource_limits)


def validate_path_access(plugin_id: str, path: str) -> bool:
    """
    Validate if a plugin can access a specific path using the global manager.

    Args:
        plugin_id: ID of the plugin
        path: Path to validate

    Returns:
        True if access is allowed, False otherwise
    """
    manager = get_resource_isolation_manager()
    return manager.validate_path_access(plugin_id, path)


def validate_network_access(plugin_id: str, host: str) -> bool:
    """
    Validate if a plugin can connect to a specific network host using the global manager.

    Args:
        plugin_id: ID of the plugin
        host: Host to validate

    Returns:
        True if access is allowed, False otherwise
    """
    manager = get_resource_isolation_manager()
    return manager.validate_network_access(plugin_id, host)


def cleanup_plugin_isolation(plugin_id: str) -> bool:
    """
    Clean up resource isolation for a plugin using the global manager.

    Args:
        plugin_id: ID of the plugin

    Returns:
        True if cleanup was successful, False otherwise
    """
    manager = get_resource_isolation_manager()
    return manager.cleanup_plugin_isolation(plugin_id)


def get_plugin_resource_usage(plugin_id: str) -> Dict[str, Any]:
    """
    Get resource usage information for a plugin using the global manager.

    Args:
        plugin_id: ID of the plugin

    Returns:
        Dictionary with resource usage information
    """
    manager = get_resource_isolation_manager()
    return manager.get_plugin_resource_usage(plugin_id)


__all__ = [
    "SecurityManager",
    "ResourceIsolationManager",
    "SecurityLevel",
    "ResourceLimitType",
    "ResourceLimits",
    "get_resource_isolation_manager",
    "initialize_plugin_isolation",
    "validate_path_access",
    "validate_network_access",
    "cleanup_plugin_isolation",
    "get_plugin_resource_usage"
]