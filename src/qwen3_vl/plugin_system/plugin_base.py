"""
Unified Plugin Base Class for the Qwen3-VL System

This module provides a consistent base class for all plugins in the system,
ensuring they have all required attributes including state management.
"""

import abc
import threading
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import time
from contextlib import contextmanager


class PluginState(Enum):
    """Enumeration of possible plugin states."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: str  # Using str instead of PluginType to avoid circular imports
    dependencies: List[str] = None  # List of plugin names this plugin depends on
    compatibility: List[str] = None  # List of compatible system versions
    config_schema: Optional[Dict[str, Any]] = None  # JSON schema for plugin configuration
    created_at: float = time.time()

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.compatibility is None:
            self.compatibility = []


class PluginConfig:
    """Configuration for a plugin."""
    def __init__(self, plugin_name: str, config_data: Optional[Dict[str, Any]] = None):
        self.plugin_name = plugin_name
        self.config_data = config_data or {}
        self.lock = threading.RLock()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        with self.lock:
            return self.config_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        with self.lock:
            self.config_data[key] = value

    def update(self, new_config: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        with self.lock:
            self.config_data.update(new_config)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        with self.lock:
            return self.config_data.copy()


class IPlugin(abc.ABC):
    """Base interface for all plugins."""

    @abc.abstractmethod
    def initialize(self, config: PluginConfig) -> bool:
        """Initialize the plugin with the given configuration."""
        pass

    @abc.abstractmethod
    def activate(self) -> bool:
        """Activate the plugin."""
        pass

    @abc.abstractmethod
    def deactivate(self) -> bool:
        """Deactivate the plugin."""
        pass

    @abc.abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass

    @abc.abstractmethod
    def get_config(self) -> PluginConfig:
        """Get plugin configuration."""
        pass

    @property
    @abc.abstractmethod
    def state(self) -> PluginState:
        """Get the current state of the plugin."""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Get the name of the plugin."""
        pass


class BasePlugin(IPlugin):
    """Base implementation for plugins with proper state management."""

    def __init__(self, metadata: PluginMetadata):
        self._metadata = metadata
        self._config = PluginConfig(metadata.name)
        self._state = PluginState.UNLOADED
        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._lock = threading.RLock()

    def initialize(self, config: PluginConfig) -> bool:
        """Initialize the plugin with the given configuration."""
        with self._lock:
            try:
                self._config = config
                self._state = PluginState.INITIALIZED
                self._logger.info(f"Plugin {self._metadata.name} initialized")
                return True
            except Exception as e:
                self._logger.error(f"Error initializing plugin {self._metadata.name}: {e}")
                self._state = PluginState.ERROR
                return False

    def activate(self) -> bool:
        """Activate the plugin."""
        with self._lock:
            try:
                if self._state in [PluginState.INITIALIZED, PluginState.INACTIVE]:
                    self._state = PluginState.ACTIVE
                    self._logger.info(f"Plugin {self._metadata.name} activated")
                    return True
                else:
                    self._logger.warning(f"Cannot activate plugin {self._metadata.name} in state {self._state}")
                    return False
            except Exception as e:
                self._logger.error(f"Error activating plugin {self._metadata.name}: {e}")
                self._state = PluginState.ERROR
                return False

    def deactivate(self) -> bool:
        """Deactivate the plugin."""
        with self._lock:
            try:
                if self._state == PluginState.ACTIVE:
                    self._state = PluginState.INACTIVE
                    self._logger.info(f"Plugin {self._metadata.name} deactivated")
                    return True
                else:
                    self._logger.warning(f"Cannot deactivate plugin {self._metadata.name} in state {self._state}")
                    return False
            except Exception as e:
                self._logger.error(f"Error deactivating plugin {self._metadata.name}: {e}")
                self._state = PluginState.ERROR
                return False

    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        return self._metadata

    def get_config(self) -> PluginConfig:
        """Get plugin configuration."""
        return self._config

    @property
    def state(self) -> PluginState:
        """Get the current state of the plugin."""
        with self._lock:
            return self._state

    @property
    def name(self) -> str:
        """Get the name of the plugin."""
        return self._metadata.name

    def _validate_config(self) -> bool:
        """Validate the current configuration against the schema."""
        schema = self._metadata.config_schema
        if not schema:
            return True  # No schema to validate against

        # Simple validation - check if required fields are present
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in self._config.config_data:
                self._logger.error(f"Missing required configuration field: {field}")
                return False

        return True


# Specific plugin interfaces
class IOptimizationPlugin(IPlugin):
    """Interface for optimization plugins."""

    @abc.abstractmethod
    def optimize(self, *args, **kwargs) -> Any:
        """Apply optimization to the given inputs."""
        pass


class IMemoryManagementPlugin(IPlugin):
    """Interface for memory management plugins."""

    @abc.abstractmethod
    def allocate_memory(self, size: int, hints: Optional[Dict[str, Any]] = None) -> Any:
        """Allocate memory with optional hints."""
        pass

    @abc.abstractmethod
    def free_memory(self, memory_handle: Any) -> bool:
        """Free allocated memory."""
        pass


class IAttentionPlugin(IPlugin):
    """Interface for attention mechanism plugins."""

    @abc.abstractmethod
    def compute_attention(self, query: Any, key: Any, value: Any, **kwargs) -> Any:
        """Compute attention with the plugin's implementation."""
        pass


class ICPUPlugin(IPlugin):
    """Interface for CPU optimization plugins."""

    @abc.abstractmethod
    def optimize_cpu_usage(self, *args, **kwargs) -> Any:
        """Optimize CPU usage."""
        pass


class IPowerManagementPlugin(IPlugin):
    """Interface for power management plugins."""

    @abc.abstractmethod
    def manage_power(self, *args, **kwargs) -> Any:
        """Manage power consumption."""
        pass


class IThermalManagementPlugin(IPlugin):
    """Interface for thermal management plugins."""

    @abc.abstractmethod
    def manage_thermal(self, *args, **kwargs) -> Any:
        """Manage thermal conditions."""
        pass


class ICacheManagementPlugin(IPlugin):
    """Interface for cache management plugins."""

    @abc.abstractmethod
    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abc.abstractmethod
    def put_in_cache(self, key: str, value: Any) -> bool:
        """Put value in cache."""
        pass


class IModelOptimizationPlugin(IPlugin):
    """Interface for model optimization plugins."""

    @abc.abstractmethod
    def optimize_model(self, model: Any, *args, **kwargs) -> Any:
        """Optimize the given model."""
        pass