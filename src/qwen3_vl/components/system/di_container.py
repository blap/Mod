"""
Dependency injection container for Qwen3-VL model components.

This module provides a container for managing dependencies between
different components of the Qwen3-VL model with proper separation of concerns.
"""
from typing import Dict, Type, Any, Optional
from dataclasses import dataclass


@dataclass
class ComponentRegistry:
    """
    Registry for component classes and their dependencies.
    """
    name: str
    cls: Type
    dependencies: list
    singleton: bool = True


class DIContainer:
    """
    Dependency injection container for managing Qwen3-VL model components.
    """

    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._registries: Dict[str, ComponentRegistry] = {}
        self._config = None  # Will be set later

    def register(self, name: str, cls: Type, dependencies: list = None, singleton: bool = True):
        """
        Register a component with its dependencies.

        Args:
            name: Name of the component
            cls: Class of the component
            dependencies: List of dependency names
            singleton: Whether to create a singleton instance
        """
        if dependencies is None:
            dependencies = []

        registry = ComponentRegistry(name, cls, dependencies, singleton)
        self._registries[name] = registry

    def register_config(self, config):
        """
        Register the main configuration.

        Args:
            config: Configuration instance
        """
        self._config = config
        # Register configuration components as well
        self._components['main_config'] = config
        self._components['memory_config'] = getattr(config, 'memory_config', None)
        self._components['attention_config'] = getattr(config, 'attention_config', None)
        self._components['routing_config'] = getattr(config, 'routing_config', None)
        self._components['hardware_config'] = getattr(config, 'hardware_config', None)

    def get(self, name: str):
        """
        Get a component by name, creating it if necessary.

        Args:
            name: Name of the component to retrieve

        Returns:
            The requested component instance
        """
        if name in self._components:
            return self._components[name]

        if name not in self._registries:
            raise ValueError(f"Component '{name}' is not registered")

        registry = self._registries[name]

        # Resolve dependencies recursively
        resolved_deps = {}
        for dep_name in registry.dependencies:
            resolved_deps[dep_name] = self.get(dep_name)

        # Create the component instance with resolved dependencies
        if name == "memory_manager":
            # Import here to avoid circular import
            from ..memory_management.manager import MemoryManager
            # Create memory manager with memory config
            instance = MemoryManager(self._config.memory_config)
        elif name == "memory_config":
            instance = getattr(self._config, 'memory_config', None)
        elif name == "attention_config":
            instance = getattr(self._config, 'attention_config', None)
        elif name == "routing_config":
            instance = getattr(self._config, 'routing_config', None)
        elif name == "hardware_config":
            instance = getattr(self._config, 'hardware_config', None)
        else:
            # For other components, try to instantiate with resolved dependencies
            try:
                instance = registry.cls(**resolved_deps)
            except TypeError:
                # If direct instantiation fails, try with config
                instance = registry.cls(self._config)

        # Store singleton if requested
        if registry.singleton:
            self._components[name] = instance

        return instance

    def build_default_registry(self):
        """
        Build the default registry with common Qwen3-VL components.
        """
        # Import here to avoid circular import
        from ...config.memory_config import MemoryConfig
        from ...config.attention_config import AttentionConfig
        from ...config.routing_config import RoutingConfig
        from ...config.hardware_config import HardwareConfig

        # Register configuration components
        self.register('memory_config', MemoryConfig, singleton=True)
        self.register('attention_config', AttentionConfig, singleton=True)
        self.register('routing_config', RoutingConfig, singleton=True)
        self.register('hardware_config', HardwareConfig, singleton=True)

        # Register core components - these are handled specially in the get method
        # so we don't need to register their classes directly to avoid circular imports
        self.register('memory_manager', lambda: None, [], singleton=True)  # Placeholder


def create_default_container(config) -> DIContainer:
    """
    Create a default dependency injection container with common components.

    Args:
        config: Qwen3VLConfig instance

    Returns:
        Configured DIContainer instance
    """
    container = DIContainer()
    container.register_config(config)
    container.build_default_registry()

    return container


def setup_qwen3_vl_system(config=None):
    """
    Set up the Qwen3-VL system with dependency injection.

    Args:
        config: Qwen3VLConfig instance (if None, creates default config)

    Returns:
        Configured DIContainer instance
    """
    if config is None:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        config = Qwen3VLConfig()

    container = create_default_container(config)

    # Additional setup can be done here

    return container