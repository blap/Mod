"""
Plugin System for Inference-PIO

This module implements the plugin system for the Inference-PIO framework.
"""

import importlib
import importlib.util
import inspect
import logging
import os
import pkgutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from ..common.base_plugin_interface import ModelPluginInterface
from ..common.security_manager import SecurityLevel, ResourceLimits, initialize_plugin_isolation, cleanup_plugin_isolation


logger = logging.getLogger(__name__)


class PluginManager:
    """
    Plugin manager for the Inference-PIO system.
    """

    def __init__(self):
        self.plugins: Dict[str, ModelPluginInterface] = {}
        self.active_plugins: Dict[str, ModelPluginInterface] = {}
        self.plugin_paths: List[Path] = []
        self.security_enabled = True  # Flag to enable/disable security features
        
    def register_plugin(self, plugin: ModelPluginInterface, name: Optional[str] = None) -> bool:
        """
        Register a plugin with the manager.

        Args:
            plugin: The plugin to register
            name: Optional name for the plugin (defaults to metadata.name)

        Returns:
            True if registration was successful, False otherwise
        """
        try:
            plugin_name = name or plugin.metadata.name
            
            if plugin_name in self.plugins:
                logger.warning(f"Plugin {plugin_name} already registered, overwriting")
            
            self.plugins[plugin_name] = plugin
            logger.info(f"Registered plugin: {plugin_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register plugin: {e}")
            return False
    
    def load_plugin_from_path(self, plugin_path: Union[str, Path]) -> bool:
        """
        Load a plugin from a file path.

        Args:
            plugin_path: Path to the plugin file

        Returns:
            True if loading was successful, False otherwise
        """
        try:
            plugin_path = Path(plugin_path)
            
            if not plugin_path.exists() or not plugin_path.suffix == ".py":
                logger.error(f"Invalid plugin file: {plugin_path}")
                return False
            
            # Add the plugin directory to Python path if not already there
            plugin_dir = str(plugin_path.parent)
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            # Import the module
            module_name = plugin_path.stem
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, ModelPluginInterface) 
                    and obj != ModelPluginInterface 
                    and obj.__module__ == module.__name__
                ):
                    try:
                        # Try to instantiate the plugin
                        # Most plugins will have a factory function, but if not, try direct instantiation
                        if hasattr(module, f"create_{obj.__name__.lower()}"):
                            factory_func = getattr(module, f"create_{obj.__name__.lower()}")
                            plugin_instance = factory_func()
                        else:
                            # If no factory function, try to instantiate directly
                            # This assumes the plugin class has a default constructor or accepts metadata
                            plugin_instance = obj()
                        
                        # Register the plugin
                        self.register_plugin(plugin_instance)
                        logger.info(f"Loaded and registered plugin from {plugin_path}: {obj.__name__}")
                    except Exception as e:
                        logger.error(f"Failed to instantiate plugin {obj.__name__}: {e}")
                        continue
            
            return True
        except Exception as e:
            logger.error(f"Failed to load plugin from path {plugin_path}: {e}")
            return False
    
    def load_plugins_from_directory(self, directory: Union[str, Path]) -> int:
        """
        Load all plugins from a directory.

        Args:
            directory: Directory to load plugins from

        Returns:
            Number of plugins successfully loaded
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Invalid plugin directory: {directory}")
            return 0
        
        loaded_count = 0
        for plugin_file in directory.glob("*.py"):
            if self.load_plugin_from_path(plugin_file):
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} plugins from directory: {directory}")
        return loaded_count
    
    def get_plugin(self, name: str) -> Optional[ModelPluginInterface]:
        """
        Get a plugin by name.

        Args:
            name: Name of the plugin

        Returns:
            Plugin instance if found, None otherwise
        """
        return self.plugins.get(name)
    
    def activate_plugin(self, name: str, **kwargs) -> bool:
        """
        Activate a plugin by name.

        Args:
            name: Name of the plugin to activate
            **kwargs: Additional parameters for activation

        Returns:
            True if activation was successful, False otherwise
        """
        if name not in self.plugins:
            logger.error(f"Plugin {name} not found")
            return False

        plugin = self.plugins[name]

        if name in self.active_plugins:
            logger.info(f"Plugin {name} already active")
            return True

        try:
            # Initialize security for the plugin if security is enabled
            if self.security_enabled:
                security_level = kwargs.get('security_level', SecurityLevel.MEDIUM_TRUST)
                resource_limits = kwargs.get('resource_limits',
                                           ResourceLimits(
                                               cpu_percent=80.0,
                                               memory_gb=8.0,
                                               gpu_memory_gb=4.0 if torch.cuda.is_available() else 0.0,
                                               disk_space_gb=10.0
                                           ))

                if hasattr(plugin, 'initialize_security'):
                    if not plugin.initialize_security(security_level=security_level, resource_limits=resource_limits):
                        logger.error(f"Failed to initialize security for plugin {name}")
                        return False
                else:
                    logger.warning(f"Plugin {name} does not have security initialization method")

            # Initialize the plugin if not already initialized
            if not plugin._initialized:
                if not plugin.initialize(**kwargs):
                    logger.error(f"Failed to initialize plugin {name}")
                    return False

            # Activate the plugin
            self.active_plugins[name] = plugin
            logger.info(f"Activated plugin: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to activate plugin {name}: {e}")
            return False
    
    def deactivate_plugin(self, name: str) -> bool:
        """
        Deactivate a plugin by name.

        Args:
            name: Name of the plugin to deactivate

        Returns:
            True if deactivation was successful, False otherwise
        """
        if name not in self.active_plugins:
            logger.info(f"Plugin {name} not active")
            return True

        try:
            plugin = self.active_plugins[name]

            # Perform cleanup if needed
            if hasattr(plugin, 'cleanup'):
                plugin.cleanup()

            # Perform security cleanup if security is enabled
            if self.security_enabled and hasattr(plugin, 'cleanup_security'):
                plugin.cleanup_security()

            # Remove from active plugins
            del self.active_plugins[name]
            logger.info(f"Deactivated plugin: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to deactivate plugin {name}: {e}")
            return False
    
    def execute_plugin(self, name: str, *args, **kwargs) -> Any:
        """
        Execute a plugin's main functionality.

        Args:
            name: Name of the plugin to execute
            *args: Arguments to pass to the plugin
            **kwargs: Keyword arguments to pass to the plugin

        Returns:
            Result of plugin execution
        """
        if name not in self.active_plugins:
            logger.error(f"Plugin {name} not active")
            return None

        plugin = self.active_plugins[name]

        try:
            # Validate resource limits before execution if security is enabled
            if self.security_enabled and hasattr(plugin, 'get_resource_usage'):
                resource_usage = plugin.get_resource_usage()
                # Log resource usage for monitoring
                logger.debug(f"Plugin {name} resource usage: {resource_usage}")

            # Call the appropriate method based on plugin type
            if hasattr(plugin, 'infer'):
                return plugin.infer(*args, **kwargs)
            elif hasattr(plugin, 'generate_text'):
                return plugin.generate_text(*args, **kwargs)
            elif hasattr(plugin, 'load_model'):
                return plugin.load_model(*args, **kwargs)
            else:
                logger.error(f"Plugin {name} does not have an executable method")
                return None
        except Exception as e:
            logger.error(f"Failed to execute plugin {name}: {e}")
            return None
    
    def list_plugins(self) -> List[str]:
        """
        List all registered plugins.

        Returns:
            List of plugin names
        """
        return list(self.plugins.keys())
    
    def list_active_plugins(self) -> List[str]:
        """
        List all active plugins.

        Returns:
            List of active plugin names
        """
        return list(self.active_plugins.keys())

    def discover_and_load_plugins(self, models_directory: Optional[Union[str, Path]] = None) -> int:
        """
        Automatically discover and load all available plugins from the models directory.

        Args:
            models_directory: Directory containing model plugins (defaults to models directory)

        Returns:
            Number of plugins successfully loaded
        """
        if models_directory is None:
            # Try to find the models directory relative to this file
            current_file_dir = Path(__file__).parent.parent
            models_directory = current_file_dir / "models"

        models_directory = Path(models_directory)

        if not models_directory.exists() or not models_directory.is_dir():
            logger.error(f"Models directory does not exist: {models_directory}")
            return 0

        loaded_count = 0

        # Iterate through each subdirectory in the models directory
        for model_dir in models_directory.iterdir():
            if model_dir.is_dir():
                # Look for plugin.py in the model directory
                plugin_file = model_dir / "plugin.py"
                if plugin_file.exists():
                    if self.load_plugin_from_path(plugin_file):
                        loaded_count += 1
                    else:
                        logger.warning(f"Failed to load plugin from {plugin_file}")

                # Also look for __init__.py which might contain plugin definitions
                init_file = model_dir / "__init__.py"
                if init_file.exists():
                    # Try to import the module and find plugins
                    try:
                        module_name = f"inference_pio.models.{model_dir.name}"
                        spec = importlib.util.spec_from_file_location(module_name, init_file)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        # Look for plugin classes in the module
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (
                                issubclass(obj, ModelPluginInterface)
                                and obj != ModelPluginInterface
                                and obj.__module__ == module.__name__
                            ):
                                try:
                                    # Try to find a factory function for the plugin
                                    factory_func_name = f"create_{obj.__name__.lower()}"
                                    if hasattr(module, factory_func_name):
                                        factory_func = getattr(module, factory_func_name)
                                        plugin_instance = factory_func()
                                        self.register_plugin(plugin_instance)
                                        logger.info(f"Discovered and registered plugin from {init_file}: {obj.__name__}")
                                        loaded_count += 1
                                    else:
                                        # Try to instantiate directly if no factory function exists
                                        plugin_instance = obj()
                                        self.register_plugin(plugin_instance)
                                        logger.info(f"Discovered and registered plugin from {init_file}: {obj.__name__}")
                                        loaded_count += 1
                                except Exception as e:
                                    logger.error(f"Failed to instantiate plugin {obj.__name__}: {e}")
                                    continue
                    except Exception as e:
                        logger.error(f"Failed to import module from {init_file}: {e}")
                        continue

        logger.info(f"Auto-discovered and loaded {loaded_count} plugins from {models_directory}")
        return loaded_count


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """
    Get the global plugin manager instance.

    Returns:
        Plugin manager instance
    """
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def register_plugin(plugin: ModelPluginInterface, name: Optional[str] = None) -> bool:
    """
    Register a plugin with the global plugin manager.

    Args:
        plugin: The plugin to register
        name: Optional name for the plugin

    Returns:
        True if registration was successful, False otherwise
    """
    return get_plugin_manager().register_plugin(plugin, name)


def load_plugin_from_path(plugin_path: Union[str, Path]) -> bool:
    """
    Load a plugin from a file path using the global plugin manager.

    Args:
        plugin_path: Path to the plugin file

    Returns:
        True if loading was successful, False otherwise
    """
    return get_plugin_manager().load_plugin_from_path(plugin_path)


def load_plugins_from_directory(directory: Union[str, Path]) -> int:
    """
    Load all plugins from a directory using the global plugin manager.

    Args:
        directory: Directory to load plugins from

    Returns:
        Number of plugins successfully loaded
    """
    return get_plugin_manager().load_plugins_from_directory(directory)


def activate_plugin(name: str, **kwargs) -> bool:
    """
    Activate a plugin by name using the global plugin manager.

    Args:
        name: Name of the plugin to activate
        **kwargs: Additional parameters for activation

    Returns:
        True if activation was successful, False otherwise
    """
    return get_plugin_manager().activate_plugin(name, **kwargs)


def execute_plugin(name: str, *args, **kwargs) -> Any:
    """
    Execute a plugin's main functionality using the global plugin manager.

    Args:
        name: Name of the plugin to execute
        *args: Arguments to pass to the plugin
        **kwargs: Keyword arguments to pass to the plugin

    Returns:
        Result of plugin execution
    """
    return get_plugin_manager().execute_plugin(name, *args, **kwargs)


def discover_and_load_plugins(models_directory: Optional[Union[str, Path]] = None) -> int:
    """
    Automatically discover and load all available plugins from the models directory
    using the global plugin manager.

    Args:
        models_directory: Directory containing model plugins (defaults to models directory)

    Returns:
        Number of plugins successfully loaded
    """
    return get_plugin_manager().discover_and_load_plugins(models_directory)


__all__ = [
    "PluginManager",
    "get_plugin_manager",
    "register_plugin",
    "load_plugin_from_path",
    "load_plugins_from_directory",
    "activate_plugin",
    "execute_plugin",
    "discover_and_load_plugins"
]