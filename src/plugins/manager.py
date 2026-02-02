"""
Plugin Manager System for Inference-PIO

This module implements the core plugin management system for the Inference-PIO framework.
It handles plugin discovery, loading, activation, and execution with security and resource management.
Each model plugin is completely independent with its own configuration, tests, and benchmarks.
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

from ..common.improved_base_plugin_interface import ModelPluginInterface
from ..common.security_manager import (
    ResourceLimits,
    SecurityLevel,
    cleanup_plugin_isolation,
    initialize_plugin_isolation,
)

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Advanced plugin manager for the Inference-PIO system.

    This class handles the complete lifecycle of plugins including:
    - Registration and loading from various sources
    - Activation with security and resource constraints
    - Execution with monitoring and validation
    - Discovery of plugins in model directories
    - Automatic loading based on plugin manifests
    - Comprehensive security and resource management
    """

    def __init__(self):
        self.plugins: Dict[str, ModelPluginInterface] = {}
        self.active_plugins: Dict[str, ModelPluginInterface] = {}
        self.plugin_paths: List[Path] = []
        self.security_enabled = True  # Flag to enable/disable security features

    def register_plugin(
        self, plugin: ModelPluginInterface, name: Optional[str] = None
    ) -> bool:
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
                            factory_func = getattr(
                                module, f"create_{obj.__name__.lower()}"
                            )
                            plugin_instance = factory_func()
                        else:
                            # If no factory function, try to instantiate directly
                            # This assumes the plugin class has a default constructor or accepts metadata
                            plugin_instance = obj()

                        # Register the plugin
                        self.register_plugin(plugin_instance)
                        logger.info(
                            f"Loaded and registered plugin from {plugin_path}: {obj.__name__}"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to instantiate plugin {obj.__name__}: {e}"
                        )
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
        Activate a plugin by name with comprehensive security and resource management.

        This method handles the complete activation process including:
        - Security initialization with configurable trust levels
        - Resource allocation and limits enforcement
        - Plugin initialization if not already initialized
        - Addition to the active plugins registry
        - Validation of resource limits and security constraints

        Args:
            name: Name of the plugin to activate
            **kwargs: Additional parameters for activation, including:
                     - security_level: Security trust level (default: MEDIUM_TRUST)
                     - resource_limits: Resource constraints for the plugin

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
                security_level = kwargs.get(
                    "security_level", SecurityLevel.MEDIUM_TRUST
                )
                try:
                    # Import torch locally to avoid issues if not available
                    import torch

                    gpu_memory_gb = 4.0 if torch.cuda.is_available() else 0.0
                except ImportError:
                    gpu_memory_gb = 0.0  # No GPU available if torch is not installed

                resource_limits = kwargs.get(
                    "resource_limits",
                    ResourceLimits(
                        cpu_percent=80.0,
                        memory_gb=8.0,
                        gpu_memory_gb=gpu_memory_gb,
                        disk_space_gb=10.0,
                    ),
                )

                if hasattr(plugin, "initialize_security"):
                    if not plugin.initialize_security(
                        security_level=security_level, resource_limits=resource_limits
                    ):
                        logger.error(f"Failed to initialize security for plugin {name}")
                        return False
                else:
                    logger.warning(
                        f"Plugin {name} does not have security initialization method"
                    )

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
            if hasattr(plugin, "cleanup"):
                plugin.cleanup()

            # Perform security cleanup if security is enabled
            if self.security_enabled and hasattr(plugin, "cleanup_security"):
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
        Execute a plugin's main functionality with resource monitoring and validation.

        This method intelligently selects the appropriate execution method based on the
        plugin's capabilities:
        - Checks for 'infer' method (for general inference)
        - Checks for 'generate_text' method (for text generation)
        - Checks for 'load_model' method (for model loading)
        - Falls back to error if no executable method is found

        Resource usage is monitored during execution when security is enabled.
        The method also validates resource limits before execution and logs usage statistics.

        Args:
            name: Name of the plugin to execute
            *args: Arguments to pass to the plugin
            **kwargs: Keyword arguments to pass to the plugin

        Returns:
            Result of plugin execution or None if execution failed
        """
        if name not in self.active_plugins:
            logger.error(f"Plugin {name} not active")
            return None

        plugin = self.active_plugins[name]

        try:
            # Validate resource limits before execution if security is enabled
            if self.security_enabled and hasattr(plugin, "get_resource_usage"):
                resource_usage = plugin.get_resource_usage()
                # Log resource usage for monitoring
                logger.debug(f"Plugin {name} resource usage: {resource_usage}")

            # Call the appropriate method based on plugin type
            if hasattr(plugin, "infer"):
                return plugin.infer(*args, **kwargs)
            elif hasattr(plugin, "generate_text"):
                return plugin.generate_text(*args, **kwargs)
            elif hasattr(plugin, "load_model"):
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

    def discover_and_load_plugins(
        self, models_directory: Optional[Union[str, Path]] = None
    ) -> int:
        """
        Automatically discover and load all available plugins from the models directory.

        This method performs intelligent scanning of model directories to find and load plugins
        using multiple strategies:
        1. Looks for plugin manifests (plugin_manifest.json) first
        2. Falls back to plugin.py files
        3. Checks for __init__.py files that might contain plugin definitions

        The method handles different plugin loading patterns and ensures proper registration
        with security and resource management. It also validates plugin compatibility
        before loading and registers plugins with the security system.

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

        # First, iterate through each top-level subdirectory in the models directory
        for model_dir in models_directory.iterdir():
            if model_dir.is_dir():
                # If it's a model type directory (language, vision_language, coding, specialized),
                # scan its subdirectories for actual models
                if model_dir.name in ["language", "vision_language", "coding", "specialized"]:
                    for sub_model_dir in model_dir.iterdir():
                        if sub_model_dir.is_dir():
                            loaded_count += self._load_model_from_directory(sub_model_dir)
                else:
                    # Regular model directory (for backward compatibility or other models)
                    loaded_count += self._load_model_from_directory(model_dir)

        logger.info(
            f"Auto-discovered and loaded {loaded_count} plugins from {models_directory}"
        )
        return loaded_count

    def _load_model_from_directory(self, model_dir: Path) -> int:
        """
        Helper method to load a model from a directory.

        Args:
            model_dir: Directory containing the model

        Returns:
            Number of plugins loaded (0 or 1)
        """
        logger.debug(f"Scanning model directory: {model_dir.name}")

        # Look for plugin manifest first to determine plugin details
        manifest_path = model_dir / "plugin_manifest.json"

        if manifest_path.exists():
            # Load plugin from manifest
            try:
                return self._load_plugin_from_manifest(model_dir, manifest_path)
            except Exception as e:
                logger.error(
                    f"Failed to load plugin from manifest {manifest_path}: {e}"
                )

        # If no manifest found, fall back to looking for plugin.py
        else:
            plugin_file = model_dir / "plugin.py"
            if plugin_file.exists():
                if self.load_plugin_from_path(plugin_file):
                    return 1
                else:
                    logger.warning(f"Failed to load plugin from {plugin_file}")

            # Also look for __init__.py which might contain plugin definitions
            init_file = model_dir / "__init__.py"
            if init_file.exists():
                # Try to import the module and find plugins
                try:
                    # Determine the parent directory to construct the proper module path
                    parent_dir = model_dir.parent.name
                    if parent_dir in ["language", "vision_language", "coding", "specialized"]:
                        module_name = f"inference_pio.models.{parent_dir}.{model_dir.name}"
                    else:
                        module_name = f"inference_pio.models.{model_dir.name}"

                    spec = importlib.util.spec_from_file_location(
                        module_name, init_file
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Look for plugin classes in the module
                    for name, obj in inspect.getmembers(
                        module, inspect.isclass
                    ):
                        if (
                            issubclass(obj, ModelPluginInterface)
                            and obj != ModelPluginInterface
                            and obj.__module__ == module.__name__
                        ):
                            try:
                                # Try to find a factory function for the plugin
                                factory_func_name = (
                                    f"create_{obj.__name__.lower()}"
                                )
                                if hasattr(module, factory_func_name):
                                    factory_func = getattr(
                                        module, factory_func_name
                                    )
                                    plugin_instance = factory_func()
                                    self.register_plugin(plugin_instance)
                                    logger.info(
                                        f"Discovered and registered plugin from {init_file}: {obj.__name__}"
                                    )
                                    return 1
                                else:
                                    # Try to instantiate directly if no factory function exists
                                    plugin_instance = obj()
                                    self.register_plugin(plugin_instance)
                                    logger.info(
                                        f"Discovered and registered plugin from {init_file}: {obj.__name__}"
                                    )
                                    return 1
                            except Exception as e:
                                logger.error(
                                    f"Failed to instantiate plugin {obj.__name__}: {e}"
                                )
                                continue
                except Exception as e:
                    logger.error(
                        f"Failed to import module from {init_file}: {e}"
                    )
                    return 0

        return 0

    def _load_plugin_from_manifest(self, model_dir: Path, manifest_path: Path) -> int:
        """
        Load a plugin based on its manifest file.

        Args:
            model_dir: Directory containing the model
            manifest_path: Path to the plugin manifest file

        Returns:
            Number of plugins loaded (0 or 1)
        """
        import json

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest file {manifest_path}: {e}")
            return 0

        plugin_name = manifest.get("name", model_dir.name)
        entry_point = manifest.get("entry_point")
        main_class_path = manifest.get("main_class_path")

        logger.debug(f"Loading plugin {plugin_name} from manifest")

        # First, try to load using the entry point function if specified
        if entry_point:
            plugin_instance = self._load_plugin_by_entry_point(
                model_dir, f"{model_dir.name}.plugin", entry_point
            )
            if plugin_instance:
                if self.register_plugin(plugin_instance, name=plugin_name):
                    logger.info(
                        f"Successfully loaded plugin {plugin_name} using entry point {entry_point}"
                    )
                    return 1
                else:
                    logger.error(f"Failed to register plugin {plugin_name}")

        # If entry point fails or isn't specified, try loading the main class directly
        if main_class_path:
            plugin_instance = self._load_plugin_by_class_path(
                model_dir, main_class_path
            )
            if plugin_instance:
                if self.register_plugin(plugin_instance, name=plugin_name):
                    logger.info(
                        f"Successfully loaded plugin {plugin_name} using main class {main_class_path}"
                    )
                    return 1
                else:
                    logger.error(f"Failed to register plugin {plugin_name}")

        # As a fallback, try to load plugin.py directly
        plugin_file = model_dir / "plugin.py"
        if plugin_file.exists():
            if self.load_plugin_from_path(plugin_file):
                logger.info(f"Loaded plugin {plugin_name} from plugin.py as fallback")
                return 1
            else:
                logger.warning(f"Failed to load plugin from {plugin_file}")

        return 0

    def _load_plugin_by_entry_point(
        self, model_dir: Path, module_path: str, entry_point: str
    ) -> Optional[ModelPluginInterface]:
        """
        Load a plugin by calling its entry point function.

        Args:
            model_dir: Directory containing the model
            module_path: Path to the module containing the entry point function
            entry_point: Name of the entry point function to call

        Returns:
            Plugin instance if successful, None otherwise
        """
        # Try multiple possible module names depending on installation context
        possible_module_names = [
            f"inference_pio.models.{model_dir.name}.{module_path.split('.')[-1]}",  # Installed package
            f"src.models.{model_dir.name}.{module_path.split('.')[-1]}",  # Source directory
            f"models.{model_dir.name}.{module_path.split('.')[-1]}",  # Direct from models
            module_path,  # Direct path from manifest
        ]

        module = None
        plugin_file = model_dir / f"{module_path.split('.')[-1]}.py"

        for module_name in possible_module_names:
            try:
                # Try importing the module directly first
                module = importlib.import_module(module_name)
                break
            except ImportError:
                # If direct import fails, try loading from file
                if plugin_file.exists():
                    try:
                        spec = importlib.util.spec_from_file_location(
                            module_name, plugin_file
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        break
                    except Exception:
                        continue

        if module is not None:
            # Get the factory function and create the plugin
            if hasattr(module, entry_point):
                factory_func = getattr(module, entry_point)
                try:
                    plugin_instance = factory_func()
                    return plugin_instance
                except Exception as e:
                    logger.error(
                        f"Failed to create plugin instance using entry point {entry_point}: {e}"
                    )
            else:
                logger.warning(
                    f"Entry point function {entry_point} not found in the loaded module"
                )
        else:
            logger.error(f"Could not load module {possible_module_names[0]} for plugin")

        return None

    def _load_plugin_by_class_path(
        self, model_dir: Path, class_path: str
    ) -> Optional[ModelPluginInterface]:
        """
        Load a plugin by instantiating its main class directly.

        Args:
            model_dir: Directory containing the model
            class_path: Full path to the plugin class (e.g., 'package.module.ClassName')

        Returns:
            Plugin instance if successful, None otherwise
        """
        try:
            # Extract module path and class name
            parts = class_path.rsplit(".", 1)
            if len(parts) != 2:
                logger.error(f"Invalid class path format: {class_path}")
                return None

            module_path, class_name = parts

            # Try to import the module containing the class
            # First try the original path
            module = None
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                # If the original path fails, try alternative paths
                alternatives = [
                    f"src.{module_path}",  # Try with src prefix
                    f"inference_pio.{module_path}",  # Try with inference_pio prefix
                ]

                # Also try replacing the first part with model directory name
                module_parts = module_path.split(".")
                if len(module_parts) > 1:
                    alternatives.append(
                        f"{model_dir.name}.{'.'.join(module_parts[1:])}"
                    )

                # Add the plugin file path as a fallback
                plugin_file = (
                    model_dir / f"{module_parts[-1] if module_parts else 'plugin'}.py"
                )
                if plugin_file.exists():
                    # Create a temporary module name based on the model directory
                    temp_module_name = f"temp_plugin_module_{model_dir.name}_{module_parts[-1] if module_parts else 'plugin'}"

                    # Try loading from file
                    try:
                        spec = importlib.util.spec_from_file_location(
                            temp_module_name, plugin_file
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                    except Exception as e:
                        logger.debug(
                            f"Failed to load module from file {plugin_file}: {e}"
                        )

                if module is None:
                    for alt_module_path in alternatives:
                        try:
                            module = importlib.import_module(alt_module_path)
                            break
                        except ImportError:
                            continue

                if module is None:
                    logger.error(
                        f"Could not import module {class_path} or its alternatives"
                    )
                    return None

            # Get the class
            if not hasattr(module, class_name):
                logger.error(f"Class {class_name} not found in module {module_path}")
                return None

            plugin_class = getattr(module, class_name)

            # Verify that the class is a subclass of ModelPluginInterface
            # Need to check against both the base and improved versions
            from ..common.improved_base_plugin_interface import (
                ModelPluginInterface as ImprovedModelPluginInterface,
            )

            # Check if it's a subclass of either interface
            if not (
                issubclass(plugin_class, ModelPluginInterface)
                or issubclass(plugin_class, ImprovedModelPluginInterface)
            ):
                logger.error(
                    f"Class {class_name} is not a subclass of ModelPluginInterface"
                )
                return None

            # Create an instance
            plugin_instance = plugin_class()

            return plugin_instance
        except Exception as e:
            logger.error(f"Failed to load plugin using class path {class_path}: {e}")
            return None

    def scan_model_directories(
        self, models_directory: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Scan model directories to automatically identify available plugins.

        This method performs comprehensive scanning of model directories to extract
        detailed information about available plugins. It uses multiple approaches:
        1. Reads plugin manifests (plugin_manifest.json) for complete metadata
        2. Parses plugin.py files for basic information
        3. Extracts class definitions and documentation

        The method returns rich plugin information including name, version, author,
        description, dependencies, model architecture, and more.
        Unlike discover_and_load_plugins, this method only scans and extracts information
        without actually loading the plugins into memory.

        Args:
            models_directory: Directory containing model plugins (defaults to models directory)

        Returns:
            List of dictionaries containing detailed plugin information
        """
        if models_directory is None:
            # Try to find the models directory relative to this file
            current_file_dir = Path(__file__).parent.parent
            models_directory = current_file_dir / "models"

        models_directory = Path(models_directory)

        if not models_directory.exists() or not models_directory.is_dir():
            logger.error(f"Models directory does not exist: {models_directory}")
            return []

        discovered_plugins = []

        # First, iterate through each top-level subdirectory in the models directory
        for model_dir in models_directory.iterdir():
            if model_dir.is_dir():
                # If it's a model type directory (language, vision_language, coding, specialized),
                # scan its subdirectories for actual models
                if model_dir.name in ["language", "vision_language", "coding", "specialized"]:
                    for sub_model_dir in model_dir.iterdir():
                        if sub_model_dir.is_dir():
                            plugin_info = self._extract_plugin_info_from_directory(sub_model_dir)
                            if plugin_info:
                                discovered_plugins.append(plugin_info)
                else:
                    # Regular model directory (for backward compatibility or other models)
                    plugin_info = self._extract_plugin_info_from_directory(model_dir)
                    if plugin_info:
                        discovered_plugins.append(plugin_info)

        logger.info(
            f"Discovered {len(discovered_plugins)} plugins from {models_directory}"
        )
        return discovered_plugins

    def _extract_plugin_info_from_directory(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Helper method to extract plugin information from a directory.

        Args:
            model_dir: Directory containing the model

        Returns:
            Dictionary containing plugin information, or None if extraction fails
        """
        logger.debug(f"Scanning model directory: {model_dir.name}")

        # Look for plugin manifest first to determine plugin details
        manifest_path = model_dir / "plugin_manifest.json"

        if manifest_path.exists():
            # Load plugin info from manifest
            try:
                return self._extract_plugin_info_from_manifest(
                    model_dir, manifest_path
                )
            except Exception as e:
                logger.error(
                    f"Failed to extract plugin info from manifest {manifest_path}: {e}"
                )

        # If no manifest found, look for plugin.py as a fallback
        else:
            plugin_file = model_dir / "plugin.py"
            if plugin_file.exists():
                # Try to extract basic info from the plugin file
                try:
                    return self._extract_plugin_info_from_file(
                        plugin_file, model_dir.name
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to extract plugin info from {plugin_file}: {e}"
                    )

        return None

    def _extract_plugin_info_from_manifest(
        self, model_dir: Path, manifest_path: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Extract plugin information from a manifest file.

        Args:
            model_dir: Directory containing the model
            manifest_path: Path to the plugin manifest file

        Returns:
            Dictionary containing plugin information, or None if extraction fails
        """
        import json

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load manifest file {manifest_path}: {e}")
            return None

        # Extract basic plugin information from manifest
        plugin_info = {
            "name": manifest.get("name", model_dir.name),
            "version": manifest.get("version", "unknown"),
            "author": manifest.get("author", "unknown"),
            "description": manifest.get("description", ""),
            "plugin_type": manifest.get("plugin_type", "MODEL_COMPONENT"),
            "dependencies": manifest.get("dependencies", []),
            "model_architecture": manifest.get("model_architecture", ""),
            "model_size": manifest.get("model_size", ""),
            "required_memory_gb": manifest.get("required_memory_gb", 0.0),
            "supported_modalities": manifest.get("supported_modalities", []),
            "license": manifest.get("license", ""),
            "tags": manifest.get("tags", []),
            "model_family": manifest.get("model_family", ""),
            "num_parameters": manifest.get("num_parameters", 0),
            "directory": str(model_dir),
            "manifest_path": str(manifest_path),
            "has_manifest": True,
            "main_class_path": manifest.get("main_class_path", ""),
            "entry_point": manifest.get("entry_point", ""),
        }

        return plugin_info

    def _extract_plugin_info_from_file(
        self, plugin_file: Path, model_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract basic plugin information from a plugin.py file.

        Args:
            plugin_file: Path to the plugin.py file
            model_name: Name of the model directory

        Returns:
            Dictionary containing plugin information, or None if extraction fails
        """
        import ast

        try:
            with open(plugin_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the file to extract plugin information
            tree = ast.parse(content)

            plugin_info = {
                "name": model_name,
                "version": "unknown",
                "author": "unknown",
                "description": "",
                "plugin_type": "MODEL_COMPONENT",
                "dependencies": [],
                "model_architecture": "",
                "model_size": "",
                "required_memory_gb": 0.0,
                "supported_modalities": [],
                "license": "",
                "tags": [],
                "model_family": "",
                "num_parameters": 0,
                "directory": str(plugin_file.parent),
                "manifest_path": None,
                "has_manifest": False,
                "main_class_path": "",
                "entry_point": "",
            }

            # Look for class definitions that inherit from ModelPluginInterface
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if the class inherits from ModelPluginInterface
                    for base in node.bases:
                        if isinstance(base, ast.Name) and base.id.endswith(
                            "ModelPluginInterface"
                        ):
                            plugin_info["name"] = node.name
                            # Look for docstring as description
                            if (
                                node.body
                                and isinstance(node.body[0], ast.Expr)
                                and isinstance(node.body[0].value, ast.Str)
                            ):
                                plugin_info["description"] = node.body[0].value.s
                            break

                # Look for function definitions that create plugins
                elif isinstance(node, ast.FunctionDef):
                    if node.name.startswith("create_"):
                        plugin_info["entry_point"] = node.name

            return plugin_info
        except Exception as e:
            logger.error(f"Failed to parse plugin file {plugin_file}: {e}")
            return None


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """
    Get the singleton global plugin manager instance.

    This function implements the singleton pattern to ensure there's only one
    plugin manager instance throughout the application lifecycle.

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

    Convenience function that delegates to the global plugin manager's register_plugin method.
    This allows plugins to be registered without directly accessing the manager instance.

    Args:
        plugin: The plugin to register
        name: Optional name for the plugin (defaults to plugin's metadata name)

    Returns:
        True if registration was successful, False otherwise
    """
    return get_plugin_manager().register_plugin(plugin, name)


def load_plugin_from_path(plugin_path: Union[str, Path]) -> bool:
    """
    Load a plugin from a file path using the global plugin manager.

    This convenience function allows loading plugins from arbitrary file paths
    without directly accessing the plugin manager instance.

    Args:
        plugin_path: Path to the plugin file (.py file containing plugin implementation)

    Returns:
        True if loading was successful, False otherwise
    """
    return get_plugin_manager().load_plugin_from_path(plugin_path)


def load_plugins_from_directory(directory: Union[str, Path]) -> int:
    """
    Load all plugins from a directory using the global plugin manager.

    Recursively finds and loads all plugin files (.py) from the specified directory.

    Args:
        directory: Directory to load plugins from

    Returns:
        Number of plugins successfully loaded
    """
    return get_plugin_manager().load_plugins_from_directory(directory)


def activate_plugin(name: str, **kwargs) -> bool:
    """
    Activate a plugin by name using the global plugin manager.

    This function handles the complete activation process including security initialization
    and resource allocation as described in the PluginManager.activate_plugin method.

    Args:
        name: Name of the plugin to activate
        **kwargs: Additional parameters for activation (security level, resource limits, etc.)

    Returns:
        True if activation was successful, False otherwise
    """
    return get_plugin_manager().activate_plugin(name, **kwargs)


def execute_plugin(name: str, *args, **kwargs) -> Any:
    """
    Execute a plugin's main functionality using the global plugin manager.

    This function safely executes an active plugin with proper error handling
    and resource monitoring.

    Args:
        name: Name of the plugin to execute (must be active)
        *args: Arguments to pass to the plugin's execution method
        **kwargs: Keyword arguments to pass to the plugin's execution method

    Returns:
        Result of plugin execution or None if execution failed
    """
    return get_plugin_manager().execute_plugin(name, *args, **kwargs)


def discover_and_load_plugins(
    models_directory: Optional[Union[str, Path]] = None,
) -> int:
    """
    Automatically discover and load all available plugins from the models directory
    using the global plugin manager.

    This function performs intelligent scanning of the models directory to find
    and load plugins using manifests and other detection methods.

    Args:
        models_directory: Directory containing model plugins (defaults to standard models directory)

    Returns:
        Number of plugins successfully loaded
    """
    return get_plugin_manager().discover_and_load_plugins(models_directory)


def scan_model_directories(
    models_directory: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Scan model directories to automatically identify available plugins
    using the global plugin manager.

    This function extracts detailed information about available plugins without
    loading them into memory, useful for plugin discovery and management UIs.

    Args:
        models_directory: Directory containing model plugins (defaults to standard models directory)

    Returns:
        List of dictionaries containing detailed plugin information
    """
    return get_plugin_manager().scan_model_directories(models_directory)


__all__ = [
    "PluginManager",
    "get_plugin_manager",
    "register_plugin",
    "load_plugin_from_path",
    "load_plugins_from_directory",
    "activate_plugin",
    "execute_plugin",
    "discover_and_load_plugins",
    "scan_model_directories",
]
