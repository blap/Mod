"""
Configuration Integration for Inference-PIO System

This module provides integration between the dynamic configuration system
and the existing plugin architecture in the Inference-PIO system.
"""

from typing import Any, Dict, Optional, Union
from .config_manager import get_config_manager, DynamicConfig
from .config_loader import get_config_loader
from .config_validator import get_config_validator
from .base_plugin_interface import ModelPluginInterface
from .optimization_manager import get_optimization_manager
from .optimization_config import get_config_manager as get_optimization_config_manager
import logging

logger = logging.getLogger(__name__)


class ConfigurablePluginMixin:
    """
    Mixin class that adds configuration capabilities to plugins.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config_manager = get_config_manager()
        self._config_loader = get_config_loader()
        self._config_validator = get_config_validator()
        self._optimization_manager = get_optimization_manager()
        self._optimization_config_manager = get_optimization_config_manager()
        self._active_config_name = None
    
    def load_configuration(self, config_path: str, config_name: str) -> bool:
        """
        Load a configuration from a file.
        
        Args:
            config_path: Path to the configuration file
            config_name: Name to assign to the loaded configuration
            
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            success = self._config_loader.load_config_from_file(config_path, config_name)
            if success:
                logger.info(f"Loaded configuration '{config_name}' from {config_path}")
                # Validate the loaded configuration
                config = self._config_manager.get_config(config_name)
                if config:
                    is_valid, errors = self._config_validator.validate_config(config)
                    if not is_valid:
                        logger.warning(f"Loaded configuration '{config_name}' has validation errors: {errors}")
            return success
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            return False
    
    def save_configuration(self, config_name: str, config_path: str, format: str = "json") -> bool:
        """
        Save a configuration to a file.
        
        Args:
            config_name: Name of the configuration to save
            config_path: Path to save the configuration file
            format: Format to save in ("json" or "yaml")
            
        Returns:
            True if saving was successful, False otherwise
        """
        try:
            success = self._config_manager.save_config(config_name, config_path, format)
            if success:
                logger.info(f"Saved configuration '{config_name}' to {config_path}")
            return success
        except Exception as e:
            logger.error(f"Failed to save configuration '{config_name}' to {config_path}: {e}")
            return False
    
    def get_configuration(self, config_name: str) -> Optional[DynamicConfig]:
        """
        Get a configuration by name.
        
        Args:
            config_name: Name of the configuration to retrieve
            
        Returns:
            Configuration object or None if not found
        """
        return self._config_manager.get_config(config_name)
    
    def update_configuration(self, config_name: str, updates: Dict[str, Any]) -> bool:
        """
        Update a configuration with new values.
        
        Args:
            config_name: Name of the configuration to update
            updates: Dictionary of field updates
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            success = self._config_manager.update_config(config_name, updates)
            if success:
                logger.info(f"Updated configuration '{config_name}'")
                
                # Validate the updated configuration
                config = self._config_manager.get_config(config_name)
                if config:
                    is_valid, errors = self._config_validator.validate_config(config)
                    if not is_valid:
                        logger.warning(f"Updated configuration '{config_name}' has validation errors: {errors}")
            return success
        except Exception as e:
            logger.error(f"Failed to update configuration '{config_name}': {e}")
            return False
    
    def activate_configuration(self, config_name: str, model_id: Optional[str] = None) -> bool:
        """
        Activate a configuration for use with the plugin.
        
        Args:
            config_name: Name of the configuration to activate
            model_id: Optional model identifier (defaults to plugin's metadata name)
            
        Returns:
            True if activation was successful, False otherwise
        """
        if model_id is None:
            model_id = getattr(self, 'metadata', {}).get('name', 'default_model')
        
        try:
            success = self._config_manager.activate_config_for_model(model_id, config_name)
            if success:
                self._active_config_name = config_name
                logger.info(f"Activated configuration '{config_name}' for model '{model_id}'")
                
                # Apply optimizations based on the configuration
                self._apply_config_optimizations(config_name)
            return success
        except Exception as e:
            logger.error(f"Failed to activate configuration '{config_name}' for model '{model_id}': {e}")
            return False
    
    def get_active_configuration(self, model_id: Optional[str] = None) -> Optional[DynamicConfig]:
        """
        Get the active configuration for the plugin.
        
        Args:
            model_id: Optional model identifier (defaults to plugin's metadata name)
            
        Returns:
            Active configuration object or None if no configuration is active
        """
        if model_id is None:
            model_id = getattr(self, 'metadata', {}).get('name', 'default_model')
        
        return self._config_manager.get_active_config_for_model(model_id)
    
    def create_configuration_from_template(self, template_name: str, config_name: str, 
                                         overrides: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a configuration from a template with optional overrides.
        
        Args:
            template_name: Name of the template to use
            config_name: Name for the new configuration
            overrides: Optional dictionary of field overrides
            
        Returns:
            True if creation was successful, False otherwise
        """
        try:
            success = self._config_loader.create_config_from_template(
                template_name, config_name, overrides
            )
            if success:
                logger.info(f"Created configuration '{config_name}' from template '{template_name}'")
                
                # Validate the created configuration
                config = self._config_manager.get_config(config_name)
                if config:
                    is_valid, errors = self._config_validator.validate_config(config)
                    if not is_valid:
                        logger.warning(f"Created configuration '{config_name}' has validation errors: {errors}")
            return success
        except Exception as e:
            logger.error(f"Failed to create configuration from template '{template_name}': {e}")
            return False
    
    def _apply_config_optimizations(self, config_name: str):
        """
        Apply optimizations based on the configuration.
        
        Args:
            config_name: Name of the configuration to apply optimizations for
        """
        try:
            config = self._config_manager.get_config(config_name)
            if not config:
                logger.warning(f"Configuration '{config_name}' not found for optimization application")
                return
            
            # Extract optimization settings from the configuration
            optimization_names = []
            
            # Map configuration flags to optimization names
            if hasattr(config, 'use_flash_attention_2') and config.use_flash_attention_2:
                optimization_names.append('flash_attention')
            
            if hasattr(config, 'use_sparse_attention') and config.use_sparse_attention:
                optimization_names.append('sparse_attention')
            
            if hasattr(config, 'use_tensor_compression') and config.use_tensor_compression:
                optimization_names.append('tensor_compression')
            
            if hasattr(config, 'use_tensor_decomposition') and config.use_tensor_decomposition:
                optimization_names.append('tensor_decomposition')
            
            if hasattr(config, 'use_structured_pruning') and config.use_structured_pruning:
                optimization_names.append('structured_pruning')
            
            if hasattr(config, 'enable_kernel_fusion') and config.enable_kernel_fusion:
                optimization_names.append('kernel_fusion')
            
            if hasattr(config, 'enable_disk_offloading') and config.enable_disk_offloading:
                optimization_names.append('disk_offloading')
            
            if hasattr(config, 'enable_activation_offloading') and config.enable_activation_offloading:
                optimization_names.append('activation_offloading')
            
            if hasattr(config, 'enable_adaptive_batching') and config.enable_adaptive_batching:
                optimization_names.append('adaptive_batching')
            
            # Apply optimizations
            if optimization_names:
                logger.info(f"Applying optimizations {optimization_names} based on configuration '{config_name}'")
                # Note: Actual optimization application would happen here
                # This is a placeholder since we don't have the model instance here
        except Exception as e:
            logger.error(f"Failed to apply optimizations from configuration '{config_name}': {e}")
    
    def get_configuration_metadata(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Metadata dictionary or None if config not found
        """
        return self._config_manager.get_config_metadata(config_name)
    
    def list_configurations(self) -> list:
        """
        List all registered configurations.
        
        Returns:
            List of configuration names
        """
        return self._config_manager.list_configs()
    
    def get_available_templates(self) -> list:
        """
        Get list of available configuration templates.
        
        Returns:
            List of template names
        """
        return self._config_loader.get_available_templates()


class ConfigurableModelPlugin(ConfigurablePluginMixin, ModelPluginInterface):
    """
    Base class for model plugins that support dynamic configuration.
    """
    
    def __init__(self, metadata):
        super().__init__(metadata)
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the plugin with the provided parameters.
        
        Args:
            **kwargs: Additional initialization parameters
            
        Returns:
            True if initialization was successful, False otherwise
        """
        # Check if a configuration is specified in kwargs
        config_name = kwargs.get('config_name')
        if config_name:
            # Activate the specified configuration
            model_id = self.metadata.name if hasattr(self, 'metadata') else 'default_model'
            self.activate_configuration(config_name, model_id)
        
        # Continue with regular initialization
        return super().initialize(**kwargs) if hasattr(super(), 'initialize') else True
    
    def load_model(self, config: Any = None) -> 'nn.Module':
        """
        Load the model with the given configuration.
        
        Args:
            config: Model configuration (optional)
            
        Returns:
            Loaded model instance
        """
        # If no config is provided, try to use the active configuration
        if config is None:
            model_id = self.metadata.name if hasattr(self, 'metadata') else 'default_model'
            config = self.get_active_configuration(model_id)
        
        # Continue with regular model loading
        return super().load_model(config) if hasattr(super(), 'load_model') else None


# Helper function to apply configurations to plugins
def apply_configuration_to_plugin(plugin: ModelPluginInterface, config_name: str, 
                               model_id: Optional[str] = None) -> bool:
    """
    Apply a configuration to a plugin.
    
    Args:
        plugin: Plugin to apply configuration to
        config_name: Name of the configuration to apply
        model_id: Optional model identifier
        
    Returns:
        True if application was successful, False otherwise
    """
    if not hasattr(plugin, 'activate_configuration'):
        logger.error("Plugin does not support dynamic configuration")
        return False
    
    return plugin.activate_configuration(config_name, model_id)


# Helper function to create configurations from profiles
def create_config_from_profile(plugin: ModelPluginInterface, model_type: str, 
                             profile_name: str, config_name: str, **overrides) -> bool:
    """
    Create a configuration from a predefined profile.
    
    Args:
        plugin: Plugin to create configuration for
        model_type: Type of model ('glm', 'qwen3_4b', 'qwen3_coder', 'qwen3_vl')
        profile_name: Name of the profile ('performance', 'memory_efficient', 'balanced')
        config_name: Name for the new configuration
        **overrides: Additional overrides to apply
        
    Returns:
        True if creation was successful, False otherwise
    """
    if not hasattr(plugin, 'create_configuration_from_template'):
        logger.error("Plugin does not support dynamic configuration")
        return False
    
    try:
        from .config_loader import create_config_from_profile as loader_create_profile
        config_params = loader_create_profile(model_type, profile_name, **overrides)
        
        # Create a temporary config to determine the appropriate template
        if model_type == 'glm' or model_type == 'glm_4_7_flash':
            template_name = 'glm_4_7_flash'
        elif model_type == 'qwen3_coder':
            template_name = 'qwen3_coder_30b'
        elif model_type == 'qwen3_vl':
            template_name = 'qwen3_vl_2b'
        elif model_type == 'qwen3_4b':
            template_name = 'qwen3_4b_instruct_2507'
        else:
            logger.error(f"Unknown model type: {model_type}")
            return False
        
        # Create configuration from template with profile parameters
        return plugin.create_configuration_from_template(template_name, config_name, config_params)
    except Exception as e:
        logger.error(f"Failed to create configuration from profile '{profile_name}' for model '{model_type}': {e}")
        return False


__all__ = [
    "ConfigurablePluginMixin",
    "ConfigurableModelPlugin",
    "apply_configuration_to_plugin",
    "create_config_from_profile",
]