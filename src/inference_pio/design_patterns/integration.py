"""
Design Patterns Integration for Inference-PIO

This module integrates the Factory, Strategy, and Adapter patterns to provide
a unified interface for creating and managing optimized model plugins.
"""

import logging
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from .factory import (
    PluginFactoryProvider,
    OptimizationStrategyFactoryProvider,
    ModelAdapterFactoryProvider
)
from .strategy import OptimizationSelector
from .adapter import ModelIntegrationAdapter


logger = logging.getLogger(__name__)


class DesignPatternIntegration:
    """
    Main integration class that combines Factory, Strategy, and Adapter patterns.
    """
    
    def __init__(self):
        self.plugin_factory_provider = PluginFactoryProvider()
        self.optimization_factory_provider = OptimizationStrategyFactoryProvider()
        self.model_adapter_factory_provider = ModelAdapterFactoryProvider()
        self.optimization_selector = OptimizationSelector()
    
    def create_optimized_plugin(self, model_type: str, optimization_criteria: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create an optimized plugin using Factory and Strategy patterns.
        
        Args:
            model_type: Type of model to create ('glm_4_7_flash', 'qwen3_4b_instruct_2507', etc.)
            optimization_criteria: Criteria for selecting optimization strategy
            
        Returns:
            Optimized plugin instance
        """
        logger.info(f"Creating optimized plugin for model type: {model_type}")
        
        # Use Factory pattern to create the plugin
        plugin = self.plugin_factory_provider.create_plugin(model_type)
        
        # Load the model
        model = plugin.load_model()
        
        # Use Strategy pattern to select and apply optimizations
        if optimization_criteria is None:
            optimization_criteria = {}
        
        # Add model-specific information to criteria
        optimization_criteria['required_memory_gb'] = plugin.metadata.required_memory_gb
        
        optimized_model = self.optimization_selector.optimize_with_criteria(model, optimization_criteria)
        
        # Update the plugin with the optimized model
        plugin._model = optimized_model
        
        logger.info(f"Plugin created and optimized successfully for model type: {model_type}")
        return plugin
    
    def create_adapted_model(self, model_type: str, model: nn.Module, 
                           depth_ratio: float = 1.0, width_ratio: float = 1.0) -> ModelIntegrationAdapter:
        """
        Create an adapted model using the Adapter pattern.
        
        Args:
            model_type: Type of model ('glm_4_7_flash', 'qwen3_4b_instruct_2507', etc.)
            model: The model to adapt
            depth_ratio: Ratio to adjust depth (0.0 to 1.0)
            width_ratio: Ratio to adjust width (0.0 to 1.0)
            
        Returns:
            Adapted model wrapped in integration adapter
        """
        logger.info(f"Creating adapted model for type: {model_type}")
        
        # Use Factory pattern to create the appropriate adapter
        adapter = self.model_adapter_factory_provider.create_adapter(model_type, model)
        
        # Apply architectural adaptations
        adapted_model = adapter.adapt_architecture(depth_ratio, width_ratio)
        
        # Wrap in integration adapter for unified interface
        integration_adapter = ModelIntegrationAdapter(adapted_model, model_type)
        
        logger.info(f"Model adapted successfully for type: {model_type}")
        return integration_adapter
    
    def create_optimized_adapted_plugin(self, model_type: str, 
                                      optimization_criteria: Optional[Dict[str, Any]] = None,
                                      architecture_ratios: Optional[Dict[str, float]] = None) -> Any:
        """
        Create a plugin that is both optimized (using Strategy) and adapted (using Adapter)
        using the Factory pattern to create all components.
        
        Args:
            model_type: Type of model to create
            optimization_criteria: Criteria for selecting optimization strategy
            architecture_ratios: Ratios for architectural adaptations {'depth_ratio': float, 'width_ratio': float}
            
        Returns:
            Fully optimized and adapted plugin
        """
        logger.info(f"Creating fully optimized and adapted plugin for model type: {model_type}")
        
        # Step 1: Create plugin using Factory pattern
        plugin = self.plugin_factory_provider.create_plugin(model_type)
        
        # Step 2: Load the model
        model = plugin.load_model()
        
        # Step 3: Apply architectural adaptations using Adapter pattern
        if architecture_ratios is None:
            architecture_ratios = {'depth_ratio': 1.0, 'width_ratio': 1.0}
        
        depth_ratio = architecture_ratios.get('depth_ratio', 1.0)
        width_ratio = architecture_ratios.get('width_ratio', 1.0)
        
        # Create adapter and apply adaptations
        adapter = self.model_adapter_factory_provider.create_adapter(model_type, model)
        adapted_model = adapter.adapt_architecture(depth_ratio, width_ratio)
        
        # Update plugin with adapted model
        plugin._model = adapted_model
        
        # Step 4: Apply optimizations using Strategy pattern
        if optimization_criteria is None:
            optimization_criteria = {}
        
        # Add model-specific information to criteria
        optimization_criteria['required_memory_gb'] = plugin.metadata.required_memory_gb
        
        optimized_model = self.optimization_selector.optimize_with_criteria(adapted_model, optimization_criteria)
        
        # Update the plugin with the optimized model
        plugin._model = optimized_model
        
        logger.info(f"Fully optimized and adapted plugin created for model type: {model_type}")
        return plugin


# Global instance for easy access
_design_pattern_integration = None


def get_design_pattern_integration() -> DesignPatternIntegration:
    """
    Get the global design pattern integration instance.
    
    Returns:
        DesignPatternIntegration instance
    """
    global _design_pattern_integration
    if _design_pattern_integration is None:
        _design_pattern_integration = DesignPatternIntegration()
    return _design_pattern_integration


def create_optimized_plugin(model_type: str, optimization_criteria: Optional[Dict[str, Any]] = None) -> Any:
    """
    Create an optimized plugin using Factory and Strategy patterns.
    
    Args:
        model_type: Type of model to create
        optimization_criteria: Criteria for selecting optimization strategy
        
    Returns:
        Optimized plugin instance
    """
    integration = get_design_pattern_integration()
    return integration.create_optimized_plugin(model_type, optimization_criteria)


def create_adapted_model(model_type: str, model: nn.Module, 
                       depth_ratio: float = 1.0, width_ratio: float = 1.0) -> ModelIntegrationAdapter:
    """
    Create an adapted model using the Adapter pattern.
    
    Args:
        model_type: Type of model
        model: The model to adapt
        depth_ratio: Ratio to adjust depth (0.0 to 1.0)
        width_ratio: Ratio to adjust width (0.0 to 1.0)
        
    Returns:
        Adapted model wrapped in integration adapter
    """
    integration = get_design_pattern_integration()
    return integration.create_adapted_model(model_type, model, depth_ratio, width_ratio)


def create_optimized_adapted_plugin(model_type: str, 
                                 optimization_criteria: Optional[Dict[str, Any]] = None,
                                 architecture_ratios: Optional[Dict[str, float]] = None) -> Any:
    """
    Create a plugin that is both optimized and adapted.
    
    Args:
        model_type: Type of model to create
        optimization_criteria: Criteria for selecting optimization strategy
        architecture_ratios: Ratios for architectural adaptations
        
    Returns:
        Fully optimized and adapted plugin
    """
    integration = get_design_pattern_integration()
    return integration.create_optimized_adapted_plugin(model_type, optimization_criteria, architecture_ratios)


# Example usage functions for the specific models mentioned in the requirements
def create_glm_4_7_optimized_plugin(optimization_criteria: Optional[Dict[str, Any]] = None,
                                  architecture_ratios: Optional[Dict[str, float]] = None) -> Any:
    """
    Create an optimized GLM-4-7 plugin.
    
    Args:
        optimization_criteria: Criteria for selecting optimization strategy
        architecture_ratios: Ratios for architectural adaptations
        
    Returns:
        Optimized GLM-4-7 plugin
    """
    return create_optimized_adapted_plugin('glm_4_7_flash', optimization_criteria, architecture_ratios)


def create_qwen3_4b_instruct_2507_optimized_plugin(optimization_criteria: Optional[Dict[str, Any]] = None,
                                                  architecture_ratios: Optional[Dict[str, float]] = None) -> Any:
    """
    Create an optimized Qwen3-4b-instruct-2507 plugin.
    
    Args:
        optimization_criteria: Criteria for selecting optimization strategy
        architecture_ratios: Ratios for architectural adaptations
        
    Returns:
        Optimized Qwen3-4b-instruct-2507 plugin
    """
    return create_optimized_adapted_plugin('qwen3_4b_instruct_2507', optimization_criteria, architecture_ratios)


def create_qwen3_coder_30b_optimized_plugin(optimization_criteria: Optional[Dict[str, Any]] = None,
                                           architecture_ratios: Optional[Dict[str, float]] = None) -> Any:
    """
    Create an optimized Qwen3-coder-30b plugin.
    
    Args:
        optimization_criteria: Criteria for selecting optimization strategy
        architecture_ratios: Ratios for architectural adaptations
        
    Returns:
        Optimized Qwen3-coder-30b plugin
    """
    return create_optimized_adapted_plugin('qwen3_coder_30b', optimization_criteria, architecture_ratios)


def create_qwen3_vl_2b_optimized_plugin(optimization_criteria: Optional[Dict[str, Any]] = None,
                                       architecture_ratios: Optional[Dict[str, float]] = None) -> Any:
    """
    Create an optimized Qwen3-vl-2b plugin.
    
    Args:
        optimization_criteria: Criteria for selecting optimization strategy
        architecture_ratios: Ratios for architectural adaptations
        
    Returns:
        Optimized Qwen3-vl-2b plugin
    """
    return create_optimized_adapted_plugin('qwen3_vl_2b', optimization_criteria, architecture_ratios)


__all__ = [
    'DesignPatternIntegration',
    'get_design_pattern_integration',
    'create_optimized_plugin',
    'create_adapted_model',
    'create_optimized_adapted_plugin',
    'create_glm_4_7_optimized_plugin',
    'create_qwen3_4b_instruct_2507_optimized_plugin',
    'create_qwen3_coder_30b_optimized_plugin',
    'create_qwen3_vl_2b_optimized_plugin'
]