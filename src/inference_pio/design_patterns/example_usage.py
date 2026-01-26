"""
Example Usage of Design Patterns in Inference-PIO

This module demonstrates how to use the Factory, Strategy, and Adapter patterns
together in the Inference-PIO system.
"""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn

from .factory import (
    PluginFactoryProvider,
    OptimizationStrategyFactoryProvider,
    ModelAdapterFactoryProvider
)
from .strategy import OptimizationSelector
from .adapter import ModelAdapterSelector, ModelIntegrationAdapter
from .integration import (
    DesignPatternIntegration,
    create_glm_4_7_optimized_plugin,
    create_qwen3_4b_instruct_2507_optimized_plugin,
    create_qwen3_coder_30b_optimized_plugin,
    create_qwen3_vl_2b_optimized_plugin
)


logger = logging.getLogger(__name__)


def demonstrate_factory_pattern():
    """
    Demonstrate the Factory pattern for creating plugins.
    """
    print("=== Factory Pattern Demonstration ===")
    
    # Use Factory pattern to create plugins
    factory_provider = PluginFactoryProvider()
    
    # Create GLM-4.7 plugin
    glm_plugin = factory_provider.create_plugin('glm_4_7_flash')
    print(f"Created plugin: {glm_plugin.metadata.name}")
    
    # Create Qwen3-4B-Instruct-2507 plugin
    qwen4b_plugin = factory_provider.create_plugin('qwen3_4b_instruct_2507')
    print(f"Created plugin: {qwen4b_plugin.metadata.name}")
    
    # Create Qwen3-Coder-30B plugin
    qwen30b_plugin = factory_provider.create_plugin('qwen3_coder_30b')
    print(f"Created plugin: {qwen30b_plugin.metadata.name}")
    
    # Create Qwen3-VL-2B plugin
    qwen_vl_plugin = factory_provider.create_plugin('qwen3_vl_2b')
    print(f"Created plugin: {qwen_vl_plugin.metadata.name}")
    
    print()


def demonstrate_strategy_pattern():
    """
    Demonstrate the Strategy pattern for optimization selection.
    """
    print("=== Strategy Pattern Demonstration ===")
    
    # Create a mock model for demonstration
    mock_model = nn.Linear(100, 50)
    
    # Use Strategy pattern to select and apply optimizations
    selector = OptimizationSelector()
    
    # Scenario 1: Low memory environment
    low_memory_criteria = {
        'available_memory_gb': 4.0,
        'required_memory_gb': 8.0,
        'performance_priority': 'memory'
    }
    print("Scenario 1: Low memory environment")
    optimized_model_1 = selector.optimize_with_criteria(mock_model, low_memory_criteria)
    print(f"Applied optimization strategy for low memory")
    
    # Scenario 2: High memory environment
    high_memory_criteria = {
        'available_memory_gb': 32.0,
        'required_memory_gb': 8.0,
        'performance_priority': 'compute'
    }
    print("Scenario 2: High memory environment")
    optimized_model_2 = selector.optimize_with_criteria(mock_model, high_memory_criteria)
    print(f"Applied optimization strategy for high memory")
    
    # Scenario 3: Adaptive optimization
    adaptive_criteria = {
        'available_memory_gb': 16.0,
        'required_memory_gb': 8.0,
        'adaptive': True
    }
    print("Scenario 3: Adaptive optimization")
    optimized_model_3 = selector.optimize_with_criteria(mock_model, adaptive_criteria)
    print(f"Applied adaptive optimization strategy")
    
    print()


def demonstrate_adapter_pattern():
    """
    Demonstrate the Adapter pattern for model integration.
    """
    print("=== Adapter Pattern Demonstration ===")
    
    # Create mock models for different architectures
    glm_model = nn.Sequential(nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 50))
    qwen_model = nn.Sequential(nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 50))
    
    # Use Adapter pattern to adapt models
    adapter_selector = ModelAdapterSelector()
    
    # Adapt GLM-4.7 model (reduce to 50% depth and width)
    adapted_glm = adapter_selector.adapt_model('glm_4_7_flash', glm_model, depth_ratio=0.5, width_ratio=0.5)
    print(f"GLM-4.7 model adapted: depth_ratio=0.5, width_ratio=0.5")
    
    # Adapt Qwen3-4B-Instruct-2507 model (reduce to 75% depth and width)
    adapted_qwen = adapter_selector.adapt_model('qwen3_4b_instruct_2507', qwen_model, depth_ratio=0.75, width_ratio=0.75)
    print(f"Qwen3-4B-Instruct-2507 model adapted: depth_ratio=0.75, width_ratio=0.75")
    
    # Use ModelIntegrationAdapter for unified interface
    integration_adapter = ModelIntegrationAdapter(adapted_glm, 'glm_4_7_flash')
    print(f"Model wrapped in integration adapter for unified interface")
    
    print()


def demonstrate_integration():
    """
    Demonstrate the integration of all three patterns.
    """
    print("=== Full Integration Demonstration ===")
    
    # Use the integrated approach to create optimized and adapted plugins
    integration = DesignPatternIntegration()
    
    # Define optimization criteria
    optimization_criteria = {
        'available_memory_gb': 16.0,
        'required_memory_gb': 8.0,
        'performance_priority': 'balanced',
        'adaptive': True
    }
    
    # Define architecture ratios
    architecture_ratios = {
        'depth_ratio': 0.8,
        'width_ratio': 0.8
    }
    
    # Create optimized and adapted plugins for all models
    print("Creating optimized and adapted GLM-4.7 plugin...")
    glm_plugin = integration.create_optimized_adapted_plugin(
        'glm_4_7_flash',
        optimization_criteria, 
        architecture_ratios
    )
    print(f"GLM-4.7 plugin created: {glm_plugin.metadata.name}")
    
    print("Creating optimized and adapted Qwen3-4B-Instruct-2507 plugin...")
    qwen4b_plugin = integration.create_optimized_adapted_plugin(
        'qwen3_4b_instruct_2507', 
        optimization_criteria, 
        architecture_ratios
    )
    print(f"Qwen3-4B-Instruct-2507 plugin created: {qwen4b_plugin.metadata.name}")
    
    print("Creating optimized and adapted Qwen3-Coder-30B plugin...")
    qwen30b_plugin = integration.create_optimized_adapted_plugin(
        'qwen3_coder_30b', 
        optimization_criteria, 
        architecture_ratios
    )
    print(f"Qwen3-Coder-30B plugin created: {qwen30b_plugin.metadata.name}")
    
    print("Creating optimized and adapted Qwen3-VL-2B plugin...")
    qwen_vl_plugin = integration.create_optimized_adapted_plugin(
        'qwen3_vl_2b', 
        optimization_criteria, 
        architecture_ratios
    )
    print(f"Qwen3-VL-2B plugin created: {qwen_vl_plugin.metadata.name}")
    
    print()


def demonstrate_convenience_functions():
    """
    Demonstrate the convenience functions for creating specific model plugins.
    """
    print("=== Convenience Functions Demonstration ===")
    
    # Define common optimization criteria
    optimization_criteria = {
        'available_memory_gb': 16.0,
        'required_memory_gb': 8.0,
        'adaptive': True
    }
    
    # Define architecture ratios
    architecture_ratios = {
        'depth_ratio': 0.75,
        'width_ratio': 0.75
    }
    
    # Use convenience functions to create optimized plugins
    print("Creating GLM-4.7 optimized plugin...")
    glm_plugin = create_glm_4_7_optimized_plugin(optimization_criteria, architecture_ratios)
    print(f"GLM-4.7 plugin created: {glm_plugin.metadata.name}")
    
    print("Creating Qwen3-4B-Instruct-2507 optimized plugin...")
    qwen4b_plugin = create_qwen3_4b_instruct_2507_optimized_plugin(optimization_criteria, architecture_ratios)
    print(f"Qwen3-4B-Instruct-2507 plugin created: {qwen4b_plugin.metadata.name}")
    
    print("Creating Qwen3-Coder-30B optimized plugin...")
    qwen30b_plugin = create_qwen3_coder_30b_optimized_plugin(optimization_criteria, architecture_ratios)
    print(f"Qwen3-Coder-30B plugin created: {qwen30b_plugin.metadata.name}")
    
    print("Creating Qwen3-VL-2B optimized plugin...")
    qwen_vl_plugin = create_qwen3_vl_2b_optimized_plugin(optimization_criteria, architecture_ratios)
    print(f"Qwen3-VL-2B plugin created: {qwen_vl_plugin.metadata.name}")
    
    print()


def main():
    """
    Main function to demonstrate all design patterns.
    """
    print("Design Patterns Implementation in Inference-PIO")
    print("=" * 50)
    
    # Demonstrate each pattern individually
    demonstrate_factory_pattern()
    demonstrate_strategy_pattern()
    demonstrate_adapter_pattern()
    
    # Demonstrate full integration
    demonstrate_integration()
    
    # Demonstrate convenience functions
    demonstrate_convenience_functions()
    
    print("All demonstrations completed successfully!")


if __name__ == "__main__":
    main()