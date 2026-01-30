"""
Tests for Design Patterns Implementation in Inference-PIO

This module contains comprehensive tests for the Factory, Strategy, and Adapter patterns.
"""

import torch
import torch.nn as nn
from tests.utils.test_utils import (
    assert_is_instance,
    assert_is_not_none,
    assert_equal,
    assert_true,
    assert_raises,
    run_tests
)

from ..models.glm_4_7_flash.plugin import GLM_4_7_Flash_Plugin
from ..models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from ..models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from ..models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin
from .factory import (
    PluginFactoryProvider,
    OptimizationStrategyFactoryProvider,
    ModelAdapterFactoryProvider,
    GLM47PluginFactory,
    Qwen34BInstruct2507PluginFactory,
    Qwen3Coder30BPluginFactory,
    Qwen3VL2BPluginFactory
)
from .strategy import (
    MemoryOptimizationStrategy,
    ComputeOptimizationStrategy,
    AdaptiveOptimizationStrategy,
    OptimizationSelector
)
from .adapter import (
    GLM47ModelAdapter,
    Qwen34BInstruct2507ModelAdapter,
    Qwen3Coder30BModelAdapter,
    Qwen3VL2BModelAdapter,
    ModelAdapterSelector
)
from .integration import (
    DesignPatternIntegration,
    create_optimized_plugin,
    create_adapted_model,
    create_optimized_adapted_plugin
)


def test_glm_4_7_plugin_factory():
    """Test GLM-4.7 plugin factory."""
    factory = GLM47PluginFactory()
    plugin = factory.create_plugin()
    assert_is_instance(plugin, GLM_4_7_Plugin)


def test_qwen3_4b_instruct_2507_plugin_factory():
    """Test Qwen3-4B-Instruct-2507 plugin factory."""
    factory = Qwen34BInstruct2507PluginFactory()
    plugin = factory.create_plugin()
    assert_is_instance(plugin, Qwen3_4B_Instruct_2507_Plugin)


def test_qwen3_coder_30b_plugin_factory():
    """Test Qwen3-Coder-30B plugin factory."""
    factory = Qwen3Coder30BPluginFactory()
    plugin = factory.create_plugin()
    assert_is_instance(plugin, Qwen3_Coder_30B_Plugin)


def test_qwen3_vl_2b_plugin_factory():
    """Test Qwen3-VL-2B plugin factory."""
    factory = Qwen3VL2BPluginFactory()
    plugin = factory.create_plugin()
    assert_is_instance(plugin, Qwen3_VL_2B_Instruct_Plugin)


def test_plugin_factory_provider():
    """Test plugin factory provider."""
    factory_provider = PluginFactoryProvider()
    plugin_types = ['glm_4_7_flash', 'qwen3_4b_instruct_2507', 'qwen3_coder_30b', 'qwen3_vl_2b']

    for model_type in plugin_types:
        plugin = factory_provider.create_plugin(model_type)
        assert_is_not_none(plugin)


def test_invalid_model_type():
    """Test invalid model type raises error."""
    factory_provider = PluginFactoryProvider()
    try:
        factory_provider.create_plugin('invalid_model_type')
        assert_true(False, "Expected ValueError for invalid model type")
    except ValueError:
        pass  # Expected behavior


def test_memory_optimization_strategy():
    """Test memory optimization strategy."""
    # Create a simple mock model for testing
    mock_model = nn.Linear(10, 5)

    strategy = MemoryOptimizationStrategy()
    optimized_model = strategy.optimize(mock_model)
    assert_is_not_none(optimized_model)
    assert_equal(strategy.get_strategy_name(), "Memory Optimization Strategy")


def test_compute_optimization_strategy():
    """Test compute optimization strategy."""
    # Create a simple mock model for testing
    mock_model = nn.Linear(10, 5)

    strategy = ComputeOptimizationStrategy()
    optimized_model = strategy.optimize(mock_model)
    assert_is_not_none(optimized_model)
    assert_equal(strategy.get_strategy_name(), "Compute Optimization Strategy")


def test_adaptive_optimization_strategy():
    """Test adaptive optimization strategy."""
    # Create a simple mock model for testing
    mock_model = nn.Linear(10, 5)

    strategy = AdaptiveOptimizationStrategy()
    optimized_model = strategy.optimize(mock_model)
    assert_is_not_none(optimized_model)
    assert_equal(strategy.get_strategy_name(), "Adaptive Optimization Strategy")


def test_optimization_selector():
    """Test optimization selector."""
    # Create a simple mock model for testing
    mock_model = nn.Linear(10, 5)

    selector = OptimizationSelector()
    criteria = {'available_memory_gb': 16.0, 'required_memory_gb': 8.0}

    # Test with high memory availability (should select compute strategy)
    strategy = selector.select_strategy(mock_model, criteria)
    assert_is_not_none(strategy)

    # Test with low memory availability (should select memory strategy)
    low_memory_criteria = {'available_memory_gb': 2.0, 'required_memory_gb': 8.0}
    strategy = selector.select_strategy(mock_model, low_memory_criteria)
    assert_is_not_none(strategy)


def test_optimize_with_criteria():
    """Test optimization with criteria."""
    # Create a simple mock model for testing
    mock_model = nn.Linear(10, 5)

    selector = OptimizationSelector()
    criteria = {'available_memory_gb': 16.0, 'required_memory_gb': 8.0}

    optimized_model = selector.optimize_with_criteria(mock_model, criteria)
    assert_is_not_none(optimized_model)


def test_glm_4_7_model_adapter():
    """Test GLM-4.7 model adapter."""
    # Create simple mock models for testing
    glm_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    adapter = GLM47ModelAdapter(glm_model)
    adapted_model = adapter.adapt_depth(0.5)
    assert_is_not_none(adapted_model)

    adapted_model = adapter.adapt_width(0.5)
    assert_is_not_none(adapted_model)


def test_qwen3_4b_instruct_2507_model_adapter():
    """Test Qwen3-4B-Instruct-2507 model adapter."""
    # Create simple mock models for testing
    qwen_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    adapter = Qwen34BInstruct2507ModelAdapter(qwen_model)
    adapted_model = adapter.adapt_depth(0.5)
    assert_is_not_none(adapted_model)

    adapted_model = adapter.adapt_width(0.5)
    assert_is_not_none(adapted_model)


def test_qwen3_coder_30b_model_adapter():
    """Test Qwen3-Coder-30B model adapter."""
    # Create simple mock models for testing
    qwen_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    adapter = Qwen3Coder30BModelAdapter(qwen_model)
    adapted_model = adapter.adapt_depth(0.5)
    assert_is_not_none(adapted_model)

    adapted_model = adapter.adapt_width(0.5)
    assert_is_not_none(adapted_model)


def test_qwen3_vl_2b_model_adapter():
    """Test Qwen3-VL-2B model adapter."""
    # Create simple mock models for testing
    vision_model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(16, 5))

    adapter = Qwen3VL2BModelAdapter(vision_model)
    adapted_model = adapter.adapt_depth(0.5)
    assert_is_not_none(adapted_model)

    adapted_model = adapter.adapt_width(0.5)
    assert_is_not_none(adapted_model)


def test_model_adapter_selector():
    """Test model adapter selector."""
    selector = ModelAdapterSelector()

    # Create simple mock models for testing
    glm_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    qwen_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    vision_model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(16, 5))

    # Test GLM-4.7 adapter selection
    adapter = selector.select_adapter('glm_4_7_flash', glm_model)
    assert_is_instance(adapter, GLM47ModelAdapter)

    # Test Qwen3-4B-Instruct-2507 adapter selection
    adapter = selector.select_adapter('qwen3_4b_instruct_2507', qwen_model)
    assert_is_instance(adapter, Qwen34BInstruct2507ModelAdapter)

    # Test Qwen3-Coder-30B adapter selection
    adapter = selector.select_adapter('qwen3_coder_30b', qwen_model)
    assert_is_instance(adapter, Qwen3Coder30BModelAdapter)

    # Test Qwen3-VL-2B adapter selection
    adapter = selector.select_adapter('qwen3_vl_2b', vision_model)
    assert_is_instance(adapter, Qwen3VL2BModelAdapter)


def test_adapt_model():
    """Test model adaptation."""
    selector = ModelAdapterSelector()

    # Create simple mock models for testing
    glm_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    adapted_model = selector.adapt_model('glm_4_7_flash', glm_model, depth_ratio=0.5, width_ratio=0.5)
    assert_is_not_none(adapted_model)


def test_create_adapted_model():
    """Test creating adapted model."""
    integration = DesignPatternIntegration()
    mock_model = nn.Linear(10, 5)
    adapter = integration.create_adapted_model('glm_4_7_flash', mock_model, 0.5, 0.5)
    assert_is_not_none(adapter)


def test_convenience_functions_exist():
    """Test that convenience functions exist."""
    assert_true(callable(create_optimized_plugin))
    assert_true(callable(create_adapted_model))
    assert_true(callable(create_optimized_adapted_plugin))


def test_model_specific_functions():
    """Test model-specific convenience functions."""
    from .integration import (
        create_glm_4_7_optimized_plugin,
        create_qwen3_4b_instruct_2507_optimized_plugin,
        create_qwen3_coder_30b_optimized_plugin,
        create_qwen3_vl_2b_optimized_plugin
    )

    assert_true(callable(create_glm_4_7_optimized_plugin))
    assert_true(callable(create_qwen3_4b_instruct_2507_optimized_plugin))
    assert_true(callable(create_qwen3_coder_30b_optimized_plugin))
    assert_true(callable(create_qwen3_vl_2b_optimized_plugin))


if __name__ == '__main__':
    # Run the tests using custom test utilities
    test_functions = [
        test_glm_4_7_plugin_factory,
        test_qwen3_4b_instruct_2507_plugin_factory,
        test_qwen3_coder_30b_plugin_factory,
        test_qwen3_vl_2b_plugin_factory,
        test_plugin_factory_provider,
        test_invalid_model_type,
        test_memory_optimization_strategy,
        test_compute_optimization_strategy,
        test_adaptive_optimization_strategy,
        test_optimization_selector,
        test_optimize_with_criteria,
        test_glm_4_7_model_adapter,
        test_qwen3_4b_instruct_2507_model_adapter,
        test_qwen3_coder_30b_model_adapter,
        test_qwen3_vl_2b_model_adapter,
        test_model_adapter_selector,
        test_adapt_model,
        test_create_adapted_model,
        test_convenience_functions_exist,
        test_model_specific_functions
    ]
    run_tests(test_functions)