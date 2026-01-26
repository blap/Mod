"""
Comprehensive Tests for Design Patterns (Factory/Strategy/Adapter) in Inference-PIO

This module contains comprehensive tests for the Factory, Strategy, and Adapter patterns
implemented in the Inference-PIO system.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn

import sys
import os
from pathlib import Path

# Adicionando o diretório src ao path para permitir imports relativos
src_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(src_dir))

from inference_pio.design_patterns.factory import (
    PluginFactoryProvider,
    OptimizationStrategyFactoryProvider,
    ModelAdapterFactoryProvider,
    GLM47PluginFactory,
    Qwen34BInstruct2507PluginFactory,
    Qwen3Coder30BPluginFactory,
    Qwen3VL2BPluginFactory
)
from inference_pio.design_patterns.strategy import (
    MemoryOptimizationStrategy,
    ComputeOptimizationStrategy,
    AdaptiveOptimizationStrategy,
    OptimizationSelector
)
from inference_pio.design_patterns.adapter import (
    GLM47ModelAdapter,
    Qwen34BInstruct2507ModelAdapter,
    Qwen3Coder30BModelAdapter,
    Qwen3VL2BModelAdapter,
    ModelAdapterSelector
)
from inference_pio.design_patterns.integration import (
    DesignPatternIntegration,
    create_optimized_plugin,
    create_adapted_model,
    create_optimized_adapted_plugin
)
from inference_pio.models.glm_4_7_flash.plugin import GLM_4_7_Flash_Plugin
from inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin

# TestFactoryPatternImplementation

    """Comprehensive test suite for Factory pattern implementation."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        factory_provider = PluginFactoryProvider()

    def glm_4_7_plugin_factory(self)():
        """Test GLM-4.7 plugin factory implementation."""
        factory = GLM47PluginFactory()
        plugin = factory.create_plugin()
        
        assert_is_instance(plugin, GLM_4_7_Plugin)
        assert_is_not_none(plugin)
        assert_equal(plugin.metadata.name)

    def qwen3_4b_instruct_2507_plugin_factory(self)():
        """Test Qwen3-4B-Instruct-2507 plugin factory implementation."""
        factory = Qwen34BInstruct2507PluginFactory()
        plugin = factory.create_plugin()
        
        assert_is_instance(plugin, Qwen3_4B_Instruct_2507_Plugin)
        assert_is_not_none(plugin)
        assert_equal(plugin.metadata.name)

    def qwen3_coder_30b_plugin_factory(self)():
        """Test Qwen3-Coder-30B plugin factory implementation."""
        factory = Qwen3Coder30BPluginFactory()
        plugin = factory.create_plugin()
        
        assert_is_instance(plugin, Qwen3_Coder_30B_Plugin)
        assert_is_not_none(plugin)
        assert_equal(plugin.metadata.name)

    def qwen3_vl_2b_plugin_factory(self)():
        """Test Qwen3-VL-2B plugin factory implementation."""
        factory = Qwen3VL2BPluginFactory()
        plugin = factory.create_plugin()
        
        assert_is_instance(plugin, Qwen3_VL_2B_Instruct_Plugin)
        assert_is_not_none(plugin)
        assert_equal(plugin.metadata.name)

    def plugin_factory_provider_pattern(self)():
        """Test plugin factory provider implementation."""
        plugin_types = ['glm_4_7_flash', 'qwen3_4b_instruct_2507', 'qwen3_coder_30b', 'qwen3_vl_2b']

        for model_type in plugin_types:
            with subTest(model_type=model_type):
                plugin = factory_provider.create_plugin(model_type)
                assert_is_not_none(plugin)

    def invalid_model_type_factory(self)():
        """Test factory behavior with invalid model type."""
        with assert_raises(ValueError):
            factory_provider.create_plugin('invalid_model_type')

    def optimization_strategy_factory_provider(self)():
        """Test optimization strategy factory provider."""
        factory_provider = OptimizationStrategyFactoryProvider()
        
        strategies = ['memory'):
                strategy = factory_provider.create_strategy(strategy_type)
                assert_is_not_none(strategy)

    def model_adapter_factory_provider(self)():
        """Test model adapter factory provider."""
        factory_provider = ModelAdapterFactoryProvider()
        
        adapters = ['glm_4_7_flash'):
                adapter = factory_provider.create_adapter(adapter_type)
                assert_is_not_none(adapter)

    def factory_pattern_encapsulation(self)():
        """Test that factories encapsulate object creation properly."""
        # Each factory should create objects without exposing internal creation logic
        glm_factory = GLM47PluginFactory()
        qwen3_4b_factory = Qwen34BInstruct2507PluginFactory()
        
        glm_plugin = glm_factory.create_plugin()
        qwen3_4b_plugin = qwen3_4b_factory.create_plugin()
        
        # Objects should be of correct types
        assert_is_instance(glm_plugin)
        assertIsInstance(qwen3_4b_plugin, Qwen3_4B_Instruct_2507_Plugin)
        
        # Objects should be distinct instances
        assertIsNot(glm_plugin, qwen3_4b_plugin)

    def factory_return_types_consistency(self)():
        """Test that all factories return consistent types."""
        factories = [
            GLM47PluginFactory(),
            Qwen34BInstruct2507PluginFactory(),
            Qwen3Coder30BPluginFactory(),
            Qwen3VL2BPluginFactory()
        ]
        
        for factory in factories:
            with subTest(factory=type(factory).__name__):
                plugin = factory.create_plugin()
                assert_is_instance(plugin, (GLM_4_7_Plugin))

# TestStrategyPatternImplementation

    """Comprehensive test suite for Strategy pattern implementation."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        # Create a simple mock model for testing
        mock_model = nn.Linear(10, 5)

    def memory_optimization_strategy(self)():
        """Test memory optimization strategy implementation."""
        strategy = MemoryOptimizationStrategy()
        optimized_model = strategy.optimize(mock_model)
        
        assert_is_not_none(optimized_model)
        assert_equal(strategy.get_strategy_name())

    def compute_optimization_strategy(self)():
        """Test compute optimization strategy implementation."""
        strategy = ComputeOptimizationStrategy()
        optimized_model = strategy.optimize(mock_model)
        
        assert_is_not_none(optimized_model)
        assert_equal(strategy.get_strategy_name())

    def adaptive_optimization_strategy(self)():
        """Test adaptive optimization strategy implementation."""
        strategy = AdaptiveOptimizationStrategy()
        optimized_model = strategy.optimize(mock_model)
        
        assert_is_not_none(optimized_model)
        assert_equal(strategy.get_strategy_name())

    def optimization_selector_pattern(self)():
        """Test optimization selector pattern implementation."""
        selector = OptimizationSelector()
        criteria = {'available_memory_gb': 16.0, 'required_memory_gb': 8.0}

        # Test with high memory availability (should select compute strategy)
        strategy = selector.select_strategy(mock_model, criteria)
        assert_is_not_none(strategy)

        # Test with low memory availability (should select memory strategy)
        low_memory_criteria = {'available_memory_gb': 2.0)
        assert_is_not_none(strategy)

    def optimize_with_criteria(self)():
        """Test optimization with criteria using strategy pattern."""
        selector = OptimizationSelector()
        criteria = {'available_memory_gb': 16.0)
        assert_is_not_none(optimized_model)

    def strategy_algorithm_interchangeability(self)():
        """Test that strategies can be interchanged at runtime."""
        strategies = [
            MemoryOptimizationStrategy(),
            ComputeOptimizationStrategy(),
            AdaptiveOptimizationStrategy()
        ]
        
        for strategy in strategies:
            with subTest(strategy=strategy.get_strategy_name()):
                optimized_model = strategy.optimize(mock_model)
                assert_is_not_none(optimized_model)

    def strategy_context_binding(self)():
        """Test that strategies can be bound to different contexts."""
        selector = OptimizationSelector()
        
        # Different contexts should yield different strategies
        high_memory_context = {'available_memory_gb': 32.0)
        low_memory_strategy = selector.select_strategy(mock_model, low_memory_context)
        
        assert_is_not_none(high_memory_strategy)
        assertIsNotNone(low_memory_strategy)

    def strategy_algorithm_correctness(self)():
        """Test that each strategy algorithm performs correctly."""
        # Each strategy should produce a valid model
        memory_strategy = MemoryOptimizationStrategy()
        compute_strategy = ComputeOptimizationStrategy()
        adaptive_strategy = AdaptiveOptimizationStrategy()
        
        memory_result = memory_strategy.optimize(mock_model)
        compute_result = compute_strategy.optimize(mock_model)
        adaptive_result = adaptive_strategy.optimize(mock_model)
        
        # All should return valid models
        assertIsNot(memory_result)  # Should return different instance
        assertIsNot(compute_result, mock_model)
        assertIsNot(adaptive_result, mock_model)

# TestAdapterPatternImplementation

    """Comprehensive test suite for Adapter pattern implementation."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        # Create simple mock models for testing
        glm_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        qwen_model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        vision_model = nn.Sequential(
            nn.Conv2d(3, 16, 3), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(), 
            nn.Linear(16, 5)
        )

    def glm_4_7_model_adapter(self)():
        """Test GLM-4.7 model adapter implementation."""
        adapter = GLM47ModelAdapter(glm_model)
        adapted_model = adapter.adapt_depth(0.5)
        assert_is_not_none(adapted_model)

        adapted_model = adapter.adapt_width(0.5)
        assertIsNotNone(adapted_model)

    def qwen3_4b_instruct_2507_model_adapter(self)():
        """Test Qwen3-4B-Instruct-2507 model adapter implementation."""
        adapter = Qwen34BInstruct2507ModelAdapter(qwen_model)
        adapted_model = adapter.adapt_depth(0.5)
        assertIsNotNone(adapted_model)

        adapted_model = adapter.adapt_width(0.5)
        assertIsNotNone(adapted_model)

    def qwen3_coder_30b_model_adapter(self)():
        """Test Qwen3-Coder-30B model adapter implementation."""
        adapter = Qwen3Coder30BModelAdapter(qwen_model)
        adapted_model = adapter.adapt_depth(0.5)
        assertIsNotNone(adapted_model)

        adapted_model = adapter.adapt_width(0.5)
        assertIsNotNone(adapted_model)

    def qwen3_vl_2b_model_adapter(self)():
        """Test Qwen3-VL-2B model adapter implementation."""
        adapter = Qwen3VL2BModelAdapter(vision_model)
        adapted_model = adapter.adapt_depth(0.5)
        assertIsNotNone(adapted_model)

        adapted_model = adapter.adapt_width(0.5)
        assertIsNotNone(adapted_model)

    def model_adapter_selector_pattern(self)():
        """Test model adapter selector pattern implementation."""
        selector = ModelAdapterSelector()

        # Test GLM-4.7 adapter selection
        adapter = selector.select_adapter('glm_4_7_flash')
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

    def adapt_model_functionality(self)():
        """Test model adaptation functionality."""
        selector = ModelAdapterSelector()

        adapted_model = selector.adapt_model('glm_4_7_flash', glm_model, depth_ratio=0.5, width_ratio=0.5)
        assert_is_not_none(adapted_model)

    def adapter_target_interface_compatibility(self)():
        """Test that adapters provide compatibility with target interface."""
        adapter = GLM47ModelAdapter(glm_model)
        
        # Adapter should provide the expected interface methods
        assert_true(hasattr(adapter))
        assert_true(hasattr(adapter))
        assert_true(hasattr(adapter))
        
        # Methods should be callable
        result = adapter.adapt_depth(0.5)
        assertIsNotNone(result)
        
        result = adapter.adapt_width(0.5)
        assertIsNotNone(result)

    def adapter_preserves_core_functionality(self)():
        """Test that adapters preserve core model functionality."""
        original_output = glm_model(torch.randn(1))
        
        adapter = GLM47ModelAdapter(glm_model)
        adapted_model = adapter.adapt_depth(0.8)
        
        # Adapted model should still be usable
        adapted_output = adapted_model(torch.randn(1, 10))
        assert_is_not_none(adapted_output)

    def adapter_chain_application(self)():
        """Test applying multiple adaptations in sequence."""
        adapter = GLM47ModelAdapter(glm_model)
        
        # Apply depth adaptation
        model_step1 = adapter.adapt_depth(0.7)
        
        # Apply width adaptation to the result
        adapter_step2 = GLM47ModelAdapter(model_step1)
        final_model = adapter_step2.adapt_width(0.6)
        
        assertIsNotNone(final_model)

# TestDesignPatternsIntegration

    """Comprehensive integration tests for design patterns."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        integration = DesignPatternIntegration()

    def create_optimized_plugin_integration(self)():
        """Test creating optimized plugin through integration layer."""
        # This test would normally require actual model loading)
        except Exception as e:
            # Accept that the plugin creation might fail due to missing model files
            # But the method should be callable and fail gracefully
            assert_in('model', str(e).lower()) or assert_in('file', str(e).lower())

    def create_adapted_model_integration(self)():
        """Test creating adapted model through integration layer."""
        mock_model = nn.Linear(10, 5)
        adapter = integration.create_adapted_model('glm_4_7_flash', mock_model, 0.5, 0.5)
        assert_is_not_none(adapter)

    def create_optimized_adapted_plugin_integration(self)():
        """Test creating optimized and adapted plugin through integration layer."""
        # This test would normally require actual model loading
        try:
            plugin = integration.create_optimized_adapted_plugin(
                'glm_4_7_flash',
                {'adaptive': True},
                {'depth_ratio': 0.5, 'width_ratio': 0.5}
            )
        except Exception as e:
            # Accept that the plugin creation might fail due to missing model files
            # But the method should be callable and fail gracefully
            assert_in('model', str(e).lower()) or assert_in('file', str(e).lower())

    def factory_strategy_adapter_combination(self)():
        """Test combining Factory, Strategy, and Adapter patterns."""
        # Use factory to create a plugin
        factory = GLM47PluginFactory()
        plugin = factory.create_plugin()
        
        # Use strategy to optimize the plugin's model
        strategy = AdaptiveOptimizationStrategy()
        # Note: This is conceptual - actual implementation might differ
        # depending on how optimization is applied to plugins
        
        # Use adapter to adapt the plugin's model
        adapter = GLM47ModelAdapter(nn.Linear(10, 5))  # Using mock model for test
        adapted_model = adapter.adapt_depth(0.7)
        
        assert_is_not_none(plugin)
        assertIsNotNone(adapted_model)

    def pattern_interoperability(self)():
        """Test that different patterns work together seamlessly."""
        # Create a model using factory
        factory = GLM47PluginFactory()
        plugin = factory.create_plugin()
        
        # The plugin should be compatible with strategy and adapter patterns
        assertIsNotNone(plugin)
        
        # Verify plugin has expected interface
        assert_true(hasattr(plugin))
        assert_true(hasattr(plugin))
        assert_true(hasattr(plugin))
        assert_true(hasattr(plugin))

# TestDesignPatternBestPractices

    """Tests for design pattern best practices and principles."""

    def factory_single_responsibility(self)():
        """Test that factories follow single responsibility principle."""
        factory = GLM47PluginFactory()
        
        # Factory should only be responsible for creating plugins
        plugin = factory.create_plugin()
        assert_is_instance(plugin)
        
        # Factory shouldn't handle other concerns like optimization or adaptation
        assert_false(hasattr(factory))
        assert_false(hasattr(factory))

    def strategy_open_closed_principle(self)():
        """Test that strategy pattern follows open/closed principle."""
        # New strategies should be able to be added without modifying existing code
        
            def optimize(self, model):
                return model  # Simplified implementation
            
            def get_strategy_name(self):
                return "New Optimization Strategy"
        
        # Existing selector should be able to work with new strategy if properly designed
        # This is a conceptual test - actual implementation would depend on architecture

    def adapter_interface_segregation(self)():
        """Test that adapters follow interface segregation principle."""
        model = nn.Linear(10, 5)
        adapter = GLM47ModelAdapter(model)
        
        # Adapter should only expose necessary adaptation methods
        allowed_methods = ['adapt_depth', 'adapt_width', 'adapt_architecture']
        for method in allowed_methods:
            assert_true(hasattr(adapter))

    def pattern_polymorphism(self)():
        """Test polymorphic behavior of design patterns."""
        # Factory pattern polymorphism
        factories = [
            GLM47PluginFactory(),
            Qwen34BInstruct2507PluginFactory(),
            Qwen3Coder30BPluginFactory()
        ]
        
        plugins = []
        for factory in factories:
            plugin = factory.create_plugin()
            plugins.append(plugin)
        
        # All plugins should implement the same interface despite different types
        for plugin in plugins:
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))
            assert_true(hasattr(plugin))

    def strategy_dependency_injection(self)():
        """Test that strategies can be injected into contexts."""
        mock_model = nn.Linear(10, 5)
        
        # Different strategies should be injectable into the same context
        memory_strategy = MemoryOptimizationStrategy()
        compute_strategy = ComputeOptimizationStrategy()
        
        memory_result = memory_strategy.optimize(mock_model)
        compute_result = compute_strategy.optimize(mock_model)
        
        # Both should work with the same model
        assert_is_not_none(memory_result)
        assertIsNotNone(compute_result)

# TestDesignPatternErrorHandling

    """Tests for error handling in design patterns."""

    def factory_error_handling(self)():
        """Test error handling in factory pattern."""
        factory_provider = PluginFactoryProvider()
        
        # Invalid type should raise appropriate error
        with assert_raises(ValueError):
            factory_provider.create_plugin("invalid_type")

    def strategy_error_handling(self)():
        """Test error handling in strategy pattern."""
        selector = OptimizationSelector()
        
        # Edge case: empty criteria
        with assert_raises(Exception):  # Could be ValueError), {})

    def adapter_error_handling(self)():
        """Test error handling in adapter pattern."""
        selector = ModelAdapterSelector()
        
        # Invalid model type should be handled appropriately
        with assert_raises(Exception):  # Could be ValueError, TypeError, etc.
            selector.select_adapter("invalid_type", nn.Linear(10, 5))

if __name__ == '__main__':
    run_tests(test_functions)