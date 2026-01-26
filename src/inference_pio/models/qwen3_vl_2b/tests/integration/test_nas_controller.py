"""
Test suite for the NAS (Neural Architecture Search) controller and model adapters.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from ..nas_controller import (
    ContinuousNASController,
    NASConfig,
    ArchitectureAdaptationStrategy,
    NASMetrics
)
from ..model_adapter import (
    BaseModelAdapter,
    GLM47ModelAdapter,
    Qwen34BInstruct2507ModelAdapter,
    Qwen3Coder30BModelAdapter,
    Qwen3VL2BModelAdapter,
    get_model_adapter
)

class DummyModel(nn.Module):
    """Dummy model for testing purposes."""
    
    def __init__(self, hidden_size=512, num_layers=12):
        super().__init__()
        config = Mock()
        config.hidden_size = hidden_size
        config.num_attention_heads = 8
        config.intermediate_size = hidden_size * 4
        
        # Create dummy transformer layers
        transformer = Mock()
        transformer.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        
        # Create word embeddings
        transformer.word_embeddings = nn.Embedding(1000, hidden_size)
    
    def forward(self, x):
        return x

# TestNASController

    """Test cases for the NAS controller."""
    
    def setup_helper():
        """Set up test fixtures."""
        config = NASConfig(
            strategy=ArchitectureAdaptationStrategy.COMBINED_ADAPTIVE,
            min_depth_ratio=0.3,
            max_depth_ratio=1.0,
            min_width_ratio=0.3,
            max_width_ratio=1.0,
            latency_target_ms=100.0,
            memory_budget_mb=2048.0,
            accuracy_tradeoff_factor=0.7,
            adaptation_frequency=10
        )
        controller = ContinuousNASController(config)
    
    def initialization(self)():
        """Test NAS controller initialization."""
        assert_equal(controller.config.strategy, ArchitectureAdaptationStrategy.COMBINED_ADAPTIVE)
        assert_equal(controller.state.depth_ratio, 1.0)
        assert_equal(controller.state.width_ratio, 1.0)
    
    def should_adapt_by_frequency(self)():
        """Test that adaptation happens based on frequency."""
        # Initially, should not adapt (frequency is 10, count is 0)
        controller.inference_count = 9
        result = controller._should_adapt(complexity=0.5, latency=50.0, memory=1000.0)
        assert_false(result)
        
        # Should adapt when reaching frequency threshold
        controller.inference_count = 10
        result = controller._should_adapt(complexity=0.5)
        assert_true(result)
    
    def should_adapt_by_latency(self)():
        """Test that adaptation happens when latency exceeds target."""
        controller.inference_count = 5  # Below frequency threshold
        result = controller._should_adapt(complexity=0.5)  # Above target
        assert_true(result)
    
    def should_adapt_by_memory(self)():
        """Test that adaptation happens when memory exceeds budget."""
        controller.inference_count = 5  # Below frequency threshold
        result = controller._should_adapt(complexity=0.5)  # Above budget
        assert_true(result)
    
    def calculate_adaptation_ratios_combined_strategy(self)():
        """Test calculation of adaptation ratios for combined strategy."""
        depth)
        # With high complexity, should increase ratios slightly
        assertGreaterEqual(depth, 1.0)
        assertGreaterEqual(width, 1.0)
        
        # With low complexity, should decrease ratios
        depth, width = controller._calculate_adaptation_ratios(
            complexity=0.2,  # Low complexity
            latency=80.0,    # Within target
            memory=1800.0    # Within budget
        )
        # Ratios should be lower but still within bounds
        assertGreaterEqual(depth, 0.3)
        assertGreaterEqual(width, 0.3)
    
    def calculate_adaptation_ratios_latency_based_strategy(self)():
        """Test calculation of adaptation ratios for latency-based strategy."""
        # Change strategy to latency-based
        controller.config.strategy = ArchitectureAdaptationStrategy.LATENCY_BASED
        
        # With high latency, should reduce ratios
        depth, width = controller._calculate_adaptation_ratios(
            complexity=0.5,
            latency=200.0,  # Much higher than target
            memory=1000.0
        )
        assert_less(depth, 1.0)
        assert_less(width, 1.0)
        
        # With low latency, should increase ratios
        depth, width = controller._calculate_adaptation_ratios(
            complexity=0.5,
            latency=20.0,   # Much lower than target
            memory=1000.0
        )
        assertGreaterEqual(depth, 1.0)
        assertGreaterEqual(width, 1.0)
    
    def calculate_adaptation_ratios_memory_based_strategy(self)():
        """Test calculation of adaptation ratios for memory-based strategy."""
        # Change strategy to memory-based
        controller.config.strategy = ArchitectureAdaptationStrategy.MEMORY_BASED
        
        # With high memory usage, should reduce ratios
        depth, width = controller._calculate_adaptation_ratios(
            complexity=0.5,
            latency=50.0,
            memory=3000.0  # Much higher than budget
        )
        assert_less(depth, 1.0)
        assert_less(width, 1.0)
        
        # With low memory usage, should increase ratios
        depth, width = controller._calculate_adaptation_ratios(
            complexity=0.5,
            latency=50.0,
            memory=500.0   # Much lower than budget
        )
        assertGreaterEqual(depth, 1.0)
        assertGreaterEqual(width, 1.0)
    
    def estimate_accuracy_preservation(self)():
        """Test accuracy preservation estimation."""
        # With full architecture (ratios = 1.0), should preserve most accuracy
        controller.state.depth_ratio = 1.0
        controller.state.width_ratio = 1.0
        accuracy = controller._estimate_accuracy_preservation()
        assertGreaterEqual(accuracy, 0.9)
        
        # With reduced architecture, should have lower accuracy preservation
        controller.state.depth_ratio = 0.5
        controller.state.width_ratio = 0.5
        accuracy = controller._estimate_accuracy_preservation()
        assert_less(accuracy, 1.0)
        assertGreaterEqual(accuracy, 0.5)

# TestModelAdapters

    """Test cases for model adapters."""
    
    def setup_helper():
        """Set up test fixtures."""
        dummy_model = DummyModel()
        nas_controller = Mock()
    
    def base_model_adapter(self)():
        """Test base model adapter functionality."""
        adapter = BaseModelAdapter(dummy_model, nas_controller)
        
        # Test that methods return the original model by default
        result = adapter.adapt_depth(0.8)
        assertIs(result, dummy_model)
        
        result = adapter.adapt_width(0.8)
        assertIs(result, dummy_model)
        
        result = adapter.adapt_architecture(0.8, 0.8)
        assertIs(result, dummy_model)
    
    def glm47_model_adapter_capture_architecture(self)():
        """Test GLM-4.7 model adapter architecture capture."""
        adapter = GLM47ModelAdapter(dummy_model, nas_controller)
        arch_info = adapter._capture_original_architecture()
        
        assert_in('num_layers', arch_info)
        assert_in('hidden_size', arch_info)
        assert_equal(arch_info['num_layers'], 12)
        assert_equal(arch_info['hidden_size'], 512)
    
    def qwen3_4b_model_adapter_capture_architecture(self)():
        """Test Qwen3-4B model adapter architecture capture."""
        adapter = Qwen34BInstruct2507ModelAdapter(dummy_model, nas_controller)
        arch_info = adapter._capture_original_architecture()
        
        assert_in('num_layers', arch_info)
        assert_in('hidden_size', arch_info)
        assert_equal(arch_info['num_layers'], 12)
        assert_equal(arch_info['hidden_size'], 512)
    
    def qwen3_coder_model_adapter_capture_architecture(self)():
        """Test Qwen3-Coder model adapter architecture capture."""
        adapter = Qwen3Coder30BModelAdapter(dummy_model, nas_controller)
        arch_info = adapter._capture_original_architecture()
        
        assert_in('num_layers', arch_info)
        assert_in('hidden_size', arch_info)
        assert_equal(arch_info['num_layers'], 12)
        assert_equal(arch_info['hidden_size'], 512)
    
    def qwen3_vl_model_adapter_capture_architecture(self)():
        """Test Qwen3-VL model adapter architecture capture."""
        adapter = Qwen3VL2BModelAdapter(dummy_model, nas_controller)
        arch_info = adapter._capture_original_architecture()
        
        assert_in('num_layers', arch_info)
        assert_in('hidden_size', arch_info)
        assert_equal(arch_info['num_layers'], 12)
        assert_equal(arch_info['hidden_size'], 512)
    
    def get_model_adapter_factory(self)():
        """Test the model adapter factory function."""
        # Test with a model that looks like GLM
        glm_model = Mock()
        type(glm_model).__name__ = "GLM47Model"
        adapter = get_model_adapter(glm_model, nas_controller)
        assert_is_instance(adapter, BaseModelAdapter)  # Will default to base adapter
        
        # Test with a model that looks like Qwen3-4B
        qwen4b_model = Mock()
        type(qwen4b_model).__name__ = "Qwen34BInstruct2507Model"
        adapter = get_model_adapter(qwen4b_model, nas_controller)
        assert_is_instance(adapter, BaseModelAdapter)  # Will default to base adapter
        
        # Test with unknown model type (should return base adapter)
        unknown_model = Mock()
        type(unknown_model).__name__ = "UnknownModel"
        adapter = get_model_adapter(unknown_model, nas_controller)
        assert_is_instance(adapter, BaseModelAdapter)

# TestIntegration

    """Integration tests for NAS controller with model adapters."""
    
    def setup_helper():
        """Set up test fixtures."""
        config = NASConfig(
            strategy=ArchitectureAdaptationStrategy.COMBINED_ADAPTIVE,
            min_depth_ratio=0.3,
            max_depth_ratio=1.0,
            min_width_ratio=0.3,
            max_width_ratio=1.0,
            latency_target_ms=100.0,
            memory_budget_mb=2048.0,
            accuracy_tradeoff_factor=0.7,
            adaptation_frequency=10
        )
        controller = ContinuousNASController(config)
        dummy_model = DummyModel()
    
    def adapt_architecture_integration(self)():
        """Test the full architecture adaptation process."""
        # Create a real input tensor for testing
        input_tensor = torch.randn(2, 10, 512)

        # Create an adapter
        adapter = GLM47ModelAdapter(dummy_model, controller)

        # Test that the controller can adapt the model
        adapted_model, metrics = controller.adapt_architecture(
            dummy_model, input_tensor
        )

        # Check that the result is valid
        assert_is_instance(adapted_model, nn.Module)
        assert_is_instance(metrics, NASMetrics)
        assertGreaterEqual(metrics.input_complexity, 0.0)
        assertGreaterEqual(metrics.processing_time_ms, 0.0)
        assertGreaterEqual(metrics.memory_used_mb, 0.0)

if __name__ == '__main__':
    run_tests(test_functions)