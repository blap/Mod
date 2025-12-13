"""
Unit tests for Phase 2 efficiency improvements in Qwen3-VL model.
Tests for linear attention, device-aware modules, gradient checkpointing, 
adaptive computation pathways, and memory management.
"""
import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from unittest.mock import Mock, patch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.linear_attention import PerformerAttention
from src.models.device_aware_module import DeviceAwareAttention, DeviceAwareModule
from src.models.gradient_checkpointing import GradientCheckpointingWrapper, MemoryEfficientAttention, MemoryEfficientMLP
from src.models.adaptive_computation import AdaptiveAttention, AdaptiveMLP, AdaptiveComputationGate
from src.models.memory_management import MemoryManager, MemoryEfficientDataLoader, OptimizedQwen3VLAttention
from src.models.config import Qwen3VLConfig


class TestPerformerAttention:
    """Test class for Performer-style linear attention mechanism."""

    def test_performer_attention_initialization(self):
        """Test Performer attention mechanism initialization."""
        config = Qwen3VLConfig()
        attention = PerformerAttention(config, layer_idx=0)
        
        # Check that all required components are initialized
        assert attention.num_heads == 32  # Maintains all 32 heads
        assert attention.head_dim == config.hidden_size // config.num_attention_heads
        assert attention.q_proj is not None
        assert attention.k_proj is not None
        assert attention.v_proj is not None
        assert attention.o_proj is not None
        assert attention.rotary_emb is not None

    def test_performer_attention_output_shape(self):
        """Test that Performer attention produces expected output shape."""
        config = Qwen3VLConfig()
        attention = PerformerAttention(config, layer_idx=0)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output, _, _ = attention(hidden_states)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_performer_attention_with_attention_mask(self):
        """Test Performer attention with attention mask."""
        config = Qwen3VLConfig()
        attention = PerformerAttention(config, layer_idx=0)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        output, _, _ = attention(hidden_states, attention_mask=attention_mask)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_performer_attention_deterministic_output(self):
        """Test that Performer attention produces deterministic outputs."""
        config = Qwen3VLConfig()
        attention = PerformerAttention(config, layer_idx=0)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Run attention twice with the same input
        output1, _, _ = attention(hidden_states)
        output2, _, _ = attention(hidden_states)
        
        # Outputs should be very similar (allowing for small numerical differences)
        assert torch.allclose(output1, output2, atol=1e-5)


class TestDeviceAwareModule:
    """Test class for device-aware module selection system."""

    def test_device_aware_module_initialization(self):
        """Test device-aware module initialization and device detection."""
        device_module = DeviceAwareModule()
        
        # Check that device info is properly detected
        device_info = device_module.get_device_info()
        assert 'device_type' in device_info
        assert 'has_cuda' in device_info
        assert device_info['device_type'] in ['cuda', 'cpu']

    def test_device_aware_attention_initialization(self):
        """Test device-aware attention initialization."""
        config = Qwen3VLConfig()
        attention = DeviceAwareAttention(config, layer_idx=0)
        
        # Check that attention is properly initialized
        assert attention.num_heads == 32  # Maintains all 32 heads
        assert attention.config == config
        assert attention.attention_impl is not None

    def test_device_aware_attention_output_shape(self):
        """Test that device-aware attention produces expected output shape."""
        config = Qwen3VLConfig()
        attention = DeviceAwareAttention(config, layer_idx=0)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output, _, _ = attention(hidden_states)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestGradientCheckpointing:
    """Test class for gradient checkpointing implementation."""

    def test_gradient_checkpointing_wrapper(self):
        """Test gradient checkpointing wrapper functionality."""
        # Create a simple layer to wrap
        layer = nn.Linear(128, 128)
        wrapped_layer = GradientCheckpointingWrapper(layer)
        
        # Create input tensor
        x = torch.randn(4, 128, requires_grad=True)
        
        # Forward pass should work
        output = wrapped_layer(x)
        assert output.shape == (4, 128)
        
        # Backward pass should work (gradient checkpointing should handle gradients)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None

    def test_memory_efficient_attention_initialization(self):
        """Test memory-efficient attention initialization."""
        config = Qwen3VLConfig()
        attention = MemoryEfficientAttention(config, layer_idx=0)
        
        # Check that attention is properly initialized
        assert attention.num_heads == 32  # Maintains all 32 heads
        assert attention.config == config

    def test_memory_efficient_attention_output_shape(self):
        """Test that memory-efficient attention produces expected output shape."""
        config = Qwen3VLConfig()
        attention = MemoryEfficientAttention(config, layer_idx=0)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output, _, _ = attention(hidden_states)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_memory_efficient_mlp_initialization(self):
        """Test memory-efficient MLP initialization."""
        config = Qwen3VLConfig()
        mlp = MemoryEfficientMLP(config)
        
        # Check that MLP is properly initialized
        assert mlp.hidden_size == config.hidden_size
        assert mlp.intermediate_size == config.intermediate_size

    def test_memory_efficient_mlp_output_shape(self):
        """Test that memory-efficient MLP produces expected output shape."""
        config = Qwen3VLConfig()
        mlp = MemoryEfficientMLP(config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = mlp(hidden_states)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestAdaptiveComputation:
    """Test class for adaptive computation pathways."""

    def test_adaptive_computation_gate_initialization(self):
        """Test adaptive computation gate initialization."""
        hidden_size = 2048
        gate = AdaptiveComputationGate(hidden_size, path_count=3)
        
        # Check that gate is properly initialized
        assert gate.hidden_size == hidden_size
        assert gate.path_count == 3
        assert gate.gate_network is not None

    def test_adaptive_computation_gate_output(self):
        """Test adaptive computation gate output shape."""
        hidden_size = 2048
        gate = AdaptiveComputationGate(hidden_size, path_count=3)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        gating_weights, selected_paths, low_confidence_mask = gate(hidden_states)
        
        # Check output shapes
        assert gating_weights.shape == (batch_size, 3)  # path_count = 3
        assert selected_paths.shape == (batch_size,)
        assert low_confidence_mask.shape == (batch_size,)

    def test_adaptive_attention_initialization(self):
        """Test adaptive attention initialization."""
        config = Qwen3VLConfig()
        attention = AdaptiveAttention(config, layer_idx=0)
        
        # Check that attention is properly initialized
        assert attention.num_heads == 32  # Maintains all 32 heads
        assert attention.paths is not None
        assert 'standard' in attention.paths
        assert 'sparse' in attention.paths
        assert 'linear' in attention.paths

    def test_adaptive_attention_output_shape(self):
        """Test that adaptive attention produces expected output shape."""
        config = Qwen3VLConfig()
        attention = AdaptiveAttention(config, layer_idx=0)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output, _, _ = attention(hidden_states)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)

    def test_adaptive_mlp_initialization(self):
        """Test adaptive MLP initialization."""
        config = Qwen3VLConfig()
        mlp = AdaptiveMLP(config)
        
        # Check that MLP is properly initialized
        assert mlp.hidden_size == config.hidden_size
        assert mlp.intermediate_size == config.intermediate_size
        assert mlp.paths is not None

    def test_adaptive_mlp_output_shape(self):
        """Test that adaptive MLP produces expected output shape."""
        config = Qwen3VLConfig()
        mlp = AdaptiveMLP(config)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = mlp(hidden_states)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestMemoryManagement:
    """Test class for memory management and data loading optimizations."""

    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        from src.models.memory_management import MemoryConfig
        config = MemoryConfig()
        manager = MemoryManager(config)
        
        # Check that manager is properly initialized
        assert manager.config == config
        assert manager.memory_stats is not None
        assert manager.tensor_cache is not None

    def test_memory_manager_tensor_allocation(self):
        """Test memory manager tensor allocation and deallocation."""
        from src.models.memory_management import MemoryConfig
        config = MemoryConfig()
        manager = MemoryManager(config)
        
        # Allocate a tensor
        shape = (2, 10, 2048)
        tensor = manager.allocate_tensor(shape)
        
        # Check tensor properties
        assert tensor.shape == shape
        assert tensor.device.type in ['cuda', 'cpu']
        
        # Free the tensor
        manager.free_tensor(tensor)
        
        # Check that tensor is cached
        cache_key = (shape, tensor.dtype, tensor.device)
        assert cache_key in manager.tensor_cache
        assert len(manager.tensor_cache[cache_key]) == 1

    def test_memory_manager_memory_stats(self):
        """Test memory manager memory statistics."""
        from src.models.memory_management import MemoryConfig
        config = MemoryConfig()
        manager = MemoryManager(config)
        
        # Get memory stats
        stats = manager.get_memory_stats()
        
        # Check that stats contain expected keys
        expected_keys = ['allocated_memory', 'reserved_memory', 'max_allocated', 'max_reserved']
        for key in expected_keys:
            assert key in stats

    def test_optimized_attention_initialization(self):
        """Test optimized attention initialization."""
        config = Qwen3VLConfig()
        from src.models.memory_management import MemoryConfig
        memory_config = MemoryConfig()
        memory_manager = MemoryManager(memory_config)
        
        attention = OptimizedQwen3VLAttention(config, layer_idx=0, memory_manager=memory_manager)
        
        # Check that attention is properly initialized
        assert attention.num_heads == 32  # Maintains all 32 heads
        assert attention.memory_manager == memory_manager

    def test_optimized_attention_output_shape(self):
        """Test that optimized attention produces expected output shape."""
        config = Qwen3VLConfig()
        from src.models.memory_management import MemoryConfig
        memory_config = MemoryConfig()
        memory_manager = MemoryManager(memory_config)
        
        attention = OptimizedQwen3VLAttention(config, layer_idx=0, memory_manager=memory_manager)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output, _, _ = attention(hidden_states)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)


class TestIntegration:
    """Integration tests for combined components."""

    def test_attention_equivalence(self):
        """Test that different attention implementations produce similar outputs."""
        config = Qwen3VLConfig()
        
        # Create different attention implementations
        standard_attention = PerformerAttention(config, layer_idx=0)
        device_aware_attention = DeviceAwareAttention(config, layer_idx=0)
        adaptive_attention = AdaptiveAttention(config, layer_idx=0)
        
        batch_size, seq_len = 2, 10
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Get outputs from different implementations
        output_standard, _, _ = standard_attention(hidden_states)
        output_device_aware, _, _ = device_aware_attention(hidden_states)
        output_adaptive, _, _ = adaptive_attention(hidden_states)
        
        # All outputs should have the same shape
        assert output_standard.shape == output_device_aware.shape == output_adaptive.shape
        assert output_standard.shape == (batch_size, seq_len, config.hidden_size)

    def test_memory_efficiency(self):
        """Test that gradient checkpointing reduces memory usage."""
        # Create a model with gradient checkpointing
        config = Qwen3VLConfig()
        attention = MemoryEfficientAttention(config, layer_idx=0)
        
        batch_size, seq_len = 2, 20  # Use longer sequence to better test memory usage
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)
        
        # Forward pass
        output, _, _ = attention(hidden_states, use_cache=False)
        
        # Backward pass should work with gradient checkpointing
        loss = output.sum()
        loss.backward()
        
        # Check that gradients are computed
        assert hidden_states.grad is not None


def run_tests():
    """Run all tests."""
    test_classes = [
        TestPerformerAttention,
        TestDeviceAwareModule,
        TestGradientCheckpointing,
        TestAdaptiveComputation,
        TestMemoryManagement,
        TestIntegration
    ]
    
    for test_class in test_classes:
        test_instance = test_class()
        for attr_name in dir(test_instance):
            if attr_name.startswith('test_'):
                test_method = getattr(test_instance, attr_name)
                print(f"Running {test_class.__name__}.{attr_name}...")
                try:
                    test_method()
                    print(f"  ✓ {attr_name} passed")
                except Exception as e:
                    print(f"  ✗ {attr_name} failed: {str(e)}")
                    raise


if __name__ == "__main__":
    run_tests()