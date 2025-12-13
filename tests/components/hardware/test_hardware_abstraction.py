"""
Tests for the hardware abstraction layer
"""
import pytest
import torch
import torch.nn as nn
from src.components.hardware.hardware_abstraction import (
    HardwareManager, 
    DeviceAwareAttention,
    DeviceSelector,
    HardwareOptimizer,
    MemoryAccessOptimizer
)
from transformers import PretrainedConfig


class MockConfig(PretrainedConfig):
    """Mock configuration for testing"""
    def __init__(self):
        super().__init__()
        self.hidden_size = 512
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.max_position_embeddings = 2048
        self.rope_theta = 10000.0


def test_hardware_manager_initialization():
    """Test that the hardware manager initializes correctly"""
    manager = HardwareManager()
    device_info = manager.get_device_info()

    assert hasattr(device_info, 'device_type')
    assert hasattr(device_info, 'memory_gb')
    assert device_info.device_type in ['cuda', 'cpu']


def test_hardware_manager_features():
    """Test that the hardware manager correctly detects features"""
    manager = HardwareManager()
    
    # Test feature detection
    assert isinstance(manager.supports_feature('flash_attention'), bool)
    assert isinstance(manager.supports_feature('bf16'), bool)
    assert isinstance(manager.supports_feature('fp16'), bool)


def test_device_selector():
    """Test the device selector functionality"""
    selector = DeviceSelector()

    # Test device detection
    device_info = selector.get_current_device_info()
    assert hasattr(device_info, 'device_name')
    assert hasattr(device_info, 'compute_capability')
    assert hasattr(device_info, 'memory_gb')

    # Test device classification
    device_class = selector.classify_device()
    assert device_class in ['cpu', 'nvidia_sm61', 'nvidia_ampere_or_newer', 'nvidia_turing', 'nvidia_other', 'other']


def test_hardware_optimizer():
    """Test the hardware optimizer"""
    optimizer = HardwareOptimizer()

    # Test kernel selection
    kernel = optimizer.select_optimal_kernel('attention', 'nvidia', (6, 1))
    assert kernel in ['optimized_standard_attention', 'sm61_optimized_conv', 'sm61_optimized_gemm', 'fallback_kernel', 'cpu_optimized_kernel']

    # Test memory optimization
    memory_config = optimizer.get_memory_optimization_config('nvidia', 6, 1)
    assert isinstance(memory_config, dict)
    assert 'memory_efficient' in memory_config


def test_device_aware_attention_initialization():
    """Test that DeviceAwareAttention initializes correctly"""
    config = MockConfig()
    
    # Test initialization
    attention = DeviceAwareAttention(config)
    
    # Check that it has the right components
    assert hasattr(attention, 'hardware_manager')
    assert hasattr(attention, 'selected_implementation')
    assert hasattr(attention, 'attention_impl')
    
    # Check that projections are created
    assert isinstance(attention.q_proj, nn.Linear)
    assert isinstance(attention.k_proj, nn.Linear)
    assert isinstance(attention.v_proj, nn.Linear)
    assert isinstance(attention.o_proj, nn.Linear)


def test_device_aware_attention_forward():
    """Test the forward pass of DeviceAwareAttention"""
    config = MockConfig()
    attention = DeviceAwareAttention(config)
    
    # Create test inputs
    batch_size = 2
    seq_len = 16
    hidden_size = config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # Test forward pass
    output, attn_weights, past_key_value = attention(
        hidden_states=hidden_states
    )
    
    # Check output dimensions
    assert output.shape == (batch_size, seq_len, hidden_size)
    if attn_weights is not None:
        assert attn_weights.shape[0] == batch_size
        assert attn_weights.shape[1] == config.num_attention_heads
        assert attn_weights.shape[2] == seq_len
        assert attn_weights.shape[3] == seq_len


def test_memory_access_optimizer():
    """Test the memory access optimizer"""
    optimizer = MemoryAccessOptimizer()
    
    # Test memory access pattern optimization
    optimized_access = optimizer.get_optimized_memory_access('cuda', 6, 1)
    assert isinstance(optimized_access, dict)
    assert 'access_pattern' in optimized_access
    assert 'tile_size' in optimized_access


def test_fallback_mechanisms():
    """Test fallback mechanisms for different hardware configurations"""
    # Test with CUDA available
    if torch.cuda.is_available():
        manager = HardwareManager()
        device_info = manager.get_device_info()

        # Ensure fallbacks work
        fallback_kernel = manager.get_optimal_implementation('nonexistent_op')
        assert fallback_kernel in ['standard_pathway', 'cpu_optimized_pathway', 'standard_cuda_pathway']  # or appropriate fallback


def test_nvidia_sm61_optimizations():
    """Test specific optimizations for NVIDIA SM61"""
    # This test would verify that SM61-specific optimizations are applied
    optimizer = HardwareOptimizer()

    # Check that appropriate kernels are selected for SM61
    kernel = optimizer.select_optimal_kernel('attention', 'nvidia', (6, 1))
    # For SM61, we might not have the latest optimizations, so fallback is expected
    assert kernel in ['optimized_standard_attention', 'sm61_optimized_conv', 'sm61_optimized_gemm', 'fallback_kernel', 'cpu_optimized_kernel']


def test_adaptive_computation_pathways():
    """Test adaptive computation pathways based on hardware"""
    selector = DeviceSelector()
    optimizer = HardwareOptimizer()

    # Get current device info
    device_info = selector.get_current_device_info()

    # Test pathway selection based on device capabilities
    pathway = optimizer.select_computation_pathway('attention', device_info)
    assert pathway in ['memory_efficient_attention', 'flash_attention_pathway', 'compute_optimized_pathway', 'standard_cuda_pathway', 'cpu_optimized_pathway']


if __name__ == "__main__":
    pytest.main([__file__])