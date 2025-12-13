"""
Comprehensive tests for hardware-specific features in the abstraction layer
"""
import pytest
import torch
import torch.nn as nn
from src.components.hardware.hardware_abstraction import (
    DeviceSelector,
    HardwareOptimizer,
    MemoryAccessOptimizer,
    HardwareManager,
    DeviceAwareAttention,
    SM61OptimizedAttentionWrapper
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


def test_device_classification():
    """Test that device classification works correctly for different hardware"""
    selector = DeviceSelector()
    device_class = selector.classify_device()
    
    # The device should be classified into one of the expected categories
    assert device_class in ['cpu', 'nvidia_sm61', 'nvidia_ampere_or_newer', 'nvidia_turing', 'nvidia_other', 'other']
    
    # Test that the classification is consistent
    device_info = selector.get_current_device_info()
    assert hasattr(device_info, 'device_type')
    assert hasattr(device_info, 'compute_capability')


def test_sm61_kernel_selection():
    """Test that SM61-specific kernels are selected appropriately"""
    optimizer = HardwareOptimizer()

    # Test attention kernel selection for SM61
    kernel = optimizer.select_optimal_kernel('attention', 'nvidia', (6, 1))
    # For SM61 on NVIDIA hardware, should be 'optimized_standard_attention'
    # If actual hardware is CPU, it would be 'cpu_optimized_kernel'
    assert kernel in ['optimized_standard_attention', 'cpu_optimized_kernel']

    # Test convolution kernel selection for SM61
    conv_kernel = optimizer.select_optimal_kernel('convolution', 'nvidia', (6, 1))
    assert conv_kernel in ['sm61_optimized_conv', 'cpu_optimized_kernel']

    # Test GEMM kernel selection for SM61
    gemm_kernel = optimizer.select_optimal_kernel('gemm', 'nvidia', (6, 1))
    assert gemm_kernel in ['sm61_optimized_gemm', 'cpu_optimized_kernel']


def test_modern_gpu_kernel_selection():
    """Test that modern GPU kernels are selected for newer architectures"""
    optimizer = HardwareOptimizer()

    # Test attention kernel selection for modern GPU (Ampere, compute capability 8.x)
    kernel = optimizer.select_optimal_kernel('attention', 'nvidia', (8, 0))
    assert kernel in ['flash_attention', 'cpu_optimized_kernel']

    # Test convolution kernel selection for modern GPU
    conv_kernel = optimizer.select_optimal_kernel('convolution', 'nvidia', (8, 0))
    assert conv_kernel in ['cudnn_optimized_conv', 'cpu_optimized_kernel']


def test_memory_optimization_configs():
    """Test memory optimization configurations for different hardware"""
    optimizer = HardwareOptimizer()
    
    # Test SM61 memory config
    sm61_config = optimizer.get_memory_optimization_config('cuda', 6, 1)
    assert sm61_config['memory_efficient'] is True
    assert sm61_config['use_gradient_checkpointing'] is True
    assert sm61_config['kv_cache_strategy'] == 'paged'
    assert sm61_config['max_memory_reduction'] is True
    
    # Test modern GPU memory config
    modern_config = optimizer.get_memory_optimization_config('cuda', 8, 0)
    assert modern_config['memory_efficient'] is False  # Can afford more compute
    assert modern_config['use_gradient_checkpointing'] is False
    assert modern_config['kv_cache_strategy'] == 'standard'
    assert modern_config['max_memory_reduction'] is False


def test_memory_access_patterns():
    """Test optimized memory access patterns for different hardware"""
    optimizer = MemoryAccessOptimizer()
    
    # Test SM61 memory access optimization
    sm61_access = optimizer.get_optimized_memory_access('cuda', 6, 1)
    assert sm61_access['access_pattern'] == 'coalesced'
    assert sm61_access['tile_size'] == 32  # Smaller tiles for better cache utilization
    assert sm61_access['memory_layout'] == 'blocked'
    assert sm61_access['prefetch_depth'] == 2
    assert sm61_access['use_async_memory_ops'] is True
    
    # Test modern GPU memory access optimization
    modern_access = optimizer.get_optimized_memory_access('cuda', 8, 0)
    assert modern_access['access_pattern'] == 'tensor_core_optimized'
    assert modern_access['tile_size'] == 128  # Larger tiles for tensor cores
    assert modern_access['memory_layout'] == 'blocked'
    assert modern_access['prefetch_depth'] == 3
    assert modern_access['use_async_memory_ops'] is True


def test_hardware_feature_detection():
    """Test that hardware feature detection works correctly"""
    manager = HardwareManager()
    
    # Test feature detection
    assert isinstance(manager.supports_feature('fp16'), bool)
    assert isinstance(manager.supports_feature('bf16'), bool)
    assert isinstance(manager.supports_feature('int8'), bool)
    assert isinstance(manager.supports_feature('flash_attention'), bool)
    assert isinstance(manager.supports_feature('tensor_cores'), bool)
    assert isinstance(manager.supports_feature('nvme'), bool)


def test_device_aware_attention_sm61_path():
    """Test that DeviceAwareAttention uses SM61-optimized path when appropriate"""
    config = MockConfig()
    
    # Since we can't control the actual hardware in tests, we'll check the implementation
    # by verifying that the correct components are used
    attention = DeviceAwareAttention(config)
    
    # Check that the attention implementation is appropriate for the hardware
    device_info = attention.hardware_manager.get_device_info()
    
    # Depending on the actual hardware, verify appropriate components are used
    assert hasattr(attention, 'q_proj')
    assert hasattr(attention, 'k_proj')
    assert hasattr(attention, 'v_proj')
    assert hasattr(attention, 'o_proj')
    
    # Verify the attention implementation is properly initialized
    assert hasattr(attention, 'attention_impl')
    
    # Check that memory optimizations are applied
    memory_config = attention.memory_config
    assert isinstance(memory_config, dict)
    assert 'memory_efficient' in memory_config


def test_sm61_attention_wrapper():
    """Test the SM61 optimized attention wrapper specifically"""
    config = MockConfig()
    
    # Create SM61 optimized attention wrapper directly
    sm61_attention = SM61OptimizedAttentionWrapper(
        config,
        layer_idx=0,
        head_dim=config.hidden_size // config.num_attention_heads,
        num_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        num_key_value_groups=config.num_attention_heads // config.num_key_value_heads
    )
    
    # Verify SM61-specific optimizations are applied
    assert hasattr(sm61_attention, 'memory_config')
    assert sm61_attention.memory_config['access_pattern'] == 'coalesced'
    assert sm61_attention.memory_config['tile_size'] == 32


def test_adaptive_pathway_selection():
    """Test adaptive pathway selection based on hardware capabilities"""
    optimizer = HardwareOptimizer()
    selector = DeviceSelector()
    
    device_info = selector.get_current_device_info()
    
    # Test attention pathway selection
    attention_pathway = optimizer.select_computation_pathway('attention', device_info)
    assert attention_pathway in ['memory_efficient_attention', 'flash_attention_pathway', 'compute_optimized_pathway', 'standard_cuda_pathway', 'cpu_optimized_pathway']
    
    # Test matmul pathway selection
    matmul_pathway = optimizer.select_computation_pathway('matmul', device_info)
    assert matmul_pathway in ['optimized_matmul_sm61', 'compute_optimized_pathway', 'standard_cuda_pathway', 'cpu_optimized_pathway']


def test_fallback_mechanisms():
    """Test that fallback mechanisms work correctly"""
    optimizer = HardwareOptimizer()
    
    # Test fallback for unknown operation
    fallback_pathway = optimizer.select_computation_pathway('unknown_operation', optimizer.device_info)
    assert fallback_pathway in ['standard_pathway', 'cpu_optimized_pathway', 'standard_cuda_pathway']
    
    # Test fallback for unknown device type
    fake_device_info = type('DeviceInfo', (), {
        'device_type': 'fake_device',
        'compute_capability': (0, 0)
    })()
    fallback_pathway2 = optimizer.select_computation_pathway('attention', fake_device_info)
    assert fallback_pathway2 in ['standard_pathway', 'cpu_optimized_pathway', 'standard_cuda_pathway']


if __name__ == "__main__":
    pytest.main([__file__])