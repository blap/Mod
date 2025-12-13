"""
Test suite for the unified Qwen3-VL configuration system.
This tests the new unified configuration implementation.
"""
import pytest
from dataclasses import dataclass
from typing import Optional, List
from src.qwen3_vl.config.unified_config import UnifiedQwen3VLConfig


def test_unified_config_basic_properties():
    """Test basic configuration properties."""
    config = UnifiedQwen3VLConfig()
    
    # Test basic language model properties
    assert config.vocab_size == 152064
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 32
    assert config.num_attention_heads == 32
    
    # Test basic vision model properties
    assert config.vision_model_type == "clip_vision_model"
    assert config.vision_hidden_size == 1152
    assert config.vision_num_hidden_layers == 24


def test_unified_config_memory_properties():
    """Test memory-related configuration properties."""
    config = UnifiedQwen3VLConfig()
    
    # Test memory properties
    assert config.use_memory_pooling is True
    assert config.memory_pool_initial_size == 1024 * 1024 * 256
    assert config.kv_cache_strategy == "hybrid"
    assert config.use_low_rank_kv_cache is True
    assert config.kv_cache_window_size == 1024
    assert config.kv_low_rank_dimension == 64
    assert config.use_gradient_checkpointing is True
    assert config.use_activation_sparsity is False
    assert config.sparsity_ratio == 0.5


def test_unified_config_attention_properties():
    """Test attention-related configuration properties."""
    config = UnifiedQwen3VLConfig()
    
    # Test attention properties
    assert config.attention_implementation == "eager"
    assert config.use_flash_attention_2 is False
    assert config.attention_dropout_prob == 0.0
    assert config.use_memory_efficient_attention is False
    assert config.rope_theta == 1000000.0
    assert config.use_rotary_embedding is True
    assert config.use_dynamic_sparse_attention is False


def test_unified_config_routing_properties():
    """Test routing-related configuration properties."""
    config = UnifiedQwen3VLConfig()
    
    # Test routing properties
    assert config.use_moe is False
    assert config.moe_num_experts == 4
    assert config.moe_top_k == 2
    assert config.use_token_level_routing is False
    assert config.use_adaptive_routing is False


def test_unified_config_hardware_properties():
    """Test hardware-related configuration properties."""
    config = UnifiedQwen3VLConfig()
    
    # Test hardware properties
    assert config.gpu_memory_fraction == 0.9
    assert config.hardware_target == "auto"
    assert config.enable_intel_optimizations is True
    assert config.enable_nvidia_optimizations is True


def test_direct_field_access():
    """Test that all fields can be accessed directly."""
    config = UnifiedQwen3VLConfig()

    # Test that fields work correctly
    original_rope_theta = config.rope_theta
    config.rope_theta = 2000000.0
    assert config.rope_theta == 2000000.0

    original_kv_cache_strategy = config.kv_cache_strategy
    config.kv_cache_strategy = "low_rank"
    assert config.kv_cache_strategy == "low_rank"

    original_sparsity_ratio = config.sparsity_ratio
    config.sparsity_ratio = 0.3
    assert config.sparsity_ratio == 0.3


def test_config_validation():
    """Test configuration validation."""
    # Test that invalid values raise appropriate errors
    with pytest.raises(ValueError):
        UnifiedQwen3VLConfig(num_hidden_layers=16)  # Must be 32
    
    with pytest.raises(ValueError):
        UnifiedQwen3VLConfig(num_attention_heads=16)  # Must be 32


def test_config_with_custom_values():
    """Test configuration with custom values."""
    config = UnifiedQwen3VLConfig(
        vocab_size=50000,
        hidden_size=1024,
        use_flash_attention_2=True,
        kv_cache_strategy="sliding_window",
        moe_num_experts=8
    )
    
    assert config.vocab_size == 50000
    assert config.hidden_size == 1024
    assert config.use_flash_attention_2 is True
    assert config.kv_cache_strategy == "sliding_window"
    assert config.moe_num_experts == 8


def test_factory_methods():
    """Test configuration factory methods."""
    # Test performance optimized config
    perf_config = UnifiedQwen3VLConfig.performance_optimized()
    assert perf_config.use_flash_attention_2 is True
    assert perf_config.use_gradient_checkpointing is False  # More memory usage for speed
    
    # Test memory efficient config
    mem_config = UnifiedQwen3VLConfig.memory_efficient()
    assert mem_config.use_gradient_checkpointing is True
    assert mem_config.use_activation_sparsity is True
    
    # Test balanced config
    balanced_config = UnifiedQwen3VLConfig.balanced()
    assert balanced_config.use_flash_attention_2 is True
    assert balanced_config.use_gradient_checkpointing is True


def test_to_dict_and_from_dict():
    """Test serialization and deserialization methods."""
    original_config = UnifiedQwen3VLConfig(use_flash_attention_2=True)
    config_dict = original_config.to_dict()
    
    # Verify key values are present
    assert 'use_flash_attention_2' in config_dict
    assert 'vocab_size' in config_dict
    assert 'hidden_size' in config_dict
    
    # Create new config from dict
    new_config = UnifiedQwen3VLConfig.from_dict(config_dict)
    assert new_config.use_flash_attention_2 == original_config.use_flash_attention_2
    assert new_config.vocab_size == original_config.vocab_size
    assert new_config.hidden_size == original_config.hidden_size


def test_repr_method():
    """Test string representation of the config."""
    config = UnifiedQwen3VLConfig(vocab_size=10000)
    repr_str = repr(config)
    assert "UnifiedQwen3VLConfig" in repr_str
    assert "vocab_size=10000" in repr_str


def test_equality():
    """Test equality comparison between configs."""
    config1 = UnifiedQwen3VLConfig()
    config2 = UnifiedQwen3VLConfig()
    config3 = UnifiedQwen3VLConfig(vocab_size=10000)
    
    assert config1 == config2
    assert config1 != config3


if __name__ == "__main__":
    pytest.main([__file__])