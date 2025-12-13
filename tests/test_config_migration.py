"""
Test suite for the configuration migration system.
This tests the migration from old modular configs to the unified config.
"""
import pytest
from src.qwen3_vl.config.unified_config import UnifiedQwen3VLConfig
from src.qwen3_vl.config.config_migration import migrate_from_modular_to_unified, ConfigConverterFactory
from src.qwen3_vl.config.config import Qwen3VLConfig  # Import old config


def test_migration_from_modular_to_unified():
    """Test migration from old modular config to unified config."""
    # Create an old config with some custom values
    old_config = Qwen3VLConfig(
        vocab_size=30000,
        hidden_size=1024,
        use_flash_attention_2=True,
        attention_dropout_prob=0.1,
        kv_cache_strategy="sliding_window",
        sparsity_ratio=0.3,
        moe_num_experts=8,
        gpu_memory_fraction=0.8
    )
    
    # Migrate to unified config
    new_config = migrate_from_modular_to_unified(old_config)
    
    # Verify values were transferred correctly
    assert new_config.vocab_size == 30000
    assert new_config.hidden_size == 1024
    assert new_config.use_flash_attention_2 is True
    assert new_config.attention_dropout_prob == 0.1
    assert new_config.kv_cache_strategy == "sliding_window"
    assert new_config.sparsity_ratio == 0.3
    assert new_config.moe_num_experts == 8
    assert new_config.gpu_memory_fraction == 0.8


def test_config_converter_factory():
    """Test the configuration converter factory."""
    # Test with dictionary
    config_dict = {
        'vocab_size': 20000,
        'hidden_size': 512,
        'use_flash_attention_2': True
    }
    config_from_dict = ConfigConverterFactory.convert_to_unified(config_dict)
    assert config_from_dict.vocab_size == 20000
    assert config_from_dict.hidden_size == 512
    assert config_from_dict.use_flash_attention_2 is True
    
    # Test with old config
    old_config = Qwen3VLConfig(vocab_size=25000, hidden_size=768)
    config_from_old = ConfigConverterFactory.convert_to_unified(old_config)
    assert config_from_old.vocab_size == 25000
    assert config_from_old.hidden_size == 768
    
    # Test with unified config (should return same object)
    unified_config = UnifiedQwen3VLConfig(vocab_size=15000)
    same_config = ConfigConverterFactory.convert_to_unified(unified_config)
    assert same_config is unified_config  # Should be the same object
    
    # Test with invalid type
    with pytest.raises(ValueError):
        ConfigConverterFactory.convert_to_unified("invalid_type")


def test_config_factory_methods():
    """Test the factory methods for different config types."""
    balanced_config = ConfigConverterFactory.create_default_config("balanced")
    perf_config = ConfigConverterFactory.create_default_config("performance_optimized")
    mem_config = ConfigConverterFactory.create_default_config("memory_efficient")
    
    # Verify they are different configs
    assert balanced_config != perf_config
    assert balanced_config != mem_config
    assert perf_config != mem_config
    
    # Verify specific properties based on factory methods
    assert perf_config.use_flash_attention_2 is True
    assert mem_config.use_gradient_checkpointing is True
    assert mem_config.use_activation_sparsity is True


def test_backward_compatibility():
    """Test that the unified config maintains backward compatibility."""
    # Create a unified config
    unified_config = UnifiedQwen3VLConfig(
        use_flash_attention_2=True,
        kv_cache_strategy="low_rank",
        sparsity_ratio=0.4
    )
    
    # Verify that expected attributes are accessible
    assert unified_config.use_flash_attention_2 is True
    assert unified_config.kv_cache_strategy == "low_rank"
    assert unified_config.sparsity_ratio == 0.4
    
    # Test that values can be changed
    unified_config.use_flash_attention_2 = False
    assert unified_config.use_flash_attention_2 is False
    
    unified_config.kv_cache_strategy = "hybrid"
    assert unified_config.kv_cache_strategy == "hybrid"


if __name__ == "__main__":
    pytest.main([__file__])