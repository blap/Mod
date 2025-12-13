"""
Comprehensive tests for the Qwen3-VL configuration system.

This module provides unit tests for the configuration management system
with proper separation of concerns and validation.
"""
import pytest
import tempfile
import os
from src.qwen3_vl.config import (
    Qwen3VLConfig, 
    ConfigFactory, 
    MemoryConfig, 
    AttentionConfig, 
    RoutingConfig, 
    HardwareConfig,
    ConfigValidator,
    ConfigValidatorService
)
from src.qwen3_vl.components.system import create_default_container


def test_memory_config_creation():
    """Test creation of MemoryConfig with default values."""
    config = MemoryConfig()
    
    # Test default values
    assert config.use_memory_pooling == True
    assert config.memory_pool_initial_size == 1024 * 1024 * 256  # 256MB
    assert config.memory_pool_max_size == 1024 * 1024 * 1024    # 1GB
    assert config.memory_pool_growth_factor == 1.5
    assert config.use_buddy_allocation == True
    assert config.kv_cache_strategy == "hybrid"
    assert config.use_low_rank_kv_cache == True
    assert config.kv_cache_window_size == 1024
    assert config.kv_low_rank_dimension == 64
    assert config.use_gradient_checkpointing == True
    assert config.use_activation_sparsity == False
    assert config.sparsity_ratio == 0.5


def test_attention_config_creation():
    """Test creation of AttentionConfig with default values."""
    config = AttentionConfig()
    
    # Test default values
    assert config.attention_implementation == "eager"
    assert config.use_flash_attention_2 == False
    assert config.flash_attention_causal == True
    assert config.attention_dropout_prob == 0.0
    assert config.use_dynamic_sparse_attention == False
    assert config.sparse_attention_sparsity_ratio == 0.5
    assert config.vision_sparse_attention_sparsity_ratio == 0.4
    assert config.sparse_attention_pattern == "top_k"
    assert config.sparse_attention_num_blocks == 32
    assert config.rope_theta == 1000000.0
    assert config.use_rotary_embedding == True
    assert config.num_attention_heads == 32


def test_routing_config_creation():
    """Test creation of RoutingConfig with default values."""
    config = RoutingConfig()
    
    # Test default values
    assert config.use_moe == False
    assert config.moe_num_experts == 4
    assert config.moe_top_k == 2
    assert config.moe_use_residual == True
    assert config.moe_jitter_noise == 0.01
    assert config.moe_normalize_gate == True
    assert config.moe_capacity_factor == 1.0
    assert config.moe_drop_tokens == True
    assert config.moe_use_tutel == False
    assert config.moe_router_zloss_coef == 1e-4
    assert config.moe_router_aux_loss_coef == 1e-2


def test_hardware_config_creation():
    """Test creation of HardwareConfig with default values."""
    config = HardwareConfig()
    
    # Test default values
    assert config.hardware_detection_timeout == 30
    assert config.hardware_fallback_enabled == True
    assert config.gpu_memory_fraction == 0.9
    assert config.gpu_precision == "mixed"
    assert config.use_pinned_memory == True
    assert config.use_cuda_streams == True
    assert config.hardware_target == "auto"
    assert config.enable_intel_optimizations == True
    assert config.enable_nvidia_optimizations == True
    assert config.nvme_cache_enabled == True
    assert config.nvme_cache_size == 1024 * 1024 * 1024 * 10  # 10GB
    assert config.nvme_cache_policy == "lru"
    assert config.power_management_enabled == True
    assert config.thermal_throttling_enabled == True


def test_qwen3vl_config_creation():
    """Test creation of Qwen3VLConfig with modular components."""
    config = Qwen3VLConfig()
    
    # Test main config values
    assert config.num_hidden_layers == 32
    assert config.num_attention_heads == 32
    assert config.vocab_size == 152064
    assert config.hidden_size == 2048
    
    # Test that modular components are initialized
    assert isinstance(config.memory_config, MemoryConfig)
    assert isinstance(config.attention_config, AttentionConfig)
    assert isinstance(config.routing_config, RoutingConfig)
    assert isinstance(config.hardware_config, HardwareConfig)
    
    # Test that modular components have proper defaults
    assert config.memory_config.use_memory_pooling == True
    assert config.attention_config.attention_implementation == "eager"
    assert config.routing_config.use_moe == False
    assert config.hardware_config.hardware_target == "auto"


def test_config_validation_success():
    """Test that valid configurations pass validation."""
    config = Qwen3VLConfig()
    is_valid, errors = ConfigValidator.validate_full_config(config)
    
    assert is_valid == True
    assert len(errors) == 0


def test_config_validation_memory_errors():
    """Test validation of invalid memory configuration."""
    # Create config with invalid memory settings
    config = Qwen3VLConfig()
    config.memory_config.memory_pool_initial_size = -100  # Invalid
    config.memory_config.kv_cache_window_size = 0  # Invalid
    config.memory_config.sparsity_ratio = 1.5  # Invalid
    
    is_valid, errors = ConfigValidator.validate_full_config(config)
    
    assert is_valid == False
    assert len(errors) > 0
    # Check for specific error messages
    error_messages = " ".join(errors)
    assert "memory_pool_initial_size must be positive" in error_messages
    assert "kv_cache_window_size must be positive" in error_messages
    assert "sparsity_ratio must be between 0.0 and 1.0" in error_messages


def test_config_validation_attention_errors():
    """Test validation of invalid attention configuration."""
    config = Qwen3VLConfig()
    config.attention_config.attention_implementation = "invalid_implementation"  # Invalid
    config.attention_config.attention_dropout_prob = 1.5  # Invalid
    config.attention_config.num_attention_heads = 0  # Invalid
    
    is_valid, errors = ConfigValidator.validate_full_config(config)
    
    assert is_valid == False
    assert len(errors) > 0
    error_messages = " ".join(errors)
    assert "attention_implementation must be one of" in error_messages
    assert "attention_dropout_prob must be between 0.0 and 1.0" in error_messages
    assert "num_attention_heads must be positive" in error_messages


def test_config_validation_routing_errors():
    """Test validation of invalid routing configuration."""
    config = Qwen3VLConfig()
    config.routing_config.moe_num_experts = 0  # Invalid
    config.routing_config.moe_top_k = 5  # Invalid (greater than num_experts)
    config.routing_config.moe_jitter_noise = -0.1  # Invalid
    
    is_valid, errors = ConfigValidator.validate_full_config(config)
    
    assert is_valid == False
    assert len(errors) > 0
    error_messages = " ".join(errors)
    assert "moe_num_experts must be at least 1" in error_messages
    assert "moe_top_k must be between 1 and moe_num_experts" in error_messages
    assert "moe_jitter_noise must be non-negative" in error_messages


def test_config_validation_hardware_errors():
    """Test validation of invalid hardware configuration."""
    config = Qwen3VLConfig()
    config.hardware_config.gpu_memory_fraction = 1.5  # Invalid
    config.hardware_config.hardware_detection_timeout = 0  # Invalid
    config.hardware_config.cpu_threads = -1  # Invalid
    
    is_valid, errors = ConfigValidator.validate_full_config(config)
    
    assert is_valid == False
    assert len(errors) > 0
    error_messages = " ".join(errors)
    assert "gpu_memory_fraction must be between 0 and 1.0" in error_messages
    assert "hardware_detection_timeout must be positive" in error_messages
    assert "cpu_threads must be positive or None" in error_messages


def test_config_factory_from_dict():
    """Test ConfigFactory creation from dictionary."""
    config_dict = {
        "hidden_size": 1024,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "memory_config": {
            "use_memory_pooling": True,
            "memory_pool_initial_size": 128 * 1024 * 1024,  # 128MB
        },
        "attention_config": {
            "use_flash_attention_2": True,
            "attention_dropout_prob": 0.1,
        },
        "routing_config": {
            "use_moe": True,
            "moe_num_experts": 8,
        },
        "hardware_config": {
            "gpu_precision": "fp16",
            "hardware_target": "nvidia_sm61",
        }
    }
    
    config = ConfigFactory.from_dict(config_dict)
    
    # Test main config values
    assert config.hidden_size == 1024
    assert config.num_hidden_layers == 32
    assert config.num_attention_heads == 32
    
    # Test modular config values
    assert config.memory_config.use_memory_pooling == True
    assert config.memory_config.memory_pool_initial_size == 128 * 1024 * 1024
    assert config.attention_config.use_flash_attention_2 == True
    assert config.attention_config.attention_dropout_prob == 0.1
    assert config.routing_config.use_moe == True
    assert config.routing_config.moe_num_experts == 8
    assert config.hardware_config.gpu_precision == "fp16"
    assert config.hardware_config.hardware_target == "nvidia_sm61"


def test_config_factory_save_and_load():
    """Test saving and loading configuration."""
    # Create a config
    original_config = ConfigFactory.create_optimized_config_for_hardware("intel_i5_10210u")
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        ConfigFactory.save_config(original_config, temp_path)
        
        # Load from file
        loaded_config = ConfigFactory.from_json_file(temp_path)
        
        # Compare key values
        assert loaded_config.num_hidden_layers == original_config.num_hidden_layers
        assert loaded_config.num_attention_heads == original_config.num_attention_heads
        assert loaded_config.memory_config.use_gradient_checkpointing == original_config.memory_config.use_gradient_checkpointing
        assert loaded_config.attention_config.use_flash_attention_2 == original_config.attention_config.use_flash_attention_2
        assert loaded_config.routing_config.use_moe == original_config.routing_config.use_moe
        assert loaded_config.hardware_config.hardware_target == original_config.hardware_config.hardware_target
        
        # Validate loaded config
        is_valid, errors = ConfigValidator.validate_full_config(loaded_config)
        assert is_valid == True
        assert len(errors) == 0
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_config_validator_service():
    """Test the configuration validation service."""
    config = Qwen3VLConfig()
    report = ConfigValidatorService.validate_and_report(config)
    
    assert report["is_valid"] == True
    assert report["total_errors"] == 0
    assert "components_validated" in report
    assert "validation_timestamp" in report
    
    # Check that each component validation returned empty lists
    for component_errors in report["components_validated"].values():
        assert len(component_errors) == 0


def test_dependency_injection_container():
    """Test the dependency injection container."""
    config = ConfigFactory.create_optimized_config_for_hardware("intel_i5_10210u")
    container = create_default_container(config)
    
    # Test that we can get the main config
    retrieved_config = container.get('main_config')
    assert retrieved_config is config
    
    # Test that we can get memory config
    memory_config = container.get('memory_config')
    assert memory_config is config.memory_config
    
    # Test that we can get attention config
    attention_config = container.get('attention_config')
    assert attention_config is config.attention_config
    
    # Test that we can get routing config
    routing_config = container.get('routing_config')
    assert routing_config is config.routing_config
    
    # Test that we can get hardware config
    hardware_config = container.get('hardware_config')
    assert hardware_config is config.hardware_config
    
    # Test that we can get memory manager
    memory_manager = container.get('memory_manager')
    assert memory_manager is not None
    # Verify that the memory manager was created with the correct config
    assert memory_manager.config is config.memory_config


def test_optimized_configs():
    """Test creation of optimized configurations for different hardware."""
    # Test Intel i5-10210u optimized config
    intel_config = ConfigFactory.create_optimized_config_for_hardware("intel_i5_10210u")
    assert intel_config.memory_config.use_gradient_checkpointing == True
    assert intel_config.torch_dtype == "float16"
    assert intel_config.memory_config.use_activation_sparsity == True
    assert intel_config.routing_config.use_moe == True
    assert intel_config.attention_config.use_flash_attention_2 == True
    
    # Test NVIDIA SM61 optimized config
    nvidia_config = ConfigFactory.create_optimized_config_for_hardware("nvidia_sm61")
    assert nvidia_config.memory_config.use_gradient_checkpointing == True
    assert nvidia_config.torch_dtype == "float16"
    assert nvidia_config.attention_config.use_flash_attention_2 == True
    assert nvidia_config.hardware_config.enable_tensor_cores == True
    
    # Test generic optimized config
    generic_config = ConfigFactory.create_optimized_config_for_hardware("generic")
    assert generic_config.memory_config.use_gradient_checkpointing == True
    assert generic_config.torch_dtype == "float16"


if __name__ == "__main__":
    # Run the tests
    test_memory_config_creation()
    test_attention_config_creation()
    test_routing_config_creation()
    test_hardware_config_creation()
    test_qwen3vl_config_creation()
    test_config_validation_success()
    test_config_validation_memory_errors()
    test_config_validation_attention_errors()
    test_config_validation_routing_errors()
    test_config_validation_hardware_errors()
    test_config_factory_from_dict()
    test_config_factory_save_and_load()
    test_config_validator_service()
    test_dependency_injection_container()
    test_optimized_configs()
    
    print("All tests passed!")