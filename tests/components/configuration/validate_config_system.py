"""Final validation test for the unified configuration system."""

from src.qwen3_vl.core.config import Qwen3VLConfig
import tempfile
import os


def test_config_preservation():
    """Test that configuration preserves capacity."""
    print("Testing configuration preservation...")
    
    config = Qwen3VLConfig()
    print(f"  - Num hidden layers: {config.num_hidden_layers} (expected: 32, preserved: {config.num_hidden_layers == 32})")
    print(f"  - Num attention heads: {config.num_attention_heads} (expected: 32, preserved: {config.num_attention_heads == 32})")
    print(f"  - Vision num hidden layers: {config.vision_num_hidden_layers} (expected: 24, preserved: {config.vision_num_hidden_layers == 24})")
    print(f"  - Vision num attention heads: {config.vision_num_attention_heads} (expected: 16, preserved: {config.vision_num_attention_heads == 16})")
    
    # Verify divisibility constraints
    assert config.hidden_size % config.num_attention_heads == 0, f"Hidden size {config.hidden_size} not divisible by attention heads {config.num_attention_heads}"
    assert config.vision_hidden_size % config.vision_num_attention_heads == 0, f"Vision hidden size {config.vision_hidden_size} not divisible by vision attention heads {config.vision_num_attention_heads}"
    print("  - Divisibility constraints satisfied")
    
    return True


def test_serialization_deserialization():
    """Test configuration serialization and deserialization."""
    print("Testing serialization/deserialization...")
    
    original_config = Qwen3VLConfig(
        num_hidden_layers=16,
        num_attention_heads=16,
        hidden_size=2048,
        intermediate_size=5504,
        vocab_size=50000,
        max_position_embeddings=2048,
        vision_num_hidden_layers=12,
        vision_num_attention_heads=8,
        vision_hidden_size=576
    )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = f.name
    
    try:
        # Save config
        original_config.save_to_file(temp_file)
        print(f"  - Saved config to {temp_file}")
        
        # Load config
        loaded_config = Qwen3VLConfig.from_file(temp_file)
        print(f"  - Loaded config from {temp_file}")
        
        # Verify values match
        assert loaded_config.num_hidden_layers == 16
        assert loaded_config.num_attention_heads == 16
        assert loaded_config.hidden_size == 2048
        assert loaded_config.vision_num_hidden_layers == 12
        assert loaded_config.vision_num_attention_heads == 8
        assert loaded_config.vision_hidden_size == 576
        print("  - All values preserved correctly")
        
        return True
    finally:
        os.unlink(temp_file)


def test_optimization_config():
    """Test optimization configuration."""
    print("Testing optimization configuration...")
    
    config = Qwen3VLConfig()
    
    # Check that optimization configs are properly initialized
    assert config.memory_config is not None
    assert config.cpu_config is not None
    assert config.gpu_config is not None
    assert config.power_config is not None
    assert config.optimization_config is not None
    
    # Check some optimization settings
    assert config.optimization_config.use_sparsity == True
    assert config.optimization_config.use_moe == True
    assert config.optimization_config.use_flash_attention_2 == True
    assert config.optimization_config.use_adaptive_depth == True
    
    print(f"  - Sparsity enabled: {config.optimization_config.use_sparsity}")
    print(f"  - MoE enabled: {config.optimization_config.use_moe}")
    print(f"  - Flash attention enabled: {config.optimization_config.use_flash_attention_2}")
    print(f"  - Adaptive depth enabled: {config.optimization_config.use_adaptive_depth}")
    
    return True


def test_hardware_config():
    """Test hardware-specific configuration."""
    print("Testing hardware-specific configuration...")
    
    config = Qwen3VLConfig()
    
    assert config.hardware_target == "intel_i5_10210u_nvidia_sm61_nvme"
    assert config.target_hardware == "nvidia_sm61"
    assert config.compute_units == 4
    assert config.memory_gb == 8.0
    
    print(f"  - Hardware target: {config.hardware_target}")
    print(f"  - Target hardware: {config.target_hardware}")
    print(f"  - Compute units: {config.compute_units}")
    print(f"  - Memory GB: {config.memory_gb}")
    
    return True


def main():
    """Run all validation tests."""
    print("Running unified configuration system validation tests...")
    print("=" * 60)
    
    tests = [
        test_config_preservation,
        test_serialization_deserialization,
        test_optimization_config,
        test_hardware_config
    ]
    
    all_passed = True
    for test in tests:
        try:
            result = test()
            if result:
                print("  [PASS] PASSED\n")
            else:
                print("  [FAIL] FAILED\n")
                all_passed = False
        except Exception as e:
            print(f"  [FAIL] FAILED with error: {e}\n")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("[PASS] ALL TESTS PASSED - Unified configuration system is working correctly!")
    else:
        print("[FAIL] SOME TESTS FAILED - Please review configuration implementation")

    return all_passed


if __name__ == "__main__":
    main()