"""
Comprehensive validation tests for the NAS system integration with Qwen3-VL architecture.
"""
import torch
import torch.nn as nn
import numpy as np
from nas_system import Qwen3VLNeuralArchitectureSearch, LayerConfig
from hardware_optimizer import HardwareOptimizer
from compatibility_tests import verify_model_capacity, verify_nas_config_capacity


def test_end_to_end_integration():
    """Test the complete integration of NAS system with Qwen3-VL."""
    print("Testing end-to-end integration...")
    
    # Create NAS system
    nas_system = Qwen3VLNeuralArchitectureSearch(
        num_layers=8,  # Using fewer layers for faster testing
        base_hidden_size=512,
        base_num_heads=8,  # Using smaller number for testing
        base_intermediate_size=2048
    )
    
    # Test with text input
    text_input = torch.randint(0, 1000, (1, 32))
    text_configs = nas_system.search_optimal_architecture(
        input_data=text_input,
        input_type="text",
        num_search_steps=2,
        num_candidates_per_step=2
    )
    
    print(f"Generated {len(text_configs)} text configurations")
    assert len(text_configs) == 8, f"Expected 8 configs, got {len(text_configs)}"
    
    # Test with vision input
    vision_input = torch.randn(1, 3, 224, 224)
    vision_configs = nas_system.search_optimal_architecture(
        input_data=vision_input,
        input_type="vision",
        num_search_steps=2,
        num_candidates_per_step=2
    )
    
    print(f"Generated {len(vision_configs)} vision configurations")
    assert len(vision_configs) == 8, f"Expected 8 configs, got {len(vision_configs)}"
    
    # Verify configurations maintain constraints
    for i, config in enumerate(text_configs):
        assert config.hidden_size % config.num_attention_heads == 0, \
            f"Text config {i}: hidden_size not divisible by num_heads"
    
    for i, config in enumerate(vision_configs):
        assert config.hidden_size % config.num_attention_heads == 0, \
            f"Vision config {i}: hidden_size not divisible by num_heads"
    
    print("OK - End-to-end integration test passed")


def test_hardware_optimization():
    """Test hardware-specific optimizations."""
    print("\nTesting hardware optimization...")
    
    optimizer = HardwareOptimizer()
    
    # Verify hardware detection
    print(f"Target hardware detected: {optimizer.is_target_hardware}")
    print(f"CPU cores: {optimizer.cpu_count}")
    print(f"Memory (GB): {optimizer.memory_gb:.2f}")
    print(f"GPU available: {optimizer.has_cuda}")
    if optimizer.has_cuda:
        print(f"GPU memory (GB): {optimizer.gpu_memory_gb:.2f}")
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(512, 512)
    
    model = SimpleModel()
    
    # Apply memory optimizations
    optimizer.apply_memory_optimizations(model)
    
    # Get inference settings
    settings = optimizer.get_inference_settings()
    print(f"Inference settings: {settings}")
    
    # Get optimal batch size
    batch_size = optimizer.get_optimal_batch_size(sequence_length=256)
    print(f"Optimal batch size for seq_len=256: {batch_size}")
    
    print("OK - Hardware optimization test passed")


def test_capacity_preservation_comprehensive():
    """Comprehensive test for capacity preservation."""
    print("\nTesting comprehensive capacity preservation...")
    
    # Create NAS system with specific capacity requirements
    nas_system = Qwen3VLNeuralArchitectureSearch(
        num_layers=4,
        base_hidden_size=1024,
        base_num_heads=16,  # Using 16 for this test to make division easier
        base_intermediate_size=4096
    )
    
    # Override search space to ensure capacity preservation
    nas_system.search_space.min_num_heads = 16
    nas_system.search_space.min_hidden_size = 1024
    
    # Generate configurations
    text_input = torch.randint(0, 1000, (1, 16))
    configs = nas_system.search_optimal_architecture(
        input_data=text_input,
        input_type="text",
        num_search_steps=2,
        num_candidates_per_step=2
    )
    
    # Verify each configuration
    for i, config in enumerate(configs):
        print(f"Config {i}: hidden_size={config.hidden_size}, heads={config.num_attention_heads}, "
              f"intermediate={config.intermediate_size}")
        # The hidden size must be divisible by the number of heads
        assert config.hidden_size % config.num_attention_heads == 0, \
            f"Config {i} hidden_size not divisible by heads"
        # For 16 heads, the minimum hidden size should be at least 256 (16*16), but we want reasonable size
        assert config.hidden_size >= 256, f"Config {i} hidden_size too small: {config.hidden_size}"
        assert config.num_attention_heads >= 16, f"Config {i} heads too small: {config.num_attention_heads}"
    
    # Verify overall capacity
    verification = verify_nas_config_capacity(configs, expected_layers=4, expected_attention_heads=16)
    print(f"Verification result: {verification['capacity_preserved']}")
    assert verification['capacity_preserved'], "Capacity not preserved"
    
    print("OK - Comprehensive capacity preservation test passed")


def test_performance_prediction():
    """Test the performance prediction capabilities."""
    print("\nTesting performance prediction...")
    
    # Create NAS system
    nas_system = Qwen3VLNeuralArchitectureSearch(
        num_layers=4,
        base_hidden_size=256,
        base_num_heads=8,
        base_intermediate_size=1024
    )
    
    # Generate a few configurations
    configs1 = [
        LayerConfig(layer_type="attention", hidden_size=256, num_attention_heads=8, intermediate_size=1024, layer_idx=i)
        for i in range(4)
    ]
    
    configs2 = [
        LayerConfig(layer_type="attention", hidden_size=512, num_attention_heads=8, intermediate_size=2048, layer_idx=i)
        for i in range(4)
    ]
    
    # Get performance predictions
    perf1 = nas_system.performance_predictor.predict_performance(configs1).item()
    perf2 = nas_system.performance_predictor.predict_performance(configs2).item()
    
    print(f"Performance prediction for config1: {perf1:.4f}")
    print(f"Performance prediction for config2: {perf2:.4f}")
    
    # Both should be valid predictions (between 0 and 1)
    assert 0 <= perf1 <= 1, f"Invalid performance prediction: {perf1}"
    assert 0 <= perf2 <= 1, f"Invalid performance prediction: {perf2}"
    
    print("OK - Performance prediction test passed")


def test_model_layer_creation():
    """Test that configurations can create valid model layers."""
    print("\nTesting model layer creation...")
    
    from nas_system import create_transformer_layer_from_config
    
    # Create various configurations
    configs = [
        LayerConfig(layer_type="attention", hidden_size=256, num_attention_heads=8, intermediate_size=1024, layer_idx=0),
        LayerConfig(layer_type="attention", hidden_size=512, num_attention_heads=16, intermediate_size=2048, layer_idx=1),
        LayerConfig(layer_type="attention", hidden_size=384, num_attention_heads=12, intermediate_size=1536, layer_idx=2),
    ]
    
    # Create layers from configurations
    layers = []
    for i, config in enumerate(configs):
        layer = create_transformer_layer_from_config(config)
        layers.append(layer)
        print(f"Created layer {i} with hidden_size={config.hidden_size}, heads={config.num_attention_heads}")
        
        # Verify layer properties
        assert layer.hidden_size == config.hidden_size
        assert layer.num_attention_heads == config.num_attention_heads
        
        # Test forward pass with dummy input
        dummy_input = torch.randn(1, 10, config.hidden_size)
        output, attn_weights = layer(dummy_input)
        assert output.shape == dummy_input.shape, f"Output shape mismatch: {output.shape} vs {dummy_input.shape}"
        print(f"  Forward pass successful, output shape: {output.shape}")
    
    print("OK - Model layer creation test passed")


def run_comprehensive_validation():
    """Run all comprehensive validation tests."""
    print("Running comprehensive validation tests for NAS system...")
    print("=" * 70)

    test_end_to_end_integration()
    test_hardware_optimization()
    test_capacity_preservation_comprehensive()
    test_performance_prediction()
    test_model_layer_creation()

    print("\n" + "=" * 70)
    print("ALL COMPREHENSIVE VALIDATION TESTS PASSED!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = run_comprehensive_validation()
    if success:
        print("\nOK - All validation tests completed successfully!")
    else:
        print("\nFAILED - Some validation tests failed!")
        exit(1)