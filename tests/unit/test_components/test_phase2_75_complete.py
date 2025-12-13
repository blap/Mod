"""
Complete test for Phase 2.75: Memory-Efficient Transformer Variants
"""
import torch
from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def test_phase2_75_complete_implementation():
    """Test the complete Phase 2.75 implementation"""
    print("Testing Phase 2.75: Memory-Efficient Transformer Variants")
    
    # Test 1: Configuration with all Phase 2.75 features enabled
    config = Qwen3VLConfig()
    config.use_moe = True
    config.use_flash_attention_2 = True
    config.use_parameter_sharing = True
    config.moe_num_experts = 4
    config.moe_top_k = 2
    
    # Reduce dimensions for testing
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_attention_heads = 4
    config.num_hidden_layers = 6  # Small number for testing
    
    print(f"Configured model with:")
    print(f"  - {config.num_hidden_layers} transformer layers")
    print(f"  - {config.num_attention_heads} attention heads")
    print(f"  - MoE enabled: {config.use_moe} ({config.moe_num_experts} experts, top-{config.moe_top_k})")
    print(f"  - FlashAttention-2 enabled: {config.use_flash_attention_2}")
    print(f"  - Parameter sharing enabled: {config.use_parameter_sharing}")
    
    # Create model
    model = Qwen3VLForConditionalGeneration(config)
    print("Model created successfully")
    
    # Test 2: Forward pass with text input
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print("Testing forward pass with text input...")
    with torch.no_grad():
        text_outputs = model(input_ids=input_ids)
        print(f"Text input forward pass successful. Output shape: {text_outputs.shape}")
    
    # Test 3: Forward pass with multimodal input (text + image)
    print("Testing forward pass with multimodal input...")
    pixel_values = torch.randn(batch_size, 3, 224, 224)  # Simulated image input
    
    with torch.no_grad():
        multimodal_outputs = model(input_ids=input_ids, pixel_values=pixel_values)
        print(f"Multimodal forward pass successful. Output shape: {multimodal_outputs.shape}")
    
    # Test 4: Verify model capacity is preserved
    assert len(model.language_model.layers) == config.num_hidden_layers, \
        f"Expected {config.num_hidden_layers} layers, got {len(model.language_model.layers)}"
    print(f"V Model has correct number of layers: {len(model.language_model.layers)}")

    # Test 5: Test generation
    print("Testing text generation...")
    generated_ids = model.generate(
        input_ids=input_ids[:1],  # Use first batch only
        max_new_tokens=5,
        do_sample=False
    )
    print(f"Generation successful. Generated shape: {generated_ids.shape}")

    # Test 6: Test with different configurations
    print("Testing different configuration combinations...")

    # Test without MoE but with FlashAttention
    config2 = Qwen3VLConfig()
    config2.use_moe = False
    config2.use_flash_attention_2 = True
    config2.use_parameter_sharing = False
    config2.hidden_size = 64
    config2.num_attention_heads = 2
    config2.num_hidden_layers = 4

    model2 = Qwen3VLForConditionalGeneration(config2)
    with torch.no_grad():
        output2 = model2(input_ids=input_ids[:1, :8])
        print(f"Model without MoE but with FlashAttention works. Output shape: {output2.shape}")

    # Test with parameter sharing only
    config3 = Qwen3VLConfig()
    config3.use_moe = False
    config3.use_flash_attention_2 = False
    config3.use_parameter_sharing = True
    config3.hidden_size = 64
    config3.num_attention_heads = 2
    config3.num_hidden_layers = 4

    model3 = Qwen3VLForConditionalGeneration(config3)
    with torch.no_grad():
        output3 = model3(input_ids=input_ids[:1, :8])
        print(f"Model with parameter sharing works. Output shape: {output3.shape}")

    print("\nV All Phase 2.75 implementation tests passed!")


def test_memory_efficiency():
    """Test memory efficiency improvements"""
    import time
    import psutil
    import gc
    
    print("\nTesting memory efficiency...")
    
    # Create model without Phase 2.75 features
    config_baseline = Qwen3VLConfig()
    config_baseline.use_moe = False
    config_baseline.use_flash_attention_2 = False
    config_baseline.use_parameter_sharing = False
    config_baseline.hidden_size = 128
    config_baseline.num_attention_heads = 4
    config_baseline.num_hidden_layers = 4
    
    baseline_model = Qwen3VLForConditionalGeneration(config_baseline)
    
    # Create model with Phase 2.75 features
    config_optimized = Qwen3VLConfig()
    config_optimized.use_moe = True
    config_optimized.use_flash_attention_2 = True
    config_optimized.use_parameter_sharing = True
    config_optimized.moe_num_experts = 4
    config_optimized.moe_top_k = 2
    config_optimized.hidden_size = 128
    config_optimized.num_attention_heads = 4
    config_optimized.num_hidden_layers = 4
    
    optimized_model = Qwen3VLForConditionalGeneration(config_optimized)
    
    # Create test input
    input_ids = torch.randint(0, config_baseline.vocab_size, (1, 32))
    
    # Measure memory usage for baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    baseline_memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    with torch.no_grad():
        baseline_output = baseline_model(input_ids=input_ids)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    baseline_memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Measure memory usage for optimized
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    optimized_memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    with torch.no_grad():
        optimized_output = optimized_model(input_ids=input_ids)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    optimized_memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"Baseline memory usage: {baseline_memory_after - baseline_memory_before:.2f} MB")
    print(f"Optimized memory usage: {optimized_memory_after - optimized_memory_before:.2f} MB")
    
    # Note: Actual memory savings depend on implementation details
    # The MoE should reduce active parameters during inference
    print("Memory efficiency test completed")


if __name__ == "__main__":
    test_phase2_75_complete_implementation()
    test_memory_efficiency()
    print("\nV All Phase 2.75 tests completed successfully!")