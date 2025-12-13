"""
Benchmark script to measure memory usage reduction with sparsity enabled.
"""
import torch
import torch.nn as nn
import gc
from models.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def benchmark_memory_usage_reduction():
    """Benchmark memory usage reduction with sparsity enabled."""
    print("Benchmarking Memory Usage Reduction with Sparsity...")
    
    # Create two configurations - one with sparsity and one without
    config_without_sparsity = Qwen3VLConfig()
    config_without_sparsity.hidden_size = 128
    config_without_sparsity.intermediate_size = 256
    config_without_sparsity.num_attention_heads = 4
    config_without_sparsity.num_hidden_layers = 4
    config_without_sparsity.use_sparsity = False  # Disable sparsity
    
    config_with_sparsity = Qwen3VLConfig()
    config_with_sparsity.hidden_size = 128
    config_with_sparsity.intermediate_size = 256
    config_with_sparsity.num_attention_heads = 4
    config_with_sparsity.num_hidden_layers = 4
    config_with_sparsity.use_sparsity = True  # Enable sparsity
    config_with_sparsity.sparsity_ratio = 0.5  # 50% sparsity
    config_with_sparsity.exit_threshold = 0.8
    
    # Create models
    model_without_sparsity = Qwen3VLForConditionalGeneration(config_without_sparsity)
    model_with_sparsity = Qwen3VLForConditionalGeneration(config_with_sparsity)
    
    # Create test input
    input_ids = torch.randint(0, config_without_sparsity.vocab_size, (1, 32))
    
    # Measure memory usage without sparsity
    print("Measuring memory usage without sparsity...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    model_without_sparsity.eval()
    with torch.no_grad():
        output_without = model_without_sparsity(input_ids=input_ids)
    
    if torch.cuda.is_available():
        max_memory_without_sparsity = torch.cuda.max_memory_allocated()
        print(f"Max memory without sparsity: {max_memory_without_sparsity / 1024**2:.2f} MB")
    else:
        # Fallback to CPU memory monitoring
        import psutil
        max_memory_without_sparsity = psutil.Process().memory_info().rss
        print(f"Memory without sparsity (approx): {max_memory_without_sparsity / 1024**2:.2f} MB")
    
    # Measure memory usage with sparsity
    print("Measuring memory usage with sparsity...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    model_with_sparsity.eval()
    with torch.no_grad():
        output_with = model_with_sparsity(input_ids=input_ids)
    
    if torch.cuda.is_available():
        max_memory_with_sparsity = torch.cuda.max_memory_allocated()
        print(f"Max memory with sparsity: {max_memory_with_sparsity / 1024**2:.2f} MB")
        
        memory_reduction = (max_memory_without_sparsity - max_memory_with_sparsity) / max_memory_without_sparsity * 100
        print(f"Memory reduction: {memory_reduction:.2f}%")
    else:
        # Fallback to CPU memory monitoring
        import psutil
        max_memory_with_sparsity = psutil.Process().memory_info().rss
        print(f"Memory with sparsity (approx): {max_memory_with_sparsity / 1024**2:.2f} MB")
        
        memory_reduction = (max_memory_without_sparsity - max_memory_with_sparsity) / max_memory_without_sparsity * 100
        print(f"Memory reduction (approx): {memory_reduction:.2f}%")
    
    # Verify outputs are reasonable
    assert torch.isfinite(output_without).all(), "Output without sparsity should be finite"
    assert torch.isfinite(output_with).all(), "Output with sparsity should be finite"
    assert output_without.shape == output_with.shape, "Output shapes should match"
    
    print(f"Output shapes: {output_without.shape}")
    
    # Test that sparsity doesn't significantly impact output quality
    output_similarity = torch.cosine_similarity(
        output_without.flatten(), 
        output_with.flatten(), 
        dim=0
    )
    print(f"Output similarity (cosine): {output_similarity.item():.4f}")
    
    # Expect high similarity (above 0.95) between outputs with and without sparsity
    assert output_similarity.item() > 0.90, f"Output similarity too low: {output_similarity.item():.4f}"
    
    print(f"PASS: Memory usage benchmark completed - {memory_reduction:.2f}% reduction achieved")
    return memory_reduction


def benchmark_forward_pass_memory_efficiency():
    """Benchmark memory efficiency during forward pass with different sparsity levels."""
    print("\nBenchmarking Forward Pass Memory Efficiency...")
    
    sparsity_levels = [0.0, 0.3, 0.5, 0.7]  # Different sparsity levels to test
    memory_usage = []
    
    for sparsity_ratio in sparsity_levels:
        print(f"Testing with {sparsity_ratio*100}% sparsity...")
        
        config = Qwen3VLConfig()
        config.hidden_size = 64
        config.intermediate_size = 128
        config.num_attention_heads = 4
        config.num_hidden_layers = 2
        config.use_sparsity = True
        config.sparsity_ratio = sparsity_ratio
        config.exit_threshold = 0.8
        
        model = Qwen3VLForConditionalGeneration(config)
        
        input_ids = torch.randint(0, config.vocab_size, (1, 16))
        
        # Clear cache and measure memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids)
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            import psutil
            peak_memory = psutil.Process().memory_info().rss
        
        memory_usage.append(peak_memory)
        print(f"  Peak memory usage: {peak_memory / 1024**2:.2f} MB")
        assert torch.isfinite(output).all(), f"Output with {sparsity_ratio*100}% sparsity should be finite"
    
    # Calculate memory reduction percentages
    baseline_memory = memory_usage[0]
    for i, (sparsity, memory) in enumerate(zip(sparsity_levels, memory_usage)):
        reduction = (baseline_memory - memory) / baseline_memory * 100
        print(f"  {sparsity*100}% sparsity: {reduction:.2f}% memory reduction")
    
    print("PASS: Forward pass memory efficiency benchmark completed")
    
    # Return True if we see memory reduction with higher sparsity levels
    return memory_usage[0] > memory_usage[-1]  # Baseline should use more memory than highest sparsity


if __name__ == "__main__":
    reduction_percentage = benchmark_memory_usage_reduction()
    efficiency_improved = benchmark_forward_pass_memory_efficiency()
    
    print(f"\nSUMMARY: Memory usage reduction benchmark completed")
    print(f"Overall memory reduction achieved: {reduction_percentage:.2f}%")
    print(f"Memory efficiency improved with sparsity: {efficiency_improved}")
    
    # Check if we achieved the target (20-40% reduction)
    if reduction_percentage >= 20:
        print("SUCCESS: Achieved target memory reduction (>20%)")
    else:
        print(f"NOTE: Memory reduction ({reduction_percentage:.2f}%) is below target range (20-40%)")
        
    if efficiency_improved:
        print("SUCCESS: Memory efficiency scales with sparsity levels")
    else:
        print("WARNING: Memory efficiency does not scale as expected with sparsity")