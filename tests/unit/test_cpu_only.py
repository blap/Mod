"""
Test script to run CPU-only baseline tests with smaller models to avoid memory issues
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from benchmarks.benchmark_utils import benchmark_model_inference, benchmark_multimodal_task, profile_memory_usage, benchmark_generation


def run_cpu_tests_with_smaller_inputs():
    """
    Run CPU tests with smaller inputs to avoid memory issues
    """
    print("Running CPU tests with smaller inputs for Qwen3-VL model...")

    # Create model configuration
    config = Qwen3VLConfig()

    # Create model instance
    print("Creating model...")
    model = Qwen3VLForConditionalGeneration(config)

    # Move model to CPU
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model has {config.num_hidden_layers} layers and {config.num_attention_heads} attention heads")

    # Test with smaller inputs to avoid memory issues
    batch_size = 1
    seq_len = 16  # Reduced from 32
    vocab_size = config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size)  # Use full size image

    input_data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values
    }

    print(f"Input shapes - input_ids: {input_ids.shape}, pixel_values: {pixel_values.shape}")

    # Test 1: Basic inference performance
    print("\n1. Testing basic inference performance...")

    try:
        inference_results = benchmark_model_inference(
            model,
            input_data,
            num_runs=3,  # Reduced from 10
            warmup_runs=1,  # Reduced from 3
            device=device
        )
        print(f"Inference results: {inference_results}")
    except Exception as e:
        print(f"Error in inference test: {e}")
        import traceback
        traceback.print_exc()

    # Test 2: Multimodal task performance
    print("\n2. Testing multimodal task performance...")

    try:
        multimodal_results = benchmark_multimodal_task(
            model,
            text_input=input_ids,
            image_input=pixel_values,
            num_runs=3,  # Reduced from 10
            warmup_runs=1  # Reduced from 3
        )
        print(f"Multimodal results: {multimodal_results}")
    except Exception as e:
        print(f"Error in multimodal test: {e}")
        import traceback
        traceback.print_exc()

    # Test 3: Memory profiling
    print("\n3. Profiling memory usage...")

    try:
        memory_results = profile_memory_usage(model, input_data, device=device)
        print(f"Memory results: {memory_results}")
    except Exception as e:
        print(f"Error in memory profiling: {e}")
        import traceback
        traceback.print_exc()

    # Test 4: Text generation performance
    print("\n4. Testing text generation performance...")

    try:
        generation_results = benchmark_generation(
            model,
            input_ids=input_ids,
            max_new_tokens=10,  # Reduced from 20
            num_runs=2,  # Reduced from 5
            warmup_runs=1  # Reduced from 2
        )
        print(f"Generation results: {generation_results}")
    except Exception as e:
        print(f"Error in generation test: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("CPU TEST SUMMARY")
    print("="*60)
    if 'inference_results' in locals():
        print(f"Average inference time: {inference_results['avg_time']:.4f}s")
        print(f"Throughput: {inference_results['throughput_samples_per_sec']:.2f} samples/sec")
    if 'multimodal_results' in locals():
        print(f"Average multimodal time: {multimodal_results['avg_time']:.4f}s")
    if 'memory_results' in locals():
        print(f"CPU memory increase: {memory_results['cpu_memory_increase_mb']:.2f} MB")
    if 'generation_results' in locals():
        print(f"Average generation time: {generation_results['avg_generation_time']:.4f}s")
        print(f"Tokens per second: {generation_results['avg_tokens_per_sec']:.2f}")
    print(f"Model capacity: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")
    print("="*60)

    return {
        'inference': inference_results if 'inference_results' in locals() else None,
        'multimodal': multimodal_results if 'multimodal_results' in locals() else None,
        'memory': memory_results if 'memory_results' in locals() else None,
        'generation': generation_results if 'generation_results' in locals() else None,
        'capacity': {'layers': config.num_hidden_layers, 'heads': config.num_attention_heads}
    }


if __name__ == "__main__":
    results = run_cpu_tests_with_smaller_inputs()
    print("\nCPU tests completed!")