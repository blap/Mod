"""
Comprehensive multimodal benchmark tests for Qwen3-VL model
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qwen3_vl.core.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from benchmarks.benchmark_utils import benchmark_multimodal_task, profile_memory_usage


def run_multimodal_benchmarks():
    """
    Run comprehensive multimodal benchmark tests for Qwen3-VL model.
    """
    print("Running multimodal benchmark tests for Qwen3-VL model...")

    # Create model configuration - ensure full capacity
    config = Qwen3VLConfig()

    # Verify capacity is preserved
    assert config.num_hidden_layers == 32, f"Expected 32 layers, got {config.num_hidden_layers}"
    assert config.num_attention_heads == 32, f"Expected 32 attention heads, got {config.num_attention_heads}"

    print(f"Configuration verified: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")

    # Test on CPU only due to memory constraints
    devices_to_test = [torch.device('cpu')]

    # Check if GPU is available and has enough memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # Estimate model size: 3.4B parameters * 2 bytes (for float16) / (1024^3) ~ 6.4GB
        estimated_model_size_gb = 6.8  # Conservative estimate for model size
        if gpu_memory_gb > estimated_model_size_gb + 1:  # Add 1GB buffer
            print(f"GPU has enough memory ({gpu_memory_gb:.2f}GB available, {estimated_model_size_gb:.2f}GB needed)")
            devices_to_test.append(torch.device('cuda'))
        else:
            print(f"GPU memory insufficient ({gpu_memory_gb:.2f}GB available, ~{estimated_model_size_gb:.2f}GB needed)")
            print("Skipping GPU tests due to memory constraints")

    results = {}

    for device in devices_to_test:
        print(f"\nTesting on {device}...")
        # Create a fresh model instance for each device to avoid memory issues
        model = Qwen3VLForConditionalGeneration(config)
        model = model.to(device)
        model.eval()

        # Create test inputs
        batch_size = 1
        seq_len = 64
        vocab_size = config.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        pixel_values = torch.randn(
            batch_size, 3, config.vision_image_size, config.vision_image_size
        ).to(device)
        
        # Test 1: Multimodal inference with different input sizes
        print(f"  1. Testing multimodal inference...")
        
        multimodal_results = benchmark_multimodal_task(
            model,
            text_input=input_ids,
            image_input=pixel_values,
            num_runs=10,
            warmup_runs=3
        )
        
        print(f"    Results: {multimodal_results}")
        
        # Test 2: Memory profiling for multimodal tasks
        print(f"  2. Profiling memory usage...")
        
        input_data = {
            'input_ids': input_ids,
            'pixel_values': pixel_values
        }
        
        memory_results = profile_memory_usage(model, input_data, device=device)
        print(f"    Memory results: {memory_results}")
        
        # Test 3: Different batch sizes
        print(f"  3. Testing different batch sizes...")
        
        batch_sizes = [1, 2, 4]
        batch_results = {}
        
        for bs in batch_sizes:
            try:
                bs_input_ids = torch.randint(0, vocab_size, (bs, seq_len)).to(device)
                bs_pixel_values = torch.randn(
                    bs, 3, config.vision_image_size, config.vision_image_size
                ).to(device)
                
                bs_results = benchmark_multimodal_task(
                    model,
                    text_input=bs_input_ids,
                    image_input=bs_pixel_values,
                    num_runs=5,
                    warmup_runs=2
                )
                
                batch_results[bs] = bs_results
                print(f"    Batch size {bs}: {bs_results['avg_time']:.4f}s avg")
            except RuntimeError as e:
                print(f"    Batch size {bs} failed: {str(e)}")
                batch_results[bs] = {'error': str(e)}
        
        # Test 4: Different sequence lengths
        print(f"  4. Testing different sequence lengths...")
        
        seq_lengths = [16, 32, 64, 128]
        seq_results = {}
        
        for sl in seq_lengths:
            try:
                sl_input_ids = torch.randint(0, vocab_size, (1, sl)).to(device)
                sl_pixel_values = torch.randn(
                    1, 3, config.vision_image_size, config.vision_image_size
                ).to(device)
                
                sl_results = benchmark_multimodal_task(
                    model,
                    text_input=sl_input_ids,
                    image_input=sl_pixel_values,
                    num_runs=5,
                    warmup_runs=2
                )
                
                seq_results[sl] = sl_results
                print(f"    Seq len {sl}: {sl_results['avg_time']:.4f}s avg")
            except RuntimeError as e:
                print(f"    Seq len {sl} failed: {str(e)}")
                seq_results[sl] = {'error': str(e)}
        
        # Store results for this device
        results[str(device)] = {
            'multimodal': multimodal_results,
            'memory': memory_results,
            'batch_sizes': batch_results,
            'sequence_lengths': seq_results
        }
    
    # Summary
    print("\n" + "="*70)
    print("MULTIMODAL BENCHMARK SUMMARY")
    print("="*70)
    
    for device, device_results in results.items():
        print(f"\n{device.upper()} RESULTS:")
        print(f"  Average multimodal time: {device_results['multimodal']['avg_time']:.4f}s")
        print(f"  Throughput: {device_results['multimodal']['throughput_samples_per_sec']:.2f} samples/sec")
        if device == 'cuda':
            print(f"  GPU memory increase: {device_results['memory']['gpu_memory_increase_mb']:.2f} MB")
            print(f"  GPU peak memory: {device_results['memory']['gpu_peak_memory_mb']:.2f} MB")
        print(f"  CPU memory increase: {device_results['memory']['cpu_memory_increase_mb']:.2f} MB")
    
    print("\nCAPACITY VERIFICATION:")
    print(f"  Layers: {config.num_hidden_layers} (preserved: {config.num_hidden_layers == 32})")
    print(f"  Attention Heads: {config.num_attention_heads} (preserved: {config.num_attention_heads == 32})")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = run_multimodal_benchmarks()
    print("\nMultimodal benchmark tests completed successfully!")