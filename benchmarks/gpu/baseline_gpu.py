"""
GPU baseline performance tests for Qwen3-VL model
"""
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from benchmarks.benchmark_utils import benchmark_model_inference, benchmark_multimodal_task, profile_memory_usage, benchmark_generation


def run_gpu_baseline_tests():
    """
    Run comprehensive GPU baseline performance tests for Qwen3-VL model.
    """
    print("Running GPU baseline performance tests for Qwen3-VL model...")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU instead.")
        return run_cpu_baseline_tests()  # Fallback to CPU tests
    
    # Create model configuration
    config = Qwen3VLConfig()
    
    # Create model instance
    print("Creating model...")
    model = Qwen3VLForConditionalGeneration(config)
    
    # Move model to GPU
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Model has {config.num_hidden_layers} layers and {config.num_attention_heads} attention heads")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Test 1: Basic inference performance
    print("\n1. Testing basic inference performance...")
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 32
    vocab_size = config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones((batch_size, seq_len)).to(device)
    pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size).to(device)
    
    input_data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values
    }
    
    # Run inference benchmark
    inference_results = benchmark_model_inference(
        model, 
        input_data, 
        num_runs=20,  # More runs on GPU for better statistics
        warmup_runs=5, 
        device=device
    )
    
    print(f"Inference results: {inference_results}")
    
    # Test 2: Multimodal task performance
    print("\n2. Testing multimodal task performance...")
    
    multimodal_results = benchmark_multimodal_task(
        model,
        text_input=input_ids,
        image_input=pixel_values,
        num_runs=20,
        warmup_runs=5
    )
    
    print(f"Multimodal results: {multimodal_results}")
    
    # Test 3: Memory profiling
    print("\n3. Profiling memory usage...")
    
    memory_results = profile_memory_usage(model, input_data, device=device)
    print(f"Memory results: {memory_results}")
    
    # Test 4: Text generation performance
    print("\n4. Testing text generation performance...")
    
    generation_results = benchmark_generation(
        model,
        input_ids=input_ids,
        max_new_tokens=20,
        num_runs=10,
        warmup_runs=3
    )
    
    print(f"Generation results: {generation_results}")
    
    # Test 5: Capacity verification
    print("\n5. Verifying model capacity...")
    
    # Verify layer count
    layer_count = config.num_hidden_layers
    head_count = config.num_attention_heads
    
    print(f"Model has {layer_count} layers and {head_count} attention heads")
    print(f"Capacity preserved: {layer_count == 32 and head_count == 32}")
    
    # Test 6: GPU-specific metrics
    print("\n6. Testing GPU-specific metrics...")
    
    gpu_metrics = {
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
        'gpu_compute_capability': torch.cuda.get_device_capability(0),
        'current_gpu_memory': torch.cuda.memory_allocated(device) / 1024**2,  # MB
        'peak_gpu_memory': torch.cuda.max_memory_reserved(device) / 1024**2  # MB
    }
    
    print(f"GPU metrics: {gpu_metrics}")
    
    # Summary
    print("\n" + "="*60)
    print("GPU BASELINE PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Average inference time: {inference_results['avg_time']:.4f}s")
    print(f"Throughput: {inference_results['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"Average multimodal time: {multimodal_results['avg_time']:.4f}s")
    print(f"GPU memory increase: {memory_results['gpu_memory_increase_mb']:.2f} MB")
    print(f"GPU peak memory: {memory_results['gpu_peak_memory_mb']:.2f} MB")
    print(f"Average generation time: {generation_results['avg_generation_time']:.4f}s")
    print(f"Tokens per second: {generation_results['avg_tokens_per_sec']:.2f}")
    print(f"Model capacity: {layer_count} layers, {head_count} heads (preserved: {layer_count == 32 and head_count == 32})")
    print(f"GPU: {gpu_metrics['gpu_name']}")
    print(f"GPU Memory: {gpu_metrics['gpu_memory_total']:.2f} GB")
    print("="*60)
    
    return {
        'inference': inference_results,
        'multimodal': multimodal_results,
        'memory': memory_results,
        'generation': generation_results,
        'capacity': {'layers': layer_count, 'heads': head_count},
        'gpu_metrics': gpu_metrics
    }


def run_cpu_baseline_tests():
    """
    Fallback CPU tests when GPU is not available.
    """
    print("Running CPU baseline performance tests for Qwen3-VL model...")
    
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
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 32
    vocab_size = config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size)
    
    input_data = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values
    }
    
    # Run inference benchmark
    inference_results = benchmark_model_inference(
        model, 
        input_data, 
        num_runs=10, 
        warmup_runs=3, 
        device=device
    )
    
    print(f"Inference results: {inference_results}")
    
    # Summary
    print("\n" + "="*60)
    print("CPU FALLBACK PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Average inference time: {inference_results['avg_time']:.4f}s")
    print(f"Throughput: {inference_results['throughput_samples_per_sec']:.2f} samples/sec")
    print("="*60)
    
    return {
        'inference': inference_results,
        'capacity': {'layers': config.num_hidden_layers, 'heads': config.num_attention_heads}
    }


if __name__ == "__main__":
    results = run_gpu_baseline_tests()
    print("\nGPU baseline tests completed successfully!")