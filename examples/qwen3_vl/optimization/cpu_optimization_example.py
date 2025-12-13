"""
Example usage of CPU optimization techniques for Qwen3-VL model
"""
import sys
import os
# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

import torch
from PIL import Image
import numpy as np
from src.qwen3_vl.optimization.cpu_optimizations import (
    CPUOptimizationConfig,
    apply_cpu_optimizations,
    OptimizedInferencePipeline
)


def create_dummy_model():
    """Create a dummy model for demonstration."""
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device('cpu')
        
        def parameters(self):
            return iter([torch.nn.Parameter(torch.randn(10, 100))])
        
        def generate(self, input_ids=None, pixel_values=None, attention_mask=None, **kwargs):
            # Simulate generation by returning dummy outputs
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            max_new_tokens = kwargs.get('max_new_tokens', 10)
            return torch.randint(0, 1000, (batch_size, max_new_tokens))
    
    return DummyModel()


def create_dummy_tokenizer():
    """Create a dummy tokenizer for demonstration."""
    class DummyTokenizer:
        def __call__(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            
            batch_size = len(texts)
            seq_len = kwargs.get('max_length', 32)
            
            # Create dummy token IDs and attention masks
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
    
    return DummyTokenizer()


def main():
    """Main example demonstrating CPU optimizations."""
    print("Qwen3-VL CPU Optimization Example")
    print("=" * 40)
    
    # Create model and tokenizer (in practice, load real ones)
    model = create_dummy_model()
    tokenizer = create_dummy_tokenizer()
    
    # Configure CPU optimizations for Intel i5-10210U + NVIDIA SM61
    config = CPUOptimizationConfig(
        num_preprocess_workers=4,  # Match i5-10210U physical cores
        preprocess_batch_size=4,   # Moderate batch size for SM61
        image_resize_size=(224, 224),
        max_text_length=512,
        cpu_gpu_overlap=True,
        transfer_async=True,
        memory_threshold=0.7,      # Conservative for SM61
        clear_cache_interval=5     # Regular cleanup
    )
    
    print(f"Configuration:")
    print(f"  - Preprocess workers: {config.num_preprocess_workers}")
    print(f"  - Batch size: {config.preprocess_batch_size}")
    print(f"  - Memory threshold: {config.memory_threshold}")
    print(f"  - Async transfer: {config.transfer_async}")
    print()
    
    # Apply CPU optimizations
    print("Applying CPU optimizations...")
    pipeline, create_loader_fn = apply_cpu_optimizations(
        model,
        tokenizer,
        num_preprocess_workers=config.num_preprocess_workers,
        preprocess_batch_size=config.preprocess_batch_size,
        memory_threshold=config.memory_threshold
    )
    print("[OK] Optimizations applied successfully")
    print()

    # Create sample inputs
    texts = [
        "Describe this image in detail",
        "What objects do you see in this picture?",
        "Explain the scene shown here"
    ]

    images = []
    for i in range(len(texts)):
        # Create random images for demonstration
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)

    print(f"Processing {len(texts)} text-image pairs...")

    # Process with optimized pipeline
    responses = pipeline.preprocess_and_infer(
        texts,
        images,
        tokenizer=tokenizer,
        max_new_tokens=15,
        do_sample=True,
        temperature=0.7
    )

    print("[OK] Processing completed successfully")
    print()

    # Display results
    print("Results:")
    for i, (text, response) in enumerate(zip(texts, responses)):
        print(f"  Input {i+1}: {text}")
        print(f"  Output {i+1}: {response}")
        print()

    # Display performance metrics
    metrics = pipeline.get_performance_metrics()
    print("Performance Metrics:")
    print(f"  - Avg preprocess time: {metrics['avg_preprocess_time']:.4f}s")
    print(f"  - Avg inference time: {metrics['avg_inference_time']:.4f}s")
    print(f"  - Total calls: {metrics['total_calls']}")
    print()

    print("CPU optimization example completed successfully! :)")
    print()
    print("Key benefits achieved:")
    print("  - Multithreaded preprocessing for faster data preparation")
    print("  - Efficient CPU-GPU coordination to minimize transfer bottlenecks")
    print("  - Memory management to prevent GPU memory issues")
    print("  - Asynchronous operations to maximize throughput")


if __name__ == "__main__":
    main()