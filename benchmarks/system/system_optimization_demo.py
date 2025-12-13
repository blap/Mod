"""
Demonstration of System-Level Optimizations for Qwen3-VL Model
This script demonstrates the implementation of profiling, multi-threading improvements,
and resource scheduling techniques for the Qwen3-VL model.
"""

import torch
import torch.nn as nn
import time
import psutil
from typing import Dict, Any
import numpy as np

from src.qwen3_vl.optimization.system_level_optimizations import (
    SystemOptimizationConfig,
    SystemOptimizer,
    OptimizedInferencePipeline,
    apply_system_level_optimizations
)


def create_demo_model():
    """Create a demonstration model similar to Qwen3-VL architecture."""
    
    class DemoQwen3VLModel(nn.Module):
        def __init__(self, 
                     hidden_size=768, 
                     num_layers=12, 
                     num_heads=12, 
                     vocab_size=50257,
                     max_position_embeddings=512):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.num_heads = num_heads
            
            # Embedding layers
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
            self.pos_encoding = nn.Parameter(torch.randn(1, max_position_embeddings, hidden_size))
            
            # Transformer layers
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    batch_first=True,
                    dropout=0.1
                ) for _ in range(num_layers)
            ])
            
            # Final layer norm and output
            self.final_norm = nn.LayerNorm(hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            
            # Vision components (simplified)
            self.vision_projection = nn.Linear(512, hidden_size)  # Simplified vision encoder
            
        def forward(self, input_ids, attention_mask=None, pixel_values=None):
            # Token embeddings
            hidden_states = self.embed_tokens(input_ids)
            
            # Add position encoding (truncated if needed)
            seq_len = hidden_states.size(1)
            pos_encoding = self.pos_encoding[:, :seq_len, :]
            hidden_states = hidden_states + pos_encoding
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Convert attention mask to attention bias
                attention_bias = (1.0 - attention_mask.unsqueeze(1).unsqueeze(1)) * torch.finfo(torch.float).min
            else:
                attention_bias = None
            
            # Transformer layers
            for layer in self.layers:
                # Note: Standard PyTorch TransformerEncoderLayer doesn't take attention_bias
                # In a real implementation, we'd use a custom attention mechanism
                hidden_states = layer(hidden_states)
            
            # Final normalization and output
            hidden_states = self.final_norm(hidden_states)
            logits = self.lm_head(hidden_states)
            
            return logits
    
    return DemoQwen3VLModel()


def run_performance_comparison():
    """Run a performance comparison between optimized and non-optimized models."""
    
    print("Creating demonstration model...")
    model = create_demo_model()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model moved to: {device}")
    
    # Create sample inputs
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones((batch_size, seq_len)).to(device)
    
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    
    print(f"\nInput shape: {input_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Test non-optimized inference (baseline)
    print("\n--- Baseline Inference (No Optimizations) ---")
    model.eval()
    
    start_time = time.time()
    with torch.no_grad():
        baseline_outputs = model(**inputs)
    baseline_time = time.time() - start_time
    
    print(f"Baseline inference time: {baseline_time:.4f} seconds")
    print(f"Output shape: {baseline_outputs.shape}")
    
    # Apply system-level optimizations
    print("\n--- Applying System-Level Optimizations ---")
    config = SystemOptimizationConfig(
        num_compute_threads=4,
        num_io_threads=2,
        num_preprocess_threads=4,
        memory_limit_ratio=0.7,
        enable_profiling=True,
        scheduling_algorithm="load_balanced"
    )
    
    optimized_pipeline = apply_system_level_optimizations(model, config)
    
    # Run optimized inference
    print("\n--- Optimized Inference ---")
    start_time = time.time()
    optimized_outputs = optimized_pipeline.run_inference(inputs)
    optimized_time = time.time() - start_time
    
    print(f"Optimized inference time: {optimized_time:.4f} seconds")
    print(f"Output shape: {optimized_outputs.shape}")
    
    # Calculate performance improvement
    speedup = baseline_time / optimized_time if optimized_time > 0 else float('inf')
    time_improvement = ((baseline_time - optimized_time) / baseline_time) * 100 if baseline_time > 0 else 0
    
    print(f"\n--- Performance Comparison ---")
    print(f"Baseline time: {baseline_time:.4f}s")
    print(f"Optimized time: {optimized_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Time improvement: {time_improvement:.2f}%")
    
    # Get detailed metrics
    metrics = optimized_pipeline.get_performance_metrics()
    print(f"\n--- Detailed Performance Metrics ---")
    print(f"Average inference time: {metrics['avg_inference_time']:.4f}s")
    print(f"Throughput: {metrics['throughput_ips']:.2f} inferences/sec")
    print(f"Average memory usage: {metrics['avg_memory_percent']:.2f}%")
    print(f"Total inferences run: {metrics['total_inferences']}")
    
    # Show system optimization report
    report = metrics['system_optimization_report']
    print(f"\n--- System Optimization Report ---")
    print(f"CPU Count: {report['hardware_info']['cpu_count']}")
    print(f"Physical CPU Cores: {report['hardware_info']['cpu_count_physical']}")
    print(f"Total Memory: {report['hardware_info']['memory_total_gb']:.2f} GB")
    print(f"Current CPU Usage: {report['system_summary']['cpu_percent']:.2f}%")
    print(f"Current Memory Usage: {report['system_summary']['memory_percent']:.2f}%")
    
    if 'gpu_info' in report['hardware_info']:
        gpu_info = report['hardware_info']['gpu_info'][0] if report['hardware_info']['gpu_info'] else {}
        if gpu_info:
            print(f"GPU Name: {gpu_info.get('name', 'N/A')}")
            print(f"GPU Memory: {gpu_info.get('total_memory_mb', 0) / 1024:.2f} GB")
    
    # Clean up
    optimized_pipeline.cleanup()
    
    return {
        'baseline_time': baseline_time,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'time_improvement': time_improvement
    }


def demonstrate_profiling():
    """Demonstrate the profiling capabilities."""
    print("\n--- System Profiling Demonstration ---")
    
    config = SystemOptimizationConfig(
        enable_profiling=True,
        profile_interval=0.5,
        profile_memory=True,
        profile_compute=True
    )
    
    system_optimizer = SystemOptimizer(config)
    
    # Let profiling run for a few seconds
    print("Collecting system profile for 2 seconds...")
    time.sleep(2)
    
    # Get system summary
    summary = system_optimizer.profiler.get_system_summary()
    print(f"System Summary:")
    print(f"  CPU Usage: {summary['cpu_percent']:.2f}%")
    print(f"  Memory Usage: {summary['memory_percent']:.2f}%")
    print(f"  Available Memory: {summary['memory_available_mb']:.2f} MB")
    print(f"  Total Profiles: {summary['total_profiles']}")
    print(f"  Uptime: {summary['uptime']:.2f}s")
    
    # Get hardware info
    hw_info = system_optimizer.profiler.get_hardware_info()
    print(f"Hardware Info:")
    print(f"  CPU Cores: {hw_info['cpu_count']}")
    print(f"  Physical Cores: {hw_info['cpu_count_physical']}")
    print(f"  Total Memory: {hw_info['memory_total_gb']:.2f} GB")
    
    # Get memory efficiency report
    mem_report = system_optimizer.memory_manager.get_memory_efficiency_report()
    print(f"Memory Efficiency Report:")
    print(f"  Memory Utilization: {mem_report['memory_utilization']:.2f}%")
    print(f"  Available Memory: {mem_report['available_memory_mb']:.2f} MB")
    print(f"  Tensor Pool Size: {mem_report['tensor_pool_size']}")
    
    # Get resource status
    resource_status = system_optimizer.resource_scheduler.get_resource_status()
    print(f"Resource Status:")
    print(f"  Resource Usage: {resource_status['resource_usage']}")
    print(f"  Scheduling Algorithm: {resource_status['scheduling_algorithm']}")
    
    # Clean up
    system_optimizer.cleanup_resources()


def main():
    """Main demonstration function."""
    print("Qwen3-VL System-Level Optimizations Demonstration")
    print("=" * 60)
    
    # Show system info
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name()}")
    
    print(f"CPU Count: {psutil.cpu_count()}")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    # Run performance comparison
    results = run_performance_comparison()
    
    # Demonstrate profiling
    demonstrate_profiling()
    
    print(f"\n--- Summary ---")
    print(f"System-level optimizations achieved {results['time_improvement']:.2f}% time improvement")
    print(f"With {results['speedup']:.2f}x speedup over baseline")
    print("Optimizations implemented:")
    print("  - Profiling and monitoring")
    print("  - Multi-threading improvements")
    print("  - Resource scheduling techniques")
    print("  - Memory management optimizations")
    print("  - Thread management and CPU scheduling")
    
    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()