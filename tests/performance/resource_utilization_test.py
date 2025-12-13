"""
Resource Utilization Measurement Test for Qwen3-VL-2B-Instruct Architecture
This test measures CPU, memory, and other resource utilization to validate efficiency gains.
"""
import sys
import os
import torch
import time
import gc
import psutil
import threading
import numpy as np
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
from collections import deque
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager, MemoryConfig
from kv_cache_optimizer import KVCacheConfig, OptimizedKVCacheManager
from src.qwen3_vl.core.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


class ResourceMonitor:
    """Monitor system resources during model execution"""
    
    def __init__(self):
        self.cpu_percentages = []
        self.memory_percentages = []
        self.gpu_memory_usage = []
        self.timestamps = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_resources(self):
        """Monitor resources in a loop"""
        while self.monitoring:
            timestamp = time.time()
            cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
            memory_percent = psutil.virtual_memory().percent
            
            self.timestamps.append(timestamp)
            self.cpu_percentages.append(cpu_percent)
            self.memory_percentages.append(memory_percent)
            
            # Monitor GPU memory if available
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                self.gpu_memory_usage.append(gpu_memory)
            else:
                self.gpu_memory_usage.append(0)
                
            time.sleep(0.1)  # Monitor every 100ms
            
    def get_statistics(self) -> Dict[str, float]:
        """Get resource usage statistics"""
        if not self.cpu_percentages:
            return {}
            
        return {
            'avg_cpu': np.mean(self.cpu_percentages),
            'max_cpu': np.max(self.cpu_percentages),
            'min_cpu': np.min(self.cpu_percentages),
            'avg_memory': np.mean(self.memory_percentages),
            'max_memory': np.max(self.memory_percentages),
            'min_memory': np.min(self.memory_percentages),
            'avg_gpu_memory_mb': np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            'max_gpu_memory_mb': np.max(self.gpu_memory_usage) if self.gpu_memory_usage else 0,
            'duration': self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0
        }


def create_baseline_model():
    """Create a baseline model without optimizations for comparison"""
    config = Qwen3VLConfig()
    config.use_sparsity = False
    config.use_gradient_checkpointing = False
    config.hidden_size = 256  # Use smaller size for testing
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.vocab_size = 500  # Smaller vocab for testing
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    return model, config


def create_optimized_model():
    """Create an optimized model with all optimizations enabled"""
    config = Qwen3VLConfig()
    config.use_sparsity = True
    config.sparsity_ratio = 0.4  # Moderate sparsity
    config.exit_threshold = 0.75
    config.use_gradient_checkpointing = True
    config.hidden_size = 256  # Use smaller size for testing
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.vocab_size = 500  # Smaller vocab for testing
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Integrate memory manager
    memory_manager = MemoryManager(MemoryConfig(memory_pool_size=2**24))  # 16MB pool
    # Integrate KV cache optimizer
    kv_config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=32,
        use_sliding_window=True,
        sliding_window_size=256,
        use_hybrid=True
    )
    kv_cache_manager = OptimizedKVCacheManager(kv_config, memory_manager)
    
    # Note: In a real implementation, these would be integrated during model construction
    model.memory_manager = memory_manager
    model.kv_cache_manager = kv_cache_manager
    
    return model, config


def measure_resource_utilization_during_inference(model, input_ids, pixel_values, name: str):
    """Measure resource utilization during model inference"""
    print(f"Measuring resource utilization for {name}...")
    
    # Create resource monitor
    monitor = ResourceMonitor()
    
    # Warm up model
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_ids=input_ids, pixel_values=pixel_values)
    
    # Clear caches
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Run inference multiple times to get meaningful measurements
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            _ = model(input_ids=input_ids, pixel_values=pixel_values)
    end_time = time.time()
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Get statistics
    stats = monitor.get_statistics()
    stats['total_time'] = end_time - start_time
    stats['throughput'] = 10 / stats['total_time']  # samples per second
    
    print(f"  Total inference time: {stats['total_time']:.4f}s")
    print(f"  Throughput: {stats['throughput']:.2f} samples/sec")
    print(f"  Avg CPU usage: {stats['avg_cpu']:.2f}%")
    print(f"  Max CPU usage: {stats['max_cpu']:.2f}%")
    print(f"  Avg memory usage: {stats['avg_memory']:.2f}%")
    print(f"  Max memory usage: {stats['max_memory']:.2f}%")
    if torch.cuda.is_available():
        print(f"  Avg GPU memory: {stats['avg_gpu_memory_mb']:.2f} MB")
        print(f"  Max GPU memory: {stats['max_gpu_memory_mb']:.2f} MB")
    
    return stats


def compare_resource_utilization():
    """Compare resource utilization between baseline and optimized models"""
    print("Comparing resource utilization between models...")
    
    # Create both models
    baseline_model, baseline_config = create_baseline_model()
    optimized_model, opt_config = create_optimized_model()
    
    # Create test inputs
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, baseline_config.vision_image_size, baseline_config.vision_image_size)
    
    # Measure baseline resource utilization
    baseline_stats = measure_resource_utilization_during_inference(
        baseline_model, input_ids, pixel_values, "baseline model"
    )
    
    # Measure optimized resource utilization
    optimized_stats = measure_resource_utilization_during_inference(
        optimized_model, input_ids, pixel_values, "optimized model"
    )
    
    # Calculate improvements
    cpu_improvement = ((baseline_stats['avg_cpu'] - optimized_stats['avg_cpu']) / 
                      baseline_stats['avg_cpu'] * 100) if baseline_stats['avg_cpu'] > 0 else 0
    memory_improvement = ((baseline_stats['avg_memory'] - optimized_stats['avg_memory']) / 
                         baseline_stats['avg_memory'] * 100) if baseline_stats['avg_memory'] > 0 else 0
    throughput_improvement = ((optimized_stats['throughput'] - baseline_stats['throughput']) / 
                             baseline_stats['throughput'] * 100) if baseline_stats['throughput'] > 0 else 0
    
    print(f"\nResource Utilization Comparison:")
    print(f"  CPU Usage Improvement: {cpu_improvement:.2f}%")
    print(f"  Memory Usage Improvement: {memory_improvement:.2f}%")
    print(f"  Throughput Improvement: {throughput_improvement:.2f}%")
    
    return {
        'baseline': baseline_stats,
        'optimized': optimized_stats,
        'improvements': {
            'cpu': cpu_improvement,
            'memory': memory_improvement,
            'throughput': throughput_improvement
        }
    }


def measure_peak_resource_usage():
    """Measure peak resource usage during intensive operations"""
    print("Measuring peak resource usage during intensive operations...")
    
    # Create optimized model
    model, config = create_optimized_model()
    
    # Create larger inputs to stress the system
    batch_size, seq_len = 4, 64  # Larger than usual
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size)
    
    # Monitor system resources
    initial_cpu = psutil.cpu_percent(interval=1)
    initial_memory = psutil.virtual_memory().percent
    initial_gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    # Run intensive inference
    model.train()  # Enable gradients for more intensive computation
    for i in range(20):
        # Simulate training-like workload with gradients
        output = model(input_ids=input_ids, pixel_values=pixel_values)
        if hasattr(output, 'loss') and output.loss is not None:
            output.loss.backward()
            # Zero out gradients for next iteration
            model.zero_grad()
    
    # Measure peak usage after intensive operations
    peak_cpu = psutil.cpu_percent(interval=1)
    peak_memory = psutil.virtual_memory().percent
    peak_gpu_memory = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    print(f"  Initial CPU: {initial_cpu:.2f}% -> Peak CPU: {peak_cpu:.2f}%")
    print(f"  Initial Memory: {initial_memory:.2f}% -> Peak Memory: {peak_memory:.2f}%")
    if torch.cuda.is_available():
        print(f"  Initial GPU Memory: {initial_gpu_memory:.2f} MB -> Peak GPU Memory: {peak_gpu_memory:.2f} MB")
    
    return {
        'initial_cpu': initial_cpu,
        'peak_cpu': peak_cpu,
        'initial_memory': initial_memory,
        'peak_memory': peak_memory,
        'initial_gpu_memory': initial_gpu_memory,
        'peak_gpu_memory': peak_gpu_memory
    }


def analyze_memory_efficiency():
    """Analyze memory efficiency of the optimization techniques"""
    print("Analyzing memory efficiency of optimization techniques...")
    
    # Test memory manager efficiency
    memory_manager = MemoryManager(MemoryConfig(memory_pool_size=2**23))  # 8MB pool
    
    # Create tensors of various sizes to test caching efficiency
    test_shapes = [
        (100, 100),
        (200, 200),
        (50, 50),
        (300, 300),
        (75, 75)
    ]
    
    # Measure cache efficiency
    allocation_times = []
    for shape in test_shapes:
        start_time = time.time()
        for _ in range(10):  # Allocate and free same shape multiple times
            tensor = memory_manager.allocate_tensor(shape, torch.float32)
            memory_manager.free_tensor(tensor)
        end_time = time.time()
        allocation_times.append(end_time - start_time)
    
    # Get cache statistics
    stats = memory_manager.get_memory_stats()
    cache_stats = stats['pool_stats']['tensor_cache']
    
    print(f"  Cache hit rate: {cache_stats['hit_rate']:.4f}")
    print(f"  Cache size: {cache_stats['cache_size']}")
    print(f"  Total requests: {cache_stats['total_requests']}")
    print(f"  Cache hits: {cache_stats['cache_hits']}")
    print(f"  Average allocation time with cache: {np.mean(allocation_times):.6f}s")
    
    # Test KV cache memory efficiency
    kv_config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=32,
        use_sliding_window=True,
        sliding_window_size=256,
        use_hybrid=True
    )
    kv_cache_manager = OptimizedKVCacheManager(kv_config, memory_manager)
    
    # Create some key-value states to test compression
    batch_size, num_heads, seq_len, head_dim = 1, 4, 128, 64
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    
    # Update cache and get statistics
    kv_cache_manager.update(key_states, value_states)
    kv_stats = kv_cache_manager.get_memory_stats()
    
    print(f"  KV cache compression ratio: {kv_stats['compression_ratio']:.4f}")
    print(f"  Memory saved: {kv_stats['memory_saved_percentage']:.2f}%")
    
    return {
        'cache_stats': cache_stats,
        'allocation_times': allocation_times,
        'kv_cache_stats': kv_stats
    }


def plot_resource_utilization(results):
    """Plot resource utilization results"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # CPU usage comparison
        models = ['Baseline', 'Optimized']
        cpu_values = [
            results['baseline']['avg_cpu'],
            results['optimized']['avg_cpu']
        ]
        axes[0, 0].bar(models, cpu_values, color=['red', 'green'], alpha=0.7)
        axes[0, 0].set_title('Average CPU Usage Comparison')
        axes[0, 0].set_ylabel('CPU Usage (%)')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Memory usage comparison
        memory_values = [
            results['baseline']['avg_memory'],
            results['optimized']['avg_memory']
        ]
        axes[0, 1].bar(models, memory_values, color=['red', 'green'], alpha=0.7)
        axes[0, 1].set_title('Average Memory Usage Comparison')
        axes[0, 1].set_ylabel('Memory Usage (%)')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Throughput comparison
        throughput_values = [
            results['baseline']['throughput'],
            results['optimized']['throughput']
        ]
        axes[1, 0].bar(models, throughput_values, color=['red', 'green'], alpha=0.7)
        axes[1, 0].set_title('Throughput Comparison')
        axes[1, 0].set_ylabel('Samples/Second')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Improvement percentages
        improvements = list(results['improvements'].values())
        improvement_labels = list(results['improvements'].keys())
        axes[1, 1].bar(improvement_labels, improvements, color=['blue', 'orange', 'purple'], alpha=0.7)
        axes[1, 1].set_title('Improvement Percentages')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('resource_utilization_analysis.png', dpi=300, bbox_inches='tight')
        print("  Resource utilization plot saved as 'resource_utilization_analysis.png'")
    except ImportError:
        print("  Matplotlib not available, skipping plot generation")


def run_resource_utilization_measurement():
    """Run all resource utilization measurement tests"""
    print("=" * 80)
    print("RESOURCE UTILIZATION MEASUREMENT TEST FOR QWEN3-VL-2B-INSTRUCT ARCHITECTURE")
    print("=" * 80)
    
    print("Creating models for resource measurement...")
    
    # Compare resource utilization
    comparison_results = compare_resource_utilization()
    
    # Measure peak resource usage
    peak_usage = measure_peak_resource_usage()
    
    # Analyze memory efficiency
    memory_efficiency = analyze_memory_efficiency()
    
    print("\n" + "=" * 80)
    print("RESOURCE UTILIZATION MEASUREMENT SUMMARY")
    print("=" * 80)
    
    print(f"CPU Usage Improvement: {comparison_results['improvements']['cpu']:.2f}%")
    print(f"Memory Usage Improvement: {comparison_results['improvements']['memory']:.2f}%")
    print(f"Throughput Improvement: {comparison_results['improvements']['throughput']:.2f}%")
    
    # Check if improvements meet targets
    cpu_target_met = comparison_results['improvements']['cpu'] >= 10  # 10% improvement target
    memory_target_met = comparison_results['improvements']['memory'] >= 10  # 10% improvement target
    throughput_target_met = comparison_results['improvements']['throughput'] >= 20  # 20% improvement target
    
    print(f"\nTarget Achievement:")
    print(f"  CPU efficiency improvement (10%+): {'✓' if cpu_target_met else '✗'}")
    print(f"  Memory efficiency improvement (10%+): {'✓' if memory_target_met else '✗'}")
    print(f"  Throughput improvement (20%+): {'✓' if throughput_target_met else '✗'}")
    
    overall_success = cpu_target_met and memory_target_met and throughput_target_met
    
    print(f"\nOverall Resource Efficiency Target: {'✓ ACHIEVED' if overall_success else '✗ NOT ACHIEVED'}")
    
    # Save detailed results
    detailed_results = {
        'comparison': comparison_results,
        'peak_usage': peak_usage,
        'memory_efficiency': memory_efficiency,
        'targets_met': {
            'cpu': cpu_target_met,
            'memory': memory_target_met,
            'throughput': throughput_target_met
        },
        'overall_success': overall_success
    }
    
    with open('resource_utilization_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print("  Detailed results saved to 'resource_utilization_results.json'")
    
    # Plot results
    plot_resource_utilization(comparison_results)
    
    return overall_success


if __name__ == "__main__":
    success = run_resource_utilization_measurement()
    
    print(f"\n{'='*80}")
    print("RESOURCE UTILIZATION MEASUREMENT STATUS:", "PASSED" if success else "FAILED")
    print(f"{'='*80}")
    
    sys.exit(0 if success else 1)