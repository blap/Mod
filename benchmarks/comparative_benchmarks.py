"""
Comparative Benchmarks between Optimized and Unoptimized Implementations
This module implements benchmarks to compare performance between optimized and unoptimized versions of the model.
"""
import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
import gc
import psutil
from dataclasses import dataclass
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qwen3_vl.config.config import Qwen3VLConfig
from models.modeling_qwen3_vl_phase2 import Qwen3VLForConditionalGeneration
from benchmarks.benchmark_utils import benchmark_model_inference, profile_memory_usage


@dataclass
class ComparativeBenchmarkConfig:
    """Configuration for comparative benchmarks"""
    # Model parameters
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    vocab_size: int = 1000
    max_position_embeddings: int = 256
    
    # Benchmark parameters
    num_warmup_runs: int = 3
    num_benchmark_runs: int = 10
    batch_sizes: list = None
    sequence_lengths: list = None
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4]
        if self.sequence_lengths is None:
            self.sequence_lengths = [64, 128, 256]


class ComparativeBenchmarker:
    """Class to run comparative benchmarks between optimized and unoptimized models"""
    
    def __init__(self, config: ComparativeBenchmarkConfig = None):
        self.config = config or ComparativeBenchmarkConfig()
    
    def create_models(self) -> Tuple[nn.Module, nn.Module, Qwen3VLConfig, Qwen3VLConfig]:
        """Create both optimized and unoptimized models"""
        # Create configuration for unoptimized model
        unoptimized_config = Qwen3VLConfig()
        unoptimized_config.hidden_size = self.config.hidden_size
        unoptimized_config.num_hidden_layers = self.config.num_hidden_layers
        unoptimized_config.num_attention_heads = self.config.num_attention_heads
        unoptimized_config.vocab_size = self.config.vocab_size
        unoptimized_config.max_position_embeddings = self.config.max_position_embeddings
        
        # Disable all optimizations for unoptimized model
        unoptimized_config.use_sparsity = False
        unoptimized_config.use_gradient_checkpointing = False
        unoptimized_config.use_moe = False
        unoptimized_config.use_flash_attention_2 = False
        unoptimized_config.use_dynamic_sparse_attention = False
        unoptimized_config.use_adaptive_depth = False
        unoptimized_config.use_context_adaptive_positional_encoding = False
        unoptimized_config.use_conditional_feature_extraction = False
        
        # Create unoptimized model
        unoptimized_model = Qwen3VLForConditionalGeneration(unoptimized_config)
        unoptimized_model.eval()
        
        # Create configuration for optimized model
        optimized_config = Qwen3VLConfig()
        optimized_config.hidden_size = self.config.hidden_size
        optimized_config.num_hidden_layers = self.config.num_hidden_layers
        optimized_config.num_attention_heads = self.config.num_attention_heads
        optimized_config.vocab_size = self.config.vocab_size
        optimized_config.max_position_embeddings = self.config.max_position_embeddings
        
        # Enable all optimizations for optimized model
        optimized_config.use_sparsity = True
        optimized_config.sparsity_ratio = 0.5
        optimized_config.exit_threshold = 0.75
        optimized_config.use_gradient_checkpointing = True
        optimized_config.use_moe = True
        optimized_config.moe_num_experts = 4
        optimized_config.moe_top_k = 2
        optimized_config.use_flash_attention_2 = True
        optimized_config.use_dynamic_sparse_attention = True
        optimized_config.use_adaptive_depth = True
        optimized_config.use_context_adaptive_positional_encoding = True
        optimized_config.use_conditional_feature_extraction = True
        
        # Create optimized model
        optimized_model = Qwen3VLForConditionalGeneration(optimized_config)
        optimized_model.eval()
        
        # Copy weights from unoptimized to optimized model to ensure same starting point
        optimized_model.load_state_dict(unoptimized_model.state_dict(), strict=False)
        
        return unoptimized_model, optimized_model, unoptimized_config, optimized_config
    
    def benchmark_performance_comparison(self, device: torch.device) -> Dict[str, Any]:
        """Benchmark performance comparison between optimized and unoptimized models"""
        print("Running performance comparison benchmark...")
        
        # Create models
        unoptimized_model, optimized_model, _, _ = self.create_models()
        unoptimized_model = unoptimized_model.to(device)
        optimized_model = optimized_model.to(device)
        
        performance_results = {}
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                print(f"  Testing batch_size={batch_size}, seq_len={seq_len}")
                
                # Create test inputs
                input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
                pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
                
                # Benchmark unoptimized model
                with torch.no_grad():
                    # Warmup
                    for _ in range(self.config.num_warmup_runs):
                        _ = unoptimized_model(input_ids=input_ids, pixel_values=pixel_values)
                
                unoptimized_times = []
                for _ in range(self.config.num_benchmark_runs):
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        _ = unoptimized_model(input_ids=input_ids, pixel_values=pixel_values)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.perf_counter()
                    
                    unoptimized_times.append(end_time - start_time)
                
                unoptimized_times = np.array(unoptimized_times)
                
                # Benchmark optimized model
                with torch.no_grad():
                    # Warmup
                    for _ in range(self.config.num_warmup_runs):
                        _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
                
                optimized_times = []
                for _ in range(self.config.num_benchmark_runs):
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.perf_counter()
                    
                    with torch.no_grad():
                        _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.perf_counter()
                    
                    optimized_times.append(end_time - start_time)
                
                optimized_times = np.array(optimized_times)
                
                # Calculate performance improvement
                unoptimized_avg_time = np.mean(unoptimized_times)
                optimized_avg_time = np.mean(optimized_times)
                
                if unoptimized_avg_time > 0:
                    speedup = unoptimized_avg_time / optimized_avg_time
                    performance_improvement = (unoptimized_avg_time - optimized_avg_time) / unoptimized_avg_time * 100
                else:
                    speedup = 1.0
                    performance_improvement = 0.0
                
                config_key = f"batch_{batch_size}_seq_{seq_len}"
                performance_results[config_key] = {
                    'unoptimized': {
                        'avg_time': float(unoptimized_avg_time),
                        'std_time': float(np.std(unoptimized_times)),
                        'min_time': float(np.min(unoptimized_times)),
                        'max_time': float(np.max(unoptimized_times)),
                        'throughput_samples_per_sec': float(batch_size / unoptimized_avg_time)
                    },
                    'optimized': {
                        'avg_time': float(optimized_avg_time),
                        'std_time': float(np.std(optimized_times)),
                        'min_time': float(np.min(optimized_times)),
                        'max_time': float(np.max(optimized_times)),
                        'throughput_samples_per_sec': float(batch_size / optimized_avg_time)
                    },
                    'improvement': {
                        'speedup_factor': float(speedup),
                        'performance_improvement_percent': float(performance_improvement),
                        'time_saved_per_inference_ms': float((unoptimized_avg_time - optimized_avg_time) * 1000)
                    }
                }
        
        return performance_results
    
    def benchmark_memory_comparison(self, device: torch.device) -> Dict[str, Any]:
        """Benchmark memory usage comparison between optimized and unoptimized models"""
        print("Running memory usage comparison benchmark...")
        
        # Create models
        unoptimized_model, optimized_model, _, _ = self.create_models()
        unoptimized_model = unoptimized_model.to(device)
        optimized_model = optimized_model.to(device)
        
        memory_results = {}
        
        for batch_size in [1, 2]:  # Use smaller batch sizes for memory testing
            for seq_len in [64, 128]:
                print(f"  Testing memory usage for batch_size={batch_size}, seq_len={seq_len}")
                
                # Create test inputs
                input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
                pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
                
                # Measure unoptimized model memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                initial_cpu_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
                if torch.cuda.is_available():
                    initial_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                    torch.cuda.reset_peak_memory_stats()
                else:
                    initial_gpu_memory = 0
                
                # Run unoptimized model
                with torch.no_grad():
                    _ = unoptimized_model(input_ids=input_ids, pixel_values=pixel_values)
                
                unoptimized_cpu_memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
                if torch.cuda.is_available():
                    unoptimized_gpu_memory_after = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                    unoptimized_gpu_peak_memory = torch.cuda.max_memory_reserved() / (1024**2)  # MB
                else:
                    unoptimized_gpu_memory_after = 0
                    unoptimized_gpu_peak_memory = 0
                
                # Measure optimized model memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                initial_opt_cpu_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
                if torch.cuda.is_available():
                    initial_opt_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                    torch.cuda.reset_peak_memory_stats()
                else:
                    initial_opt_gpu_memory = 0
                
                # Run optimized model
                with torch.no_grad():
                    _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
                
                optimized_cpu_memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
                if torch.cuda.is_available():
                    optimized_gpu_memory_after = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                    optimized_gpu_peak_memory = torch.cuda.max_memory_reserved() / (1024**2)  # MB
                else:
                    optimized_gpu_memory_after = 0
                    optimized_gpu_peak_memory = 0
                
                config_key = f"batch_{batch_size}_seq_{seq_len}"
                memory_results[config_key] = {
                    'unoptimized': {
                        'cpu_memory_increase_mb': unoptimized_cpu_memory_after - initial_cpu_memory,
                        'gpu_memory_increase_mb': unoptimized_gpu_memory_after - initial_gpu_memory if torch.cuda.is_available() else 0,
                        'gpu_peak_memory_mb': unoptimized_gpu_peak_memory
                    },
                    'optimized': {
                        'cpu_memory_increase_mb': optimized_cpu_memory_after - initial_opt_cpu_memory,
                        'gpu_memory_increase_mb': optimized_gpu_memory_after - initial_opt_gpu_memory if torch.cuda.is_available() else 0,
                        'gpu_peak_memory_mb': optimized_gpu_peak_memory
                    },
                    'improvement': {
                        'cpu_memory_reduction_mb': (unoptimized_cpu_memory_after - initial_cpu_memory) - (optimized_cpu_memory_after - initial_opt_cpu_memory),
                        'gpu_memory_reduction_mb': (unoptimized_gpu_peak_memory - initial_gpu_memory) - (optimized_gpu_peak_memory - initial_opt_gpu_memory) if torch.cuda.is_available() else 0,
                        'cpu_memory_reduction_percent': (
                            ((unoptimized_cpu_memory_after - initial_cpu_memory) - (optimized_cpu_memory_after - initial_opt_cpu_memory)) / 
                            (unoptimized_cpu_memory_after - initial_cpu_memory) * 100
                            if (unoptimized_cpu_memory_after - initial_cpu_memory) != 0 else 0
                        ),
                        'gpu_memory_reduction_percent': (
                            ((unoptimized_gpu_peak_memory - initial_gpu_memory) - (optimized_gpu_peak_memory - initial_opt_gpu_memory)) / 
                            (unoptimized_gpu_peak_memory - initial_gpu_memory) * 100
                            if (unoptimized_gpu_peak_memory - initial_gpu_memory) != 0 and torch.cuda.is_available() else 0
                        )
                    }
                }
        
        return memory_results
    
    def benchmark_throughput_comparison(self, device: torch.device) -> Dict[str, Any]:
        """Benchmark throughput comparison between optimized and unoptimized models"""
        print("Running throughput comparison benchmark...")
        
        # Create models
        unoptimized_model, optimized_model, _, _ = self.create_models()
        unoptimized_model = unoptimized_model.to(device)
        optimized_model = optimized_model.to(device)
        
        throughput_results = {}
        
        for batch_size in [1, 2, 4]:
            print(f"  Testing throughput for batch_size={batch_size}")
            
            # Create test inputs
            seq_len = 128
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
            pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Measure unoptimized model throughput
            start_time = time.time()
            unoptimized_processed = 0
            time_limit = 10.0  # Run for 10 seconds
            
            while time.time() - start_time < time_limit:
                with torch.no_grad():
                    _ = unoptimized_model(input_ids=input_ids, pixel_values=pixel_values)
                unoptimized_processed += batch_size
            
            unoptimized_time = time.time() - start_time
            unoptimized_throughput = unoptimized_processed / unoptimized_time
            
            # Measure optimized model throughput
            start_time = time.time()
            optimized_processed = 0
            
            while time.time() - start_time < time_limit:
                with torch.no_grad():
                    _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
                optimized_processed += batch_size
            
            optimized_time = time.time() - start_time
            optimized_throughput = optimized_processed / optimized_time
            
            # Calculate throughput improvement
            if unoptimized_throughput > 0:
                throughput_improvement = (optimized_throughput - unoptimized_throughput) / unoptimized_throughput * 100
                throughput_speedup = optimized_throughput / unoptimized_throughput
            else:
                throughput_improvement = 0
                throughput_speedup = 1.0
            
            config_key = f"batch_{batch_size}"
            throughput_results[config_key] = {
                'unoptimized': {
                    'throughput_samples_per_sec': unoptimized_throughput,
                    'total_samples_processed': unoptimized_processed,
                    'elapsed_time': unoptimized_time
                },
                'optimized': {
                    'throughput_samples_per_sec': optimized_throughput,
                    'total_samples_processed': optimized_processed,
                    'elapsed_time': optimized_time
                },
                'improvement': {
                    'throughput_improvement_percent': throughput_improvement,
                    'throughput_speedup_factor': throughput_speedup
                }
            }
        
        return throughput_results
    
    def run_all_comparative_benchmarks(self, device: torch.device = None) -> Dict[str, Any]:
        """Run all comparative benchmarks"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("=" * 80)
        print("COMPARATIVE BENCHMARKS: OPTIMIZED vs UNOPTIMIZED IMPLEMENTATIONS")
        print("=" * 80)
        
        print(f"Using device: {device}")
        
        results = {}
        
        # Run performance comparison
        results['performance_comparison'] = self.benchmark_performance_comparison(device)
        
        # Run memory comparison
        results['memory_comparison'] = self.benchmark_memory_comparison(device)
        
        # Run throughput comparison
        results['throughput_comparison'] = self.benchmark_throughput_comparison(device)
        
        # Generate summary
        summary = self.generate_summary(results)
        
        print("\n" + "=" * 80)
        print("COMPARATIVE BENCHMARK SUMMARY")
        print("=" * 80)
        
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return {
            'results': results,
            'summary': summary
        }
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of comparative benchmark results"""
        summary = {}
        
        # Performance comparison summary
        if 'performance_comparison' in results:
            perf_results = results['performance_comparison']
            
            # Calculate average speedup across all configurations
            speedups = [result['improvement']['speedup_factor'] for result in perf_results.values()]
            avg_speedup = np.mean(speedups) if speedups else 0
            
            # Calculate average performance improvement
            improvements = [result['improvement']['performance_improvement_percent'] for result in perf_results.values()]
            avg_improvement = np.mean(improvements) if improvements else 0
            
            summary['avg_speedup_factor'] = avg_speedup
            summary['avg_performance_improvement_percent'] = avg_improvement
            summary['max_speedup_factor'] = max(speedups) if speedups else 0
            summary['min_speedup_factor'] = min(speedups) if speedups else 0
        
        # Memory comparison summary
        if 'memory_comparison' in results:
            mem_results = results['memory_comparison']
            
            # Calculate average memory reduction
            cpu_reductions = [result['improvement']['cpu_memory_reduction_percent'] for result in mem_results.values()]
            gpu_reductions = [result['improvement']['gpu_memory_reduction_percent'] for result in mem_results.values()]
            
            avg_cpu_reduction = np.mean(cpu_reductions) if cpu_reductions else 0
            avg_gpu_reduction = np.mean(gpu_reductions) if gpu_reductions else 0
            
            summary['avg_cpu_memory_reduction_percent'] = avg_cpu_reduction
            summary['avg_gpu_memory_reduction_percent'] = avg_gpu_reduction
        
        # Throughput comparison summary
        if 'throughput_comparison' in results:
            thr_results = results['throughput_comparison']
            
            # Calculate average throughput improvement
            thr_improvements = [result['improvement']['throughput_improvement_percent'] for result in thr_results.values()]
            avg_thr_improvement = np.mean(thr_improvements) if thr_improvements else 0
            
            # Calculate average throughput speedup
            thr_speedups = [result['improvement']['throughput_speedup_factor'] for result in thr_results.values()]
            avg_thr_speedup = np.mean(thr_speedups) if thr_speedups else 0
            
            summary['avg_throughput_improvement_percent'] = avg_thr_improvement
            summary['avg_throughput_speedup_factor'] = avg_thr_speedup
        
        # Overall assessment
        performance_good = summary.get('avg_performance_improvement_percent', 0) > 10
        memory_good = summary.get('avg_gpu_memory_reduction_percent', 0) > 5
        throughput_good = summary.get('avg_throughput_improvement_percent', 0) > 10
        
        summary['optimization_effectiveness'] = {
            'performance_improvement_significant': performance_good,
            'memory_efficiency_improvement_significant': memory_good,
            'throughput_improvement_significant': throughput_good,
            'overall_optimization_successful': performance_good and memory_good and throughput_good
        }
        
        return summary


def run_comparative_benchmarks():
    """Run comparative benchmarks between optimized and unoptimized implementations"""
    config = ComparativeBenchmarkConfig()
    benchmarker = ComparativeBenchmarker(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = benchmarker.run_all_comparative_benchmarks(device)
    
    # Save results
    Path("benchmark_results").mkdir(parents=True, exist_ok=True)
    with open("benchmark_results/comparative_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_comparative_benchmarks()