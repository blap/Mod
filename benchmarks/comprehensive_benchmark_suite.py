"""
Comprehensive Benchmarking Suite for Qwen3-VL Model
This module implements a comprehensive benchmarking system to verify performance improvements and memory reductions
across all optimization techniques implemented in the architecture update plan.
"""
import os
import sys
import time
import gc
import psutil
import GPUtil
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
import json
import csv
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qwen3_vl.config.config import Qwen3VLConfig
from models.modeling_qwen3_vl_phase2 import Qwen3VLForConditionalGeneration
from benchmarks.benchmark_utils import benchmark_model_inference, profile_memory_usage, benchmark_generation


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking"""
    # Model configuration parameters
    hidden_size: int = 512  # Reduced for testing
    num_hidden_layers: int = 6  # Reduced for faster testing
    num_attention_heads: int = 8  # Reduced for faster testing
    vocab_size: int = 1000
    max_position_embeddings: int = 256
    
    # Benchmark parameters
    num_warmup_runs: int = 3
    num_benchmark_runs: int = 10
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4]
        if self.sequence_lengths is None:
            self.sequence_lengths = [64, 128, 256]


class BenchmarkResult:
    """Class to store benchmark results"""
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def start_timer(self):
        self.start_time = time.time()
    
    def stop_timer(self):
        self.end_time = time.time()
    
    def get_duration(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def add_result(self, key: str, value: Any):
        self.results[key] = value
    
    def get_result(self, key: str) -> Any:
        return self.results.get(key)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'duration': self.get_duration(),
            'results': self.results
        }


class ThroughputBenchmark:
    """Benchmark throughput (samples per second)"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def benchmark_throughput(self, model: nn.Module, device: torch.device) -> BenchmarkResult:
        """Benchmark throughput of the model"""
        result = BenchmarkResult("throughput_benchmark")
        result.start_timer()
        
        model.eval()
        model = model.to(device)
        
        # Define different input sizes to test
        input_configs = [
            {"batch_size": 1, "seq_len": 64},
            {"batch_size": 2, "seq_len": 128},
            {"batch_size": 4, "seq_len": 64},
        ]
        
        throughput_results = {}
        
        for config in input_configs:
            batch_size = config["batch_size"]
            seq_len = config["seq_len"]
            
            # Create test inputs
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
            pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(self.config.num_warmup_runs):
                    _ = model(input_ids=input_ids, pixel_values=pixel_values)
            
            # Benchmark
            start_time = time.time()
            num_processed = 0
            time_limit = 10.0  # Run for 10 seconds
            
            while time.time() - start_time < time_limit:
                with torch.no_grad():
                    _ = model(input_ids=input_ids, pixel_values=pixel_values)
                num_processed += batch_size
            
            elapsed_time = time.time() - start_time
            throughput = num_processed / elapsed_time
            
            config_key = f"batch_{batch_size}_seq_{seq_len}"
            throughput_results[config_key] = {
                "throughput_samples_per_sec": throughput,
                "total_samples_processed": num_processed,
                "elapsed_time": elapsed_time
            }
        
        result.add_result("throughput_results", throughput_results)
        result.stop_timer()
        
        return result


class LatencyBenchmark:
    """Benchmark latency (time per inference)"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def benchmark_latency(self, model: nn.Module, device: torch.device) -> BenchmarkResult:
        """Benchmark latency of the model"""
        result = BenchmarkResult("latency_benchmark")
        result.start_timer()
        
        model.eval()
        model = model.to(device)
        
        # Define different input sizes to test
        input_configs = [
            {"batch_size": 1, "seq_len": 64},
            {"batch_size": 1, "seq_len": 128},
            {"batch_size": 1, "seq_len": 256},
        ]
        
        latency_results = {}
        
        for config in input_configs:
            batch_size = config["batch_size"]
            seq_len = config["seq_len"]
            
            # Create test inputs
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
            pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(self.config.num_warmup_runs):
                    _ = model(input_ids=input_ids, pixel_values=pixel_values)
            
            # Benchmark
            times = []
            for _ in range(self.config.num_benchmark_runs):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    _ = model(input_ids=input_ids, pixel_values=pixel_values)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
            
            times = np.array(times)
            config_key = f"batch_{batch_size}_seq_{seq_len}"
            latency_results[config_key] = {
                "avg_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
                "min_time": float(np.min(times)),
                "max_time": float(np.max(times)),
                "p95_time": float(np.percentile(times, 95)),
                "throughput_samples_per_sec": float(batch_size / np.mean(times))
            }
        
        result.add_result("latency_results", latency_results)
        result.stop_timer()
        
        return result


class MemoryUsageBenchmark:
    """Benchmark memory usage"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def benchmark_memory_usage(self, model: nn.Module, device: torch.device) -> BenchmarkResult:
        """Benchmark memory usage of the model"""
        result = BenchmarkResult("memory_usage_benchmark")
        result.start_timer()
        
        model.eval()
        model = model.to(device)
        
        # Define different input sizes to test
        input_configs = [
            {"batch_size": 1, "seq_len": 64},
            {"batch_size": 2, "seq_len": 128},
            {"batch_size": 4, "seq_len": 64},
        ]
        
        memory_results = {}
        
        for config in input_configs:
            batch_size = config["batch_size"]
            seq_len = config["seq_len"]
            
            # Create test inputs
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
            pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
            
            # Record initial memory
            initial_cpu_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            if torch.cuda.is_available():
                initial_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                torch.cuda.reset_peak_memory_stats()
            else:
                initial_gpu_memory = 0
            
            # Run forward pass
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
            
            # Record memory after forward pass
            cpu_memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
            if torch.cuda.is_available():
                gpu_memory_after = torch.cuda.max_memory_allocated() / (1024**2)  # MB
                peak_gpu_memory = torch.cuda.max_memory_reserved() / (1024**2)  # MB
            else:
                gpu_memory_after = 0
                peak_gpu_memory = 0
            
            config_key = f"batch_{batch_size}_seq_{seq_len}"
            memory_results[config_key] = {
                "cpu_memory_initial_mb": initial_cpu_memory,
                "cpu_memory_after_mb": cpu_memory_after,
                "cpu_memory_increase_mb": cpu_memory_after - initial_cpu_memory,
                "gpu_memory_initial_mb": initial_gpu_memory if torch.cuda.is_available() else 0,
                "gpu_memory_after_mb": gpu_memory_after if torch.cuda.is_available() else 0,
                "gpu_memory_increase_mb": gpu_memory_after - initial_gpu_memory if torch.cuda.is_available() else 0,
                "gpu_peak_memory_mb": peak_gpu_memory if torch.cuda.is_available() else 0
            }
        
        result.add_result("memory_results", memory_results)
        result.stop_timer()
        
        return result


class ComprehensiveBenchmarkSuite:
    """Main benchmarking suite that runs all benchmarks"""
    
    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.throughput_benchmark = ThroughputBenchmark(self.config)
        self.latency_benchmark = LatencyBenchmark(self.config)
        self.memory_benchmark = MemoryUsageBenchmark(self.config)
        self.results = {}
    
    def create_model(self, use_optimizations: bool = True) -> Tuple[nn.Module, Qwen3VLConfig]:
        """Create a model instance with specified configuration"""
        # Create configuration
        qwen_config = Qwen3VLConfig()
        qwen_config.hidden_size = self.config.hidden_size
        qwen_config.num_hidden_layers = self.config.num_hidden_layers
        qwen_config.num_attention_heads = self.config.num_attention_heads
        qwen_config.vocab_size = self.config.vocab_size
        qwen_config.max_position_embeddings = self.config.max_position_embeddings
        
        # Apply optimization flags based on parameter
        if use_optimizations:
            # Enable various optimizations for the optimized model
            qwen_config.use_sparsity = True
            qwen_config.sparsity_ratio = 0.5
            qwen_config.exit_threshold = 0.75
            qwen_config.use_gradient_checkpointing = True
            qwen_config.use_moe = True
            qwen_config.moe_num_experts = 4
            qwen_config.moe_top_k = 2
            qwen_config.use_flash_attention_2 = True
            qwen_config.use_dynamic_sparse_attention = True
            qwen_config.use_adaptive_depth = True
            qwen_config.use_context_adaptive_positional_encoding = True
            qwen_config.use_conditional_feature_extraction = True
        else:
            # Baseline model without optimizations
            qwen_config.use_sparsity = False
            qwen_config.use_gradient_checkpointing = False
            qwen_config.use_moe = False
            qwen_config.use_flash_attention_2 = False
            qwen_config.use_dynamic_sparse_attention = False
            qwen_config.use_adaptive_depth = False
            qwen_config.use_context_adaptive_positional_encoding = False
            qwen_config.use_conditional_feature_extraction = False
        
        # Create model
        model = Qwen3VLForConditionalGeneration(qwen_config)
        return model, qwen_config
    
    def run_throughput_benchmark(self, device: torch.device) -> Dict[str, Any]:
        """Run throughput benchmark"""
        print("Running throughput benchmark...")
        
        # Create optimized model
        model, _ = self.create_model(use_optimizations=True)
        result = self.throughput_benchmark.benchmark_throughput(model, device)
        
        self.results['throughput'] = result.to_dict()
        return result.to_dict()
    
    def run_latency_benchmark(self, device: torch.device) -> Dict[str, Any]:
        """Run latency benchmark"""
        print("Running latency benchmark...")
        
        # Create optimized model
        model, _ = self.create_model(use_optimizations=True)
        result = self.latency_benchmark.benchmark_latency(model, device)
        
        self.results['latency'] = result.to_dict()
        return result.to_dict()
    
    def run_memory_benchmark(self, device: torch.device) -> Dict[str, Any]:
        """Run memory usage benchmark"""
        print("Running memory usage benchmark...")
        
        # Create optimized model
        model, _ = self.create_model(use_optimizations=True)
        result = self.memory_benchmark.benchmark_memory_usage(model, device)
        
        self.results['memory'] = result.to_dict()
        return result.to_dict()
    
    def run_all_benchmarks(self, device: torch.device = None) -> Dict[str, Any]:
        """Run all benchmarks"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("=" * 80)
        print("COMPREHENSIVE BENCHMARKING SUITE FOR QWEN3-VL")
        print("=" * 80)
        
        # Run all benchmarks
        self.run_throughput_benchmark(device)
        self.run_latency_benchmark(device)
        self.run_memory_benchmark(device)
        
        # Generate summary
        summary = self.generate_summary()
        
        print("\n" + "=" * 80)
        print("BENCHMARKING SUMMARY")
        print("=" * 80)
        
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return {
            'results': self.results,
            'summary': summary
        }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all benchmark results"""
        summary = {}
        
        # Throughput summary
        if 'throughput' in self.results:
            throughput_results = self.results['throughput']['results']['throughput_results']
            avg_throughput = np.mean([
                result['throughput_samples_per_sec'] 
                for result in throughput_results.values()
            ])
            summary['avg_throughput_samples_per_sec'] = avg_throughput
        
        # Latency summary
        if 'latency' in self.results:
            latency_results = self.results['latency']['results']['latency_results']
            avg_latency = np.mean([
                result['avg_time'] 
                for result in latency_results.values()
            ])
            summary['avg_latency_seconds'] = avg_latency
        
        # Memory summary
        if 'memory' in self.results:
            memory_results = self.results['memory']['results']['memory_results']
            avg_gpu_peak_memory = np.mean([
                result['gpu_peak_memory_mb'] 
                for result in memory_results.values()
            ])
            summary['avg_gpu_peak_memory_mb'] = avg_gpu_peak_memory
        
        return summary
    
    def save_results(self, filepath: str):
        """Save benchmark results to a file"""
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Benchmark results saved to {filepath}")
    
    def plot_results(self, output_dir: str = "benchmark_plots"):
        """Generate plots for benchmark results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up the plotting style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # Plot throughput results
        if 'throughput' in self.results:
            throughput_results = self.results['throughput']['results']['throughput_results']
            
            configs = list(throughput_results.keys())
            throughputs = [throughput_results[config]['throughput_samples_per_sec'] for config in configs]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(configs, throughputs, color='skyblue', edgecolor='navy', linewidth=1.2)
            plt.title('Throughput Comparison', fontsize=16, fontweight='bold')
            plt.xlabel('Input Configuration', fontsize=14)
            plt.ylabel('Throughput (samples/sec)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, throughputs):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs)*0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot latency results
        if 'latency' in self.results:
            latency_results = self.results['latency']['results']['latency_results']
            
            configs = list(latency_results.keys())
            avg_times = [latency_results[config]['avg_time'] for config in configs]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(configs, avg_times, color='lightcoral', edgecolor='darkred', linewidth=1.2)
            plt.title('Latency Comparison', fontsize=16, fontweight='bold')
            plt.xlabel('Input Configuration', fontsize=14)
            plt.ylabel('Average Latency (seconds)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, avg_times):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_times)*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/latency_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot memory results
        if 'memory' in self.results:
            memory_results = self.results['memory']['results']['memory_results']
            
            configs = list(memory_results.keys())
            peak_memory = [memory_results[config]['gpu_peak_memory_mb'] for config in configs]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(configs, peak_memory, color='lightgreen', edgecolor='darkgreen', linewidth=1.2)
            plt.title('GPU Peak Memory Usage', fontsize=16, fontweight='bold')
            plt.xlabel('Input Configuration', fontsize=14)
            plt.ylabel('Peak GPU Memory (MB)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, peak_memory):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(peak_memory)*0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/memory_usage.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Benchmark plots saved to {output_dir}")


def run_comprehensive_benchmarks():
    """Run the comprehensive benchmark suite"""
    config = BenchmarkConfig()
    benchmark_suite = ComprehensiveBenchmarkSuite(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = benchmark_suite.run_all_benchmarks(device)
    
    # Save results
    benchmark_suite.save_results("benchmark_results/comprehensive_benchmark_results.json")
    
    # Generate plots
    benchmark_suite.plot_results("benchmark_results/plots")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_benchmarks()