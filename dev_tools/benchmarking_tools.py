"""
Benchmarking Tools for Qwen3-VL Model

This module provides comprehensive benchmarking tools to compare performance 
before and after optimizations in the Qwen3-VL model.
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import psutil
import GPUtil
from contextlib import contextmanager
import copy


@dataclass
class BenchmarkResult:
    """Data class for storing benchmark results"""
    name: str
    execution_time: float
    memory_usage: float  # in MB
    gpu_memory_usage: Optional[float] = None  # in MB
    throughput: Optional[float] = None  # operations per second
    accuracy: Optional[float] = None
    config: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


class BenchmarkSuite:
    """Main benchmarking suite"""
    
    def __init__(self):
        self.results = []
        self.benchmarks = {}
        self.configurations = {}
    
    def add_benchmark(self, name: str, benchmark_func: Callable, description: str = ""):
        """Add a benchmark function to the suite"""
        self.benchmarks[name] = {
            'function': benchmark_func,
            'description': description
        }
    
    def add_configuration(self, name: str, config: Dict[str, Any]):
        """Add a configuration for benchmarking"""
        self.configurations[name] = config
    
    def run_benchmark(self, name: str, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark"""
        if name not in self.benchmarks:
            raise ValueError(f"Benchmark '{name}' not found")
        
        benchmark_func = self.benchmarks[name]['function']
        
        # Measure initial resource usage
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        initial_gpu_memory = self._get_gpu_memory() if torch.cuda.is_available() else None
        
        start_time = time.time()
        
        # Run the benchmark
        result = benchmark_func(*args, **kwargs)
        
        end_time = time.time()
        
        # Measure final resource usage
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        final_gpu_memory = self._get_gpu_memory() if torch.cuda.is_available() else None
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = final_memory - initial_memory
        gpu_memory_usage = (final_gpu_memory - initial_gpu_memory) if initial_gpu_memory is not None else None
        
        benchmark_result = BenchmarkResult(
            name=name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            timestamp=time.time()
        )
        
        self.results.append(benchmark_result)
        return benchmark_result
    
    def run_all_benchmarks(self, config_name: str = None) -> List[BenchmarkResult]:
        """Run all registered benchmarks"""
        results = []
        
        for name in self.benchmarks:
            try:
                result = self.run_benchmark(name)
                results.append(result)
            except Exception as e:
                print(f"Error running benchmark '{name}': {e}")
        
        return results
    
    def compare_benchmarks(self, baseline_config: str, test_config: str) -> Dict[str, Any]:
        """Compare benchmark results between two configurations"""
        # This would typically involve running benchmarks with different configurations
        # and comparing the results
        pass
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def save_results(self, path: str):
        """Save benchmark results to file"""
        results_data = []
        for result in self.results:
            results_data.append({
                'name': result.name,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'gpu_memory_usage': result.gpu_memory_usage,
                'throughput': result.throughput,
                'accuracy': result.accuracy,
                'config': result.config,
                'timestamp': result.timestamp
            })
        
        with open(path, 'w') as f:
            json.dump({
                'results': results_data,
                'configurations': self.configurations
            }, f, indent=2)
    
    def load_results(self, path: str):
        """Load benchmark results from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.results = []
        for result_data in data['results']:
            self.results.append(BenchmarkResult(**result_data))
        
        self.configurations = data.get('configurations', {})
    
    def visualize_results(self, figsize: tuple = (12, 8)):
        """Visualize benchmark results"""
        if not self.results:
            print("No benchmark results to visualize")
            return
        
        # Create a DataFrame for easier plotting
        df_data = []
        for result in self.results:
            df_data.append({
                'name': result.name,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'gpu_memory_usage': result.gpu_memory_usage or 0
            })
        
        df = pd.DataFrame(df_data)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Execution time
        axes[0, 0].bar(df['name'], df['execution_time'])
        axes[0, 0].set_title('Execution Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Memory usage
        axes[0, 1].bar(df['name'], df['memory_usage'])
        axes[0, 1].set_title('Memory Usage (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # GPU memory usage (if available)
        if df['gpu_memory_usage'].sum() > 0:
            axes[1, 0].bar(df['name'], df['gpu_memory_usage'])
            axes[1, 0].set_title('GPU Memory Usage (MB)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Throughput vs Memory (if available)
        if all(r.throughput is not None for r in self.results):
            throughput = [r.throughput or 0 for r in self.results]
            memory = [r.memory_usage for r in self.results]
            axes[1, 1].scatter(memory, throughput)
            axes[1, 1].set_xlabel('Memory Usage (MB)')
            axes[1, 1].set_ylabel('Throughput')
            axes[1, 1].set_title('Throughput vs Memory Usage')
        
        plt.tight_layout()
        plt.show()


class ModelBenchmarkSuite:
    """Benchmark suite specifically for Qwen3-VL model"""
    
    def __init__(self):
        self.benchmarks = {}
        self.results = []
    
    def benchmark_inference(self, model: nn.Module, input_data: torch.Tensor, 
                           num_iterations: int = 100, warmup: int = 10) -> BenchmarkResult:
        """Benchmark model inference performance"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(copy.deepcopy(input_data))
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(copy.deepcopy(input_data))
        
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        avg_time_per_iteration = execution_time / num_iterations
        memory_usage = memory_after - memory_before
        throughput = num_iterations / execution_time
        
        gpu_memory_usage = None
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return BenchmarkResult(
            name="inference",
            execution_time=avg_time_per_iteration,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            throughput=throughput
        )
    
    def benchmark_training_step(self, model: nn.Module, input_data: torch.Tensor, 
                               target_data: torch.Tensor, num_iterations: int = 50, 
                               warmup: int = 5) -> BenchmarkResult:
        """Benchmark model training step performance"""
        model.train()
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Warmup
        for _ in range(warmup):
            optimizer.zero_grad()
            output = model(input_data)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target_data)
            loss.backward()
            optimizer.step()
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            output = model(input_data)
            if isinstance(output, tuple):
                output = output[0]
            loss = criterion(output, target_data)
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        total_execution_time = end_time - start_time
        avg_time_per_iteration = total_execution_time / num_iterations if num_iterations > 0 else 0
        memory_usage = memory_after - memory_before
        throughput = num_iterations / total_execution_time if total_execution_time > 0 else 0

        gpu_memory_usage = None
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024

        return BenchmarkResult(
            name="training_step",
            execution_time=avg_time_per_iteration,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            throughput=throughput
        )
    
    def benchmark_memory_efficiency(self, model: nn.Module, sequence_lengths: List[int], 
                                   batch_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark memory efficiency with different sequence lengths and batch sizes"""
        results = []
        
        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                try:
                    # Create input tensor
                    input_tensor = torch.randn(batch_size, seq_len, model.config.hidden_size if hasattr(model, 'config') else 768)
                    
                    # Run inference and measure memory
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        input_tensor = input_tensor.cuda()
                        model = model.cuda()
                    
                    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    
                    with torch.no_grad():
                        _ = model(input_tensor)
                    
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    memory_usage = end_memory - start_memory
                    
                    gpu_memory_usage = None
                    if torch.cuda.is_available():
                        gpu_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024
                    
                    result = BenchmarkResult(
                        name=f"memory_efficiency_B{batch_size}_L{seq_len}",
                        execution_time=0,  # Not measuring time here
                        memory_usage=memory_usage,
                        gpu_memory_usage=gpu_memory_usage
                    )
                    results.append(result)
                    
                except RuntimeError as e:
                    # Handle out-of-memory errors
                    if "out of memory" in str(e).lower():
                        result = BenchmarkResult(
                            name=f"memory_efficiency_B{batch_size}_L{seq_len}",
                            execution_time=0,
                            memory_usage=-1,  # Indicate OOM
                            gpu_memory_usage=-1
                        )
                        results.append(result)
        
        return results
    
    def benchmark_throughput(self, model: nn.Module, input_data: torch.Tensor, 
                            max_batch_size: int = 32) -> List[BenchmarkResult]:
        """Benchmark throughput with different batch sizes"""
        results = []
        
        model.eval()
        
        for batch_size in [1, 2, 4, 8, 16, max_batch_size]:
            try:
                # Adjust input data for batch size
                if input_data.shape[0] < batch_size:
                    # Repeat the input to match batch size
                    repeat_times = (batch_size + input_data.shape[0] - 1) // input_data.shape[0]
                    test_input = input_data.repeat(repeat_times, 1, 1)[:batch_size]
                else:
                    test_input = input_data[:batch_size]
                
                # Warmup
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(copy.deepcopy(test_input))
                
                # Benchmark
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                start_time = time.time()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                with torch.no_grad():
                    for _ in range(20):  # Run multiple iterations for better measurement
                        _ = model(copy.deepcopy(test_input))
                
                end_time = time.time()
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                total_time = end_time - start_time
                iterations = 20
                avg_time = total_time / iterations if total_time > 0 else float('inf')
                throughput = iterations / total_time if total_time > 0 else 0
                memory_usage = memory_after - memory_before
                
                gpu_memory_usage = None
                if torch.cuda.is_available():
                    gpu_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024
                
                result = BenchmarkResult(
                    name=f"throughput_B{batch_size}",
                    execution_time=avg_time,
                    memory_usage=memory_usage,
                    gpu_memory_usage=gpu_memory_usage,
                    throughput=throughput
                )
                results.append(result)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    result = BenchmarkResult(
                        name=f"throughput_B{batch_size}",
                        execution_time=-1,  # Indicate OOM
                        memory_usage=-1,
                        gpu_memory_usage=-1,
                        throughput=0
                    )
                    results.append(result)
        
        return results


class OptimizationBenchmarkSuite:
    """Benchmark suite for comparing optimized vs unoptimized models"""
    
    def __init__(self):
        self.results = []
    
    def benchmark_optimization_impact(self, original_model: nn.Module, 
                                     optimized_model: nn.Module,
                                     input_data: torch.Tensor,
                                     target_data: Optional[torch.Tensor] = None,
                                     num_iterations: int = 100) -> Dict[str, Any]:
        """Benchmark the impact of optimizations"""
        
        print("Benchmarking original model...")
        original_inference = self._benchmark_model_inference(original_model, input_data, num_iterations)
        
        print("Benchmarking optimized model...")
        optimized_inference = self._benchmark_model_inference(optimized_model, input_data, num_iterations)
        
        # Compare results
        speedup = original_inference.execution_time / optimized_inference.execution_time if optimized_inference.execution_time > 0 else float('inf')
        memory_improvement = (original_inference.memory_usage - optimized_inference.memory_usage) / original_inference.memory_usage * 100 if original_inference.memory_usage > 0 else 0
        
        results = {
            'original': {
                'execution_time': original_inference.execution_time,
                'memory_usage': original_inference.memory_usage,
                'gpu_memory_usage': original_inference.gpu_memory_usage,
                'throughput': original_inference.throughput
            },
            'optimized': {
                'execution_time': optimized_inference.execution_time,
                'memory_usage': optimized_inference.memory_usage,
                'gpu_memory_usage': optimized_inference.gpu_memory_usage,
                'throughput': optimized_inference.throughput
            },
            'improvements': {
                'speedup_factor': speedup,
                'memory_improvement_percent': memory_improvement,
                'throughput_improvement_factor': (optimized_inference.throughput / original_inference.throughput) if original_inference.throughput > 0 else float('inf')
            }
        }
        
        self.results.append(results)
        return results
    
    def _benchmark_model_inference(self, model: nn.Module, input_data: torch.Tensor, 
                                  num_iterations: int) -> BenchmarkResult:
        """Internal method to benchmark model inference"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(copy.deepcopy(input_data))
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(copy.deepcopy(input_data))
        
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        total_time = end_time - start_time
        execution_time = total_time / num_iterations if num_iterations > 0 else 0
        memory_usage = memory_after - memory_before
        throughput = num_iterations / total_time if total_time > 0 else 0

        gpu_memory_usage = None
        if torch.cuda.is_available():
            gpu_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024

        return BenchmarkResult(
            name="inference",
            execution_time=execution_time,
            memory_usage=memory_usage,
            gpu_memory_usage=gpu_memory_usage,
            throughput=throughput
        )
    
    def visualize_optimization_comparison(self, results: Dict[str, Any], figsize: tuple = (12, 8)):
        """Visualize optimization comparison results"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Execution time comparison
        labels = ['Original', 'Optimized']
        times = [results['original']['execution_time'], results['optimized']['execution_time']]
        axes[0, 0].bar(labels, times)
        axes[0, 0].set_title('Execution Time Comparison')
        axes[0, 0].set_ylabel('Time per iteration (s)')
        
        # Memory usage comparison
        memory_usage = [results['original']['memory_usage'], results['optimized']['memory_usage']]
        axes[0, 1].bar(labels, memory_usage)
        axes[0, 1].set_title('Memory Usage Comparison')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        
        # Throughput comparison
        throughput = [results['original']['throughput'], results['optimized']['throughput']]
        axes[1, 0].bar(labels, throughput)
        axes[1, 0].set_title('Throughput Comparison')
        axes[1, 0].set_ylabel('Iterations per second')
        
        # Improvement metrics
        improvements = [
            results['improvements']['speedup_factor'],
            results['improvements']['memory_improvement_percent'],
            results['improvements']['throughput_improvement_factor']
        ]
        improvement_labels = ['Speedup', 'Memory %', 'Throughput']
        axes[1, 1].bar(improvement_labels, improvements)
        axes[1, 1].set_title('Improvement Metrics')
        axes[1, 1].set_ylabel('Factor/%')
        
        plt.tight_layout()
        plt.show()


class HardwareBenchmarkSuite:
    """Benchmark suite for hardware-specific optimizations"""
    
    def __init__(self):
        self.results = []
    
    def benchmark_cpu_vs_gpu(self, model: nn.Module, input_data: torch.Tensor, 
                            num_iterations: int = 50) -> Dict[str, Any]:
        """Compare CPU vs GPU performance"""
        results = {}
        
        # CPU benchmark
        print("Benchmarking on CPU...")
        cpu_model = model.cpu()
        cpu_input = input_data.cpu()
        
        start_time = time.time()
        cpu_memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        cpu_model.eval()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = cpu_model(copy.deepcopy(cpu_input))
        
        cpu_time = time.time() - start_time
        cpu_memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_memory_usage = cpu_memory_after - cpu_memory_before
        
        results['cpu'] = {
            'execution_time': cpu_time / num_iterations,
            'memory_usage': cpu_memory_usage,
            'throughput': num_iterations / cpu_time
        }
        
        # GPU benchmark (if available)
        if torch.cuda.is_available():
            print("Benchmarking on GPU...")
            gpu_model = model.cuda()
            gpu_input = input_data.cuda()
            
            torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.no_grad():
                    _ = gpu_model(copy.deepcopy(gpu_input))
            
            gpu_time = time.time() - start_time
            gpu_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            results['gpu'] = {
                'execution_time': gpu_time / num_iterations,
                'memory_usage': gpu_memory_usage,
                'throughput': num_iterations / gpu_time
            }
        else:
            results['gpu'] = None
        
        # Calculate speedup
        if results['gpu']:
            results['speedup'] = results['cpu']['execution_time'] / results['gpu']['execution_time']
        
        self.results.append(results)
        return results
    
    def benchmark_mixed_precision(self, model: nn.Module, input_data: torch.Tensor, 
                                 num_iterations: int = 50) -> Dict[str, Any]:
        """Benchmark mixed precision vs full precision"""
        results = {}
        
        # Full precision (FP32) benchmark
        print("Benchmarking FP32...")
        model_fp32 = copy.deepcopy(model).float()  # Ensure FP32
        input_fp32 = input_data.float()
        
        start_time = time.time()
        fp32_memory_before = psutil.Process().memory_info().rss / 1024 / 1024
        
        model_fp32.eval()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model_fp32(copy.deepcopy(input_fp32))
        
        fp32_time = time.time() - start_time
        fp32_memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        fp32_memory_usage = fp32_memory_after - fp32_memory_before
        
        results['fp32'] = {
            'execution_time': fp32_time / num_iterations,
            'memory_usage': fp32_memory_usage,
            'throughput': num_iterations / fp32_time
        }
        
        # Mixed precision (if CUDA available)
        if torch.cuda.is_available():
            print("Benchmarking Mixed Precision (FP16)...")
            model_fp16 = copy.deepcopy(model).half()  # Convert to FP16
            input_fp16 = input_data.half()
            
            torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            for _ in range(num_iterations):
                with torch.cuda.amp.autocast():  # Use automatic mixed precision
                    with torch.no_grad():
                        _ = model_fp16(copy.deepcopy(input_fp16))
            
            fp16_time = time.time() - start_time
            fp16_memory_usage = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            results['fp16'] = {
                'execution_time': fp16_time / num_iterations,
                'memory_usage': fp16_memory_usage,
                'throughput': num_iterations / fp16_time
            }
        else:
            results['fp16'] = None
        
        # Calculate improvements
        if results['fp16']:
            results['time_improvement'] = results['fp32']['execution_time'] / results['fp16']['execution_time']
            results['memory_improvement'] = results['fp32']['memory_usage'] / results['fp16']['memory_usage']
        
        self.results.append(results)
        return results


class BenchmarkReporter:
    """Generate reports from benchmark results"""
    
    def __init__(self):
        pass
    
    def generate_performance_report(self, results: List[BenchmarkResult], 
                                  title: str = "Performance Benchmark Report") -> str:
        """Generate a text report from benchmark results"""
        report_lines = [f"=== {title} ===", ""]
        
        if not results:
            report_lines.append("No benchmark results available.")
            return "\n".join(report_lines)
        
        # Summary statistics
        total_time = sum(r.execution_time for r in results if r.execution_time > 0)
        avg_time = total_time / len([r for r in results if r.execution_time > 0])
        total_memory = sum(r.memory_usage for r in results if r.memory_usage > 0)
        avg_memory = total_memory / len([r for r in results if r.memory_usage > 0])
        
        report_lines.append(f"Total Execution Time: {total_time:.4f}s")
        report_lines.append(f"Average Execution Time: {avg_time:.4f}s")
        report_lines.append(f"Total Memory Usage: {total_memory:.2f}MB")
        report_lines.append(f"Average Memory Usage: {avg_memory:.2f}MB")
        report_lines.append("")
        
        # Detailed results
        report_lines.append("Detailed Results:")
        for result in results:
            report_lines.append(f"  {result.name}:")
            report_lines.append(f"    Execution Time: {result.execution_time:.4f}s")
            report_lines.append(f"    Memory Usage: {result.memory_usage:.2f}MB")
            if result.gpu_memory_usage is not None:
                report_lines.append(f"    GPU Memory: {result.gpu_memory_usage:.2f}MB")
            if result.throughput is not None:
                report_lines.append(f"    Throughput: {result.throughput:.2f} ops/s")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_benchmark_results(self, results: List[BenchmarkResult], path: str):
        """Save benchmark results to JSON file"""
        results_data = []
        for result in results:
            results_data.append({
                'name': result.name,
                'execution_time': result.execution_time,
                'memory_usage': result.memory_usage,
                'gpu_memory_usage': result.gpu_memory_usage,
                'throughput': result.throughput,
                'accuracy': result.accuracy,
                'config': result.config,
                'timestamp': result.timestamp
            })
        
        with open(path, 'w') as f:
            json.dump(results_data, f, indent=2)


# Global benchmark instances
benchmark_suite = BenchmarkSuite()
model_benchmark_suite = ModelBenchmarkSuite()
optimization_benchmark_suite = OptimizationBenchmarkSuite()
hardware_benchmark_suite = HardwareBenchmarkSuite()
benchmark_reporter = BenchmarkReporter()


def run_model_benchmarks(model: nn.Module, input_data: torch.Tensor, 
                        target_data: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Run comprehensive model benchmarks"""
    print("Running comprehensive model benchmarks...")
    
    results = {}
    
    # Inference benchmark
    print("Running inference benchmark...")
    inference_result = model_benchmark_suite.benchmark_inference(model, input_data)
    results['inference'] = inference_result
    
    # Training benchmark (if target data provided)
    if target_data is not None:
        print("Running training benchmark...")
        training_result = model_benchmark_suite.benchmark_training_step(model, input_data, target_data)
        results['training'] = training_result
    
    # Throughput benchmark
    print("Running throughput benchmark...")
    throughput_results = model_benchmark_suite.benchmark_throughput(model, input_data)
    results['throughput'] = throughput_results
    
    # Memory efficiency benchmark
    print("Running memory efficiency benchmark...")
    memory_results = model_benchmark_suite.benchmark_memory_efficiency(
        model, sequence_lengths=[64, 128, 256, 512], batch_sizes=[1, 2, 4]
    )
    results['memory_efficiency'] = memory_results
    
    return results


def compare_optimizations(original_model: nn.Module, optimized_model: nn.Module, 
                         input_data: torch.Tensor) -> Dict[str, Any]:
    """Compare performance between original and optimized models"""
    print("Comparing optimizations...")
    
    comparison_results = optimization_benchmark_suite.benchmark_optimization_impact(
        original_model, optimized_model, input_data
    )
    
    # Visualize comparison
    optimization_benchmark_suite.visualize_optimization_comparison(comparison_results)
    
    return comparison_results


def run_hardware_benchmarks(model: nn.Module, input_data: torch.Tensor) -> Dict[str, Any]:
    """Run hardware-specific benchmarks"""
    results = {}
    
    # CPU vs GPU benchmark
    if torch.cuda.is_available():
        print("Running CPU vs GPU benchmark...")
        cpu_gpu_results = hardware_benchmark_suite.benchmark_cpu_vs_gpu(model, input_data)
        results['cpu_gpu'] = cpu_gpu_results
    
    # Mixed precision benchmark
    print("Running mixed precision benchmark...")
    mp_results = hardware_benchmark_suite.benchmark_mixed_precision(model, input_data)
    results['mixed_precision'] = mp_results
    
    return results


def example_benchmarking():
    """Example of benchmarking usage"""
    print("=== Benchmarking Example ===")
    
    # Create a simple model for benchmarking
    class SimpleModel(nn.Module):
        def __init__(self, input_size=768, hidden_size=2048, output_size=768):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    model = SimpleModel()
    
    # Create test data
    input_data = torch.randn(8, 128, 768)  # batch_size=8, seq_len=128, feature_dim=768
    target_data = torch.randn(8, 128, 768)
    
    # Run model benchmarks
    model_results = run_model_benchmarks(model, input_data, target_data)
    print(f"Model benchmarks completed. Inference time: {model_results['inference'].execution_time:.4f}s")
    
    # Create an "optimized" version (in practice, this would have actual optimizations)
    optimized_model = copy.deepcopy(model)
    
    # Compare optimizations
    comparison = compare_optimizations(model, optimized_model, input_data)
    print(f"Optimization comparison completed. Speedup: {comparison['improvements']['speedup_factor']:.2f}x")
    
    # Run hardware benchmarks
    hw_results = run_hardware_benchmarks(model, input_data)
    print(f"Hardware benchmarks completed.")
    
    # Generate a report
    all_results = list(model_results.values())
    if isinstance(model_results['throughput'], list):
        all_results.extend(model_results['throughput'])
    if isinstance(model_results['memory_efficiency'], list):
        all_results.extend(model_results['memory_efficiency'])
    
    report = benchmark_reporter.generate_performance_report(all_results)
    print("\n" + report)


def example_custom_benchmark():
    """Example of creating custom benchmarks"""
    print("\n=== Custom Benchmark Example ===")
    
    # Add a custom benchmark to the suite
    def custom_matrix_multiply_benchmark():
        """Custom benchmark for matrix multiplication"""
        a = torch.randn(1000, 1000)
        b = torch.randn(1000, 1000)
        
        start_time = time.time()
        result = torch.mm(a, b)
        end_time = time.time()
        
        return {
            'execution_time': end_time - start_time,
            'result_shape': result.shape
        }
    
    benchmark_suite.add_benchmark("matrix_multiply", custom_matrix_multiply_benchmark, 
                                "Benchmark matrix multiplication performance")
    
    # Run the custom benchmark
    result = benchmark_suite.run_benchmark("matrix_multiply")
    print(f"Custom benchmark result: {result.execution_time:.4f}s")


if __name__ == "__main__":
    example_benchmarking()
    example_custom_benchmark()