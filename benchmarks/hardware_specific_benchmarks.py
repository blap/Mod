"""
Hardware-Specific Performance Benchmarks for Target System
This module implements benchmarks specifically designed for the Intel i5-10210U + NVIDIA SM61 + NVMe SSD system.
"""
import sys
import os
import time
import torch
import torch.nn as nn
import psutil
import GPUtil
import numpy as np
from typing import Dict, Any, List, Tuple
import subprocess
import platform
from dataclasses import dataclass
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qwen3_vl.config.config import Qwen3VLConfig
from models.modeling_qwen3_vl_phase2 import Qwen3VLForConditionalGeneration
from benchmarks.benchmark_utils import benchmark_model_inference, profile_memory_usage


@dataclass
class HardwareBenchmarkConfig:
    """Configuration for hardware-specific benchmarks"""
    # Model parameters for hardware testing
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    vocab_size: int = 1000
    max_position_embeddings: int = 256
    
    # Hardware-specific parameters
    cpu_threads: int = 4  # i5-10210U has 4 cores, 8 threads
    gpu_memory_limit: int = 4096  # SM61 has limited VRAM, set limit to 4GB
    
    # Benchmark parameters
    num_warmup_runs: int = 5
    num_benchmark_runs: int = 15
    stress_test_duration: int = 30  # seconds


class HardwareSpecificBenchmarker:
    """Class to run hardware-specific benchmarks"""
    
    def __init__(self, config: HardwareBenchmarkConfig = None):
        self.config = config or HardwareBenchmarkConfig()
        self.hardware_info = self.get_hardware_info()
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get information about the target hardware"""
        hardware_info = {
            'system': platform.system(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            hardware_info.update({
                'gpu_name': gpu,
                'gpu_memory_gb': gpu_memory,
                'cuda_version': torch.version.cuda
            })
        
        # Try to get more detailed CPU info
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"], 
                    capture_output=True, text=True
                )
                cpu_name = result.stdout.strip().split('\n')[1].strip() if len(result.stdout.split('\n')) > 1 else "Unknown"
            else:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            cpu_name = line.split(':')[1].strip()
                            break
                    else:
                        cpu_name = "Unknown"
            
            hardware_info['cpu_model'] = cpu_name
        except:
            hardware_info['cpu_model'] = "Unknown"
        
        return hardware_info
    
    def create_model_for_hardware(self, use_optimizations: bool = True) -> Tuple[nn.Module, Qwen3VLConfig]:
        """Create a model configured for the specific hardware"""
        qwen_config = Qwen3VLConfig()
        qwen_config.hidden_size = self.config.hidden_size
        qwen_config.num_hidden_layers = self.config.num_hidden_layers
        qwen_config.num_attention_heads = self.config.num_attention_heads
        qwen_config.vocab_size = self.config.vocab_size
        qwen_config.max_position_embeddings = self.config.max_position_embeddings
        
        # Apply optimizations appropriate for hardware constraints
        if use_optimizations:
            # Enable optimizations suitable for resource-constrained hardware
            qwen_config.use_sparsity = True
            qwen_config.sparsity_ratio = 0.5  # Moderate sparsity for efficiency
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
        
        model = Qwen3VLForConditionalGeneration(qwen_config)
        return model, qwen_config
    
    def benchmark_cpu_performance(self) -> Dict[str, Any]:
        """Benchmark CPU performance on the target hardware"""
        print("Benchmarking CPU performance...")
        
        # Create model for CPU
        model, _ = self.create_model_for_hardware(use_optimizations=True)
        model = model.to('cpu')
        model.eval()
        
        # Set number of threads to match hardware capabilities
        torch.set_num_threads(self.config.cpu_threads)
        
        # Define test inputs
        batch_size, seq_len = 1, 128
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.config.num_warmup_runs):
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
        
        # Benchmark
        times = []
        for _ in range(self.config.num_benchmark_runs):
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        times = np.array(times)
        cpu_results = {
            'avg_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'throughput_samples_per_sec': float(batch_size / np.mean(times)),
            'num_threads_used': self.config.cpu_threads,
            'cpu_utilization_peak': max(psutil.cpu_percent(interval=1, percpu=True)),
            'memory_usage_mb': psutil.Process().memory_info().rss / (1024**2)
        }
        
        print(f"  CPU avg time: {cpu_results['avg_time']:.4f}s")
        print(f"  CPU throughput: {cpu_results['throughput_samples_per_sec']:.2f} samples/sec")
        
        return cpu_results
    
    def benchmark_gpu_performance(self) -> Dict[str, Any]:
        """Benchmark GPU performance on the target hardware (if available)"""
        if not torch.cuda.is_available():
            print("GPU not available, skipping GPU benchmark")
            return {"gpu_available": False}
        
        print("Benchmarking GPU performance...")
        
        # Create model for GPU
        model, _ = self.create_model_for_hardware(use_optimizations=True)
        device = torch.device('cuda')
        model = model.to(device)
        model.eval()
        
        # Define test inputs
        batch_size, seq_len = 1, 128
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
        pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.config.num_warmup_runs):
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        
        # Benchmark
        times = []
        for _ in range(self.config.num_benchmark_runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        times = np.array(times)
        
        # Get memory stats
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        reserved_memory = torch.cuda.max_memory_reserved() / (1024**2)  # MB
        
        gpu_results = {
            'avg_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'throughput_samples_per_sec': float(batch_size / np.mean(times)),
            'peak_memory_mb': peak_memory,
            'reserved_memory_mb': reserved_memory,
            'gpu_utilization_avg': np.mean([g.load for g in GPUtil.getGPUs()]) if GPUtil.getGPUs() else 0,
            'gpu_memory_gb': self.hardware_info.get('gpu_memory_gb', 0)
        }
        
        print(f"  GPU avg time: {gpu_results['avg_time']:.4f}s")
        print(f"  GPU throughput: {gpu_results['throughput_samples_per_sec']:.2f} samples/sec")
        print(f"  GPU peak memory: {gpu_results['peak_memory_mb']:.2f} MB")
        
        return gpu_results
    
    def benchmark_memory_bandwidth(self) -> Dict[str, Any]:
        """Benchmark memory bandwidth relevant to the target hardware"""
        print("Benchmarking memory bandwidth...")
        
        # Create large tensors to test memory bandwidth
        size = 1000  # Reduced for testing
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # Warmup
        _ = torch.mm(a, b)
        
        # Benchmark
        times = []
        for _ in range(10):
            start_time = time.perf_counter()
            result = torch.mm(a, b)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        # Rough estimate of memory bandwidth based on tensor size and time
        tensor_size_gb = (size * size * 4 * 3) / (1024**3)  # 2 inputs + 1 output, float32
        bandwidth_gbs = tensor_size_gb / avg_time
        
        memory_bandwidth_results = {
            'avg_computation_time': avg_time,
            'estimated_bandwidth_gbs': bandwidth_gbs,
            'operation_size_gb': tensor_size_gb
        }
        
        print(f"  Estimated memory bandwidth: {bandwidth_gbs:.2f} GB/s")
        
        return memory_bandwidth_results
    
    def benchmark_storage_performance(self) -> Dict[str, Any]:
        """Benchmark storage performance (relevant for NVMe SSD)"""
        print("Benchmarking storage performance...")
        
        # Create a temporary file for testing
        test_file = "storage_test.tmp"
        file_size_mb = 100  # Size in MB
        
        # Create test data
        test_data = b"x" * (file_size_mb * 1024 * 1024)  # 100MB of test data
        
        # Write test
        write_start = time.time()
        with open(test_file, "wb") as f:
            f.write(test_data)
        write_time = time.time() - write_start
        
        # Read test
        read_start = time.time()
        with open(test_file, "rb") as f:
            _ = f.read()
        read_time = time.time() - read_start
        
        # Cleanup
        os.remove(test_file)
        
        storage_results = {
            'write_speed_mbs': file_size_mb / write_time,
            'read_speed_mbs': file_size_mb / read_time,
            'file_size_mb': file_size_mb,
            'write_time_s': write_time,
            'read_time_s': read_time
        }
        
        print(f"  Storage write speed: {storage_results['write_speed_mbs']:.2f} MB/s")
        print(f"  Storage read speed: {storage_results['read_speed_mbs']:.2f} MB/s")
        
        return storage_results
    
    def stress_test_performance(self) -> Dict[str, Any]:
        """Run a stress test to evaluate performance under load"""
        print("Running stress test...")
        
        # Create model
        model, _ = self.create_model_for_hardware(use_optimizations=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Define test inputs
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
        pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # Monitor system resources during stress test
        start_time = time.time()
        operations_completed = 0
        
        # Track resource usage
        cpu_usage_over_time = []
        memory_usage_over_time = []
        if torch.cuda.is_available():
            gpu_usage_over_time = []
            gpu_memory_over_time = []
        
        while time.time() - start_time < self.config.stress_test_duration:
            # Perform inference
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
            
            operations_completed += 1
            
            # Monitor resources
            cpu_usage_over_time.append(psutil.cpu_percent())
            memory_usage_over_time.append(psutil.virtual_memory().percent)
            
            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage_over_time.append(gpus[0].load * 100)
                    gpu_memory_over_time.append(gpus[0].memoryUtil * 100)
        
        # Calculate results
        total_time = time.time() - start_time
        avg_cpu_usage = np.mean(cpu_usage_over_time)
        avg_memory_usage = np.mean(memory_usage_over_time)
        
        stress_results = {
            'operations_completed': operations_completed,
            'total_time_s': total_time,
            'throughput_ops_per_sec': operations_completed / total_time,
            'avg_cpu_usage_percent': avg_cpu_usage,
            'avg_memory_usage_percent': avg_memory_usage,
            'peak_cpu_usage_percent': max(cpu_usage_over_time),
            'peak_memory_usage_percent': max(memory_usage_over_time),
            'test_duration_s': self.config.stress_test_duration
        }
        
        if torch.cuda.is_available():
            stress_results.update({
                'avg_gpu_usage_percent': np.mean(gpu_usage_over_time),
                'avg_gpu_memory_percent': np.mean(gpu_memory_over_time),
                'peak_gpu_usage_percent': max(gpu_usage_over_time),
                'peak_gpu_memory_percent': max(gpu_memory_over_time)
            })
        
        print(f"  Stress test throughput: {stress_results['throughput_ops_per_sec']:.2f} ops/sec")
        print(f"  Avg CPU usage: {stress_results['avg_cpu_usage_percent']:.1f}%")
        print(f"  Avg memory usage: {stress_results['avg_memory_usage_percent']:.1f}%")
        
        return stress_results
    
    def run_all_hardware_benchmarks(self) -> Dict[str, Any]:
        """Run all hardware-specific benchmarks"""
        print("=" * 80)
        print("HARDWARE-SPECIFIC PERFORMANCE BENCHMARKS")
        print(f"Target System: Intel i5-10210U + NVIDIA SM61 + NVMe SSD")
        print("=" * 80)
        
        print(f"Hardware Info: {json.dumps(self.hardware_info, indent=2)}")
        
        results = {}
        
        # Run CPU benchmark
        results['cpu_performance'] = self.benchmark_cpu_performance()
        
        # Run GPU benchmark if available
        results['gpu_performance'] = self.benchmark_gpu_performance()
        
        # Run memory bandwidth benchmark
        results['memory_bandwidth'] = self.benchmark_memory_bandwidth()
        
        # Run storage performance benchmark
        results['storage_performance'] = self.benchmark_storage_performance()
        
        # Run stress test
        results['stress_test'] = self.stress_test_performance()
        
        # Generate summary
        summary = self.generate_summary(results)
        
        print("\n" + "=" * 80)
        print("HARDWARE BENCHMARK SUMMARY")
        print("=" * 80)
        
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return {
            'hardware_info': self.hardware_info,
            'results': results,
            'summary': summary
        }
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of hardware benchmark results"""
        summary = {
            'hardware_target': 'Intel i5-10210U + NVIDIA SM61 + NVMe SSD',
            'cpu_threads_detected': self.config.cpu_threads,
            'gpu_available': torch.cuda.is_available()
        }
        
        # CPU performance summary
        if 'cpu_performance' in results:
            cpu = results['cpu_performance']
            summary['cpu_avg_inference_time'] = f"{cpu['avg_time']:.4f}s"
            summary['cpu_throughput'] = f"{cpu['throughput_samples_per_sec']:.2f} samples/sec"
        
        # GPU performance summary (if available)
        if 'gpu_performance' in results and results['gpu_performance'].get('gpu_available', True):
            gpu = results['gpu_performance']
            summary['gpu_avg_inference_time'] = f"{gpu['avg_time']:.4f}s"
            summary['gpu_throughput'] = f"{gpu['throughput_samples_per_sec']:.2f} samples/sec"
            summary['gpu_peak_memory'] = f"{gpu['peak_memory_mb']:.2f} MB"
        
        # Storage performance summary
        if 'storage_performance' in results:
            storage = results['storage_performance']
            summary['storage_read_speed'] = f"{storage['read_speed_mbs']:.2f} MB/s"
            summary['storage_write_speed'] = f"{storage['write_speed_mbs']:.2f} MB/s"
        
        # Stress test summary
        if 'stress_test' in results:
            stress = results['stress_test']
            summary['stress_test_throughput'] = f"{stress['throughput_ops_per_sec']:.2f} ops/sec"
            summary['avg_cpu_under_load'] = f"{stress['avg_cpu_usage_percent']:.1f}%"
            summary['avg_memory_under_load'] = f"{stress['avg_memory_usage_percent']:.1f}%"
        
        return summary


def run_hardware_specific_benchmarks():
    """Run hardware-specific performance benchmarks"""
    config = HardwareBenchmarkConfig()
    benchmarker = HardwareSpecificBenchmarker(config)
    
    results = benchmarker.run_all_hardware_benchmarks()
    
    # Save results
    Path("benchmark_results").mkdir(parents=True, exist_ok=True)
    with open("benchmark_results/hardware_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_hardware_specific_benchmarks()