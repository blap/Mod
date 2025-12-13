"""
System-Level Benchmarks for Qwen3-VL Model Optimizations
This module implements comprehensive system-level benchmarks that measure overall performance improvements.
"""
import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
import gc
import psutil
import GPUtil
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import platform

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qwen3_vl.config.config import Qwen3VLConfig
from models.modeling_qwen3_vl_phase2 import Qwen3VLForConditionalGeneration
from benchmarks.benchmark_utils import benchmark_model_inference, profile_memory_usage


@dataclass
class SystemBenchmarkConfig:
    """Configuration for system-level benchmarks"""
    # Model parameters
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    vocab_size: int = 1000
    max_position_embeddings: int = 256
    
    # System benchmark parameters
    test_duration: int = 60  # seconds for extended tests
    concurrent_users: List[int] = None
    workload_patterns: List[str] = None  # 'light', 'medium', 'heavy'
    
    def __post_init__(self):
        if self.concurrent_users is None:
            self.concurrent_users = [1, 2, 4, 8]
        if self.workload_patterns is None:
            self.workload_patterns = ['light', 'medium', 'heavy']


class SystemBenchmarker:
    """Class to run system-level benchmarks"""
    
    def __init__(self, config: SystemBenchmarkConfig = None):
        self.config = config or SystemBenchmarkConfig()
        self.system_info = self.get_system_info()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'platform_release': platform.release(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            system_info.update({
                'gpu_name': gpu,
                'gpu_memory_gb': gpu_memory,
                'cuda_version': torch.version.cuda,
                'gpu_count': torch.cuda.device_count()
            })
        
        # Get CPU information
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
            
            system_info['cpu_model'] = cpu_name
        except:
            system_info['cpu_model'] = "Unknown"
        
        return system_info
    
    def create_models(self, use_optimizations: bool = True) -> Tuple[nn.Module, Qwen3VLConfig]:
        """Create a model with or without optimizations"""
        qwen_config = Qwen3VLConfig()
        qwen_config.hidden_size = self.config.hidden_size
        qwen_config.num_hidden_layers = self.config.num_hidden_layers
        qwen_config.num_attention_heads = self.config.num_attention_heads
        qwen_config.vocab_size = self.config.vocab_size
        qwen_config.max_position_embeddings = self.config.max_position_embeddings
        
        if use_optimizations:
            # Enable all optimizations
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
            # Disable all optimizations
            qwen_config.use_sparsity = False
            qwen_config.use_gradient_checkpointing = False
            qwen_config.use_moe = False
            qwen_config.use_flash_attention_2 = False
            qwen_config.use_dynamic_sparse_attention = False
            qwen_config.use_adaptive_depth = False
            qwen_config.use_context_adaptive_positional_encoding = False
            qwen_config.use_conditional_feature_extraction = False
        
        model = Qwen3VLForConditionalGeneration(qwen_config)
        model.eval()
        return model, qwen_config
    
    def create_workload_inputs(self, workload_pattern: str, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create inputs based on workload pattern"""
        if workload_pattern == 'light':
            seq_len = 64
            image_size = 224
        elif workload_pattern == 'medium':
            seq_len = 128
            image_size = 336
        else:  # heavy
            seq_len = 256
            image_size = 448
        
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        
        return input_ids, pixel_values
    
    def benchmark_single_user_performance(self, device: torch.device) -> Dict[str, Any]:
        """Benchmark single-user performance"""
        print("Benchmarking single-user performance...")
        
        # Create optimized model
        model, _ = self.create_models(use_optimizations=True)
        model = model.to(device)
        
        # Define test scenarios
        test_scenarios = [
            {'batch_size': 1, 'seq_len': 64, 'image_size': 224, 'name': 'light'},
            {'batch_size': 1, 'seq_len': 128, 'image_size': 336, 'name': 'medium'},
            {'batch_size': 1, 'seq_len': 256, 'image_size': 448, 'name': 'heavy'}
        ]
        
        single_user_results = {}
        
        for scenario in test_scenarios:
            batch_size = scenario['batch_size']
            seq_len = scenario['seq_len']
            image_size = scenario['image_size']
            name = scenario['name']
            
            print(f"  Testing {name} workload...")
            
            # Create inputs
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
            pixel_values = torch.randn(batch_size, 3, image_size, image_size).to(device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(input_ids=input_ids, pixel_values=pixel_values)
            
            # Benchmark
            times = []
            for _ in range(10):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    _ = model(input_ids=input_ids, pixel_values=pixel_values)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
            
            times = np.array(times)
            single_user_results[name] = {
                'input_config': {
                    'batch_size': batch_size,
                    'sequence_length': seq_len,
                    'image_size': image_size
                },
                'avg_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'throughput_samples_per_sec': float(batch_size / np.mean(times)),
                'times': times.tolist()
            }
        
        return single_user_results
    
    def benchmark_concurrent_performance(self, device: torch.device) -> Dict[str, Any]:
        """Benchmark performance under concurrent load"""
        print("Benchmarking concurrent performance...")
        
        # Create optimized model
        model, _ = self.create_models(use_optimizations=True)
        model = model.to(device)
        
        concurrent_results = {}
        
        for num_users in self.config.concurrent_users:
            print(f"  Testing with {num_users} concurrent users...")
            
            # Create inputs for each user
            user_inputs = []
            for _ in range(num_users):
                input_ids, pixel_values = self.create_workload_inputs('medium', 1)
                input_ids = input_ids.to(device)
                pixel_values = pixel_values.to(device)
                user_inputs.append((input_ids, pixel_values))
            
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    for input_ids, pixel_values in user_inputs:
                        _ = model(input_ids=input_ids, pixel_values=pixel_values)
            
            # Benchmark concurrent performance
            start_time = time.time()
            total_processed = 0
            
            # Run for a fixed duration
            while time.time() - start_time < 10:  # 10 seconds
                for input_ids, pixel_values in user_inputs:
                    with torch.no_grad():
                        _ = model(input_ids=input_ids, pixel_values=pixel_values)
                    total_processed += 1
            
            elapsed_time = time.time() - start_time
            throughput = total_processed / elapsed_time
            
            concurrent_results[f"{num_users}_users"] = {
                'num_concurrent_users': num_users,
                'total_processed': total_processed,
                'elapsed_time': elapsed_time,
                'throughput_requests_per_sec': throughput,
                'throughput_per_user': throughput / num_users if num_users > 0 else 0
            }
        
        return concurrent_results
    
    def benchmark_system_stability(self, device: torch.device) -> Dict[str, Any]:
        """Benchmark system stability over extended period"""
        print("Benchmarking system stability...")
        
        # Create optimized model
        model, _ = self.create_models(use_optimizations=True)
        model = model.to(device)
        
        # Use medium workload for stability test
        input_ids, pixel_values = self.create_workload_inputs('medium', 1)
        input_ids = input_ids.to(device)
        pixel_values = pixel_values.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
        
        # Monitor system during extended run
        start_time = time.time()
        operations_completed = 0
        
        # Track system metrics over time
        cpu_usage_history = []
        memory_usage_history = []
        if torch.cuda.is_available():
            gpu_usage_history = []
            gpu_memory_history = []
        
        while time.time() - start_time < self.config.test_duration:
            # Perform inference
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
            
            operations_completed += 1
            
            # Monitor system resources
            cpu_usage_history.append(psutil.cpu_percent())
            memory_usage_history.append(psutil.virtual_memory().percent)
            
            if torch.cuda.is_available():
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage_history.append(gpus[0].load * 100)
                    gpu_memory_history.append(gpus[0].memoryUtil * 100)
        
        total_time = time.time() - start_time
        
        # Calculate stability metrics
        avg_cpu_usage = np.mean(cpu_usage_history)
        avg_memory_usage = np.mean(memory_usage_history)
        cpu_usage_std = np.std(cpu_usage_history)
        memory_usage_std = np.std(memory_usage_history)
        
        stability_results = {
            'total_operations_completed': operations_completed,
            'test_duration_seconds': total_time,
            'average_throughput': operations_completed / total_time,
            'cpu_usage_history': cpu_usage_history,
            'memory_usage_history': memory_usage_history,
            'avg_cpu_usage_percent': avg_cpu_usage,
            'avg_memory_usage_percent': avg_memory_usage,
            'cpu_usage_std': cpu_usage_std,
            'memory_usage_std': memory_usage_std,
            'cpu_usage_stable': cpu_usage_std < 10,  # CPU usage should be relatively stable
            'memory_usage_stable': memory_usage_std < 5  # Memory usage should be relatively stable
        }
        
        if torch.cuda.is_available():
            stability_results.update({
                'gpu_usage_history': gpu_usage_history,
                'gpu_memory_history': gpu_memory_history,
                'avg_gpu_usage_percent': np.mean(gpu_usage_history),
                'avg_gpu_memory_percent': np.mean(gpu_memory_history),
                'gpu_usage_std': np.std(gpu_usage_history),
                'gpu_memory_std': np.std(gpu_memory_history),
                'gpu_usage_stable': np.std(gpu_usage_history) < 15,
                'gpu_memory_stable': np.std(gpu_memory_history) < 10
            })
        
        return stability_results
    
    def benchmark_power_efficiency(self, device: torch.device) -> Dict[str, Any]:
        """Benchmark power efficiency (approximation using CPU/GPU utilization)"""
        print("Benchmarking power efficiency...")
        
        # Create models with and without optimizations
        unoptimized_model, _ = self.create_models(use_optimizations=False)
        optimized_model, _ = self.create_models(use_optimizations=True)
        
        unoptimized_model = unoptimized_model.to(device)
        optimized_model = optimized_model.to(device)
        
        # Use consistent workload
        input_ids, pixel_values = self.create_workload_inputs('medium', 2)
        input_ids = input_ids.to(device)
        pixel_values = pixel_values.to(device)
        
        # Test unoptimized model
        start_time = time.time()
        start_cpu_percent = psutil.cpu_percent()
        if torch.cuda.is_available():
            start_gpu_percent = GPUtil.getGPUs()[0].load if GPUtil.getGPUs() else 0
        else:
            start_gpu_percent = 0
        
        with torch.no_grad():
            for _ in range(20):
                _ = unoptimized_model(input_ids=input_ids, pixel_values=pixel_values)
        
        unoptimized_time = time.time() - start_time
        unoptimized_cpu_util = psutil.cpu_percent()
        if torch.cuda.is_available():
            unoptimized_gpu_util = GPUtil.getGPUs()[0].load if GPUtil.getGPUs() else 0
        else:
            unoptimized_gpu_util = 0
        
        # Test optimized model
        start_time = time.time()
        with torch.no_grad():
            for _ in range(20):
                _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
        
        optimized_time = time.time() - start_time
        optimized_cpu_util = psutil.cpu_percent()
        if torch.cuda.is_available():
            optimized_gpu_util = GPUtil.getGPUs()[0].load if GPUtil.getGPUs() else 0
        else:
            optimized_gpu_util = 0
        
        # Calculate power efficiency metrics (approximation)
        power_efficiency_results = {
            'unoptimized': {
                'execution_time': unoptimized_time,
                'avg_cpu_utilization': unoptimized_cpu_util,
                'avg_gpu_utilization': unoptimized_gpu_util
            },
            'optimized': {
                'execution_time': optimized_time,
                'avg_cpu_utilization': optimized_cpu_util,
                'avg_gpu_utilization': optimized_gpu_util
            },
            'improvement': {
                'time_improvement_factor': unoptimized_time / optimized_time if optimized_time > 0 else 0,
                'cpu_utilization_improvement': (unoptimized_cpu_util - optimized_cpu_util) / unoptimized_cpu_util * 100 if unoptimized_cpu_util > 0 else 0,
                'gpu_utilization_improvement': (unoptimized_gpu_util - optimized_gpu_util) / unoptimized_gpu_util * 100 if unoptimized_gpu_util > 0 else 0
            }
        }
        
        return power_efficiency_results
    
    def run_all_system_benchmarks(self, device: torch.device = None) -> Dict[str, Any]:
        """Run all system-level benchmarks"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("=" * 80)
        print("SYSTEM-LEVEL BENCHMARKS FOR QWEN3-VL OPTIMIZATIONS")
        print("=" * 80)
        
        print(f"System Info: {json.dumps(self.system_info, indent=2)}")
        print(f"Using device: {device}")
        
        results = {}
        
        # Run single-user performance benchmark
        results['single_user_performance'] = self.benchmark_single_user_performance(device)
        
        # Run concurrent performance benchmark
        results['concurrent_performance'] = self.benchmark_concurrent_performance(device)
        
        # Run system stability benchmark
        results['system_stability'] = self.benchmark_system_stability(device)
        
        # Run power efficiency benchmark
        results['power_efficiency'] = self.benchmark_power_efficiency(device)
        
        # Generate summary
        summary = self.generate_summary(results)
        
        print("\n" + "=" * 80)
        print("SYSTEM-LEVEL BENCHMARK SUMMARY")
        print("=" * 80)
        
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return {
            'system_info': self.system_info,
            'results': results,
            'summary': summary
        }
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of system-level benchmark results"""
        summary = {
            'target_hardware': 'Intel i5-10210U + NVIDIA SM61 + NVMe SSD',
            'system_info_summary': {
                'cpu_model': self.system_info.get('cpu_model', 'Unknown'),
                'gpu_model': self.system_info.get('gpu_name', 'Unknown'),
                'total_memory_gb': self.system_info.get('total_memory_gb', 0)
            }
        }
        
        # Single user performance summary
        if 'single_user_performance' in results:
            single_perf = results['single_user_performance']
            light_perf = single_perf.get('light', {})
            medium_perf = single_perf.get('medium', {})
            heavy_perf = single_perf.get('heavy', {})
            
            summary['single_user_throughput_light'] = light_perf.get('throughput_samples_per_sec', 0)
            summary['single_user_throughput_medium'] = medium_perf.get('throughput_samples_per_sec', 0)
            summary['single_user_throughput_heavy'] = heavy_perf.get('throughput_samples_per_sec', 0)
        
        # Concurrent performance summary
        if 'concurrent_performance' in results:
            conc_perf = results['concurrent_performance']
            
            # Calculate throughput degradation under load
            baseline_throughput = conc_perf.get('1_users', {}).get('throughput_per_user', 0)
            max_users = max(int(k.split('_')[0]) for k in conc_perf.keys())
            max_load_throughput = conc_perf.get(f'{max_users}_users', {}).get('throughput_per_user', 0)
            
            if baseline_throughput > 0:
                throughput_degradation = (baseline_throughput - max_load_throughput) / baseline_throughput * 100
            else:
                throughput_degradation = 0
            
            summary['baseline_throughput_per_user'] = baseline_throughput
            summary['max_concurrent_throughput_per_user'] = max_load_throughput
            summary['throughput_degradation_under_max_load_percent'] = throughput_degradation
            summary['concurrent_scalability_good'] = throughput_degradation < 50  # Less than 50% degradation is good
        
        # System stability summary
        if 'system_stability' in results:
            stability = results['system_stability']
            
            summary['average_system_throughput'] = stability.get('average_throughput', 0)
            summary['cpu_usage_stable'] = stability.get('cpu_usage_stable', False)
            summary['memory_usage_stable'] = stability.get('memory_usage_stable', False)
            summary['system_stability_score'] = (
                (100 - stability.get('cpu_usage_std', 0)) * 0.4 +
                (100 - stability.get('memory_usage_std', 0)) * 0.4 +
                (stability.get('average_throughput', 0) * 10) * 0.2
            )
        
        # Power efficiency summary
        if 'power_efficiency' in results:
            power_eff = results['power_efficiency']
            
            summary['time_improvement_factor'] = power_eff['improvement'].get('time_improvement_factor', 1.0)
            summary['cpu_utilization_improvement_percent'] = power_eff['improvement'].get('cpu_utilization_improvement', 0)
            summary['gpu_utilization_improvement_percent'] = power_eff['improvement'].get('gpu_utilization_improvement', 0)
        
        # Overall system performance assessment
        performance_good = (
            summary.get('single_user_throughput_medium', 0) > 0.5 and  # At least 0.5 samples/sec for medium workload
            summary.get('concurrent_scalability_good', False) and
            summary.get('system_stability_score', 0) > 70  # Stability score > 70
        )
        
        efficiency_good = (
            summary.get('time_improvement_factor', 1.0) > 1.2 and  # At least 20% time improvement
            summary.get('cpu_utilization_improvement_percent', 0) > 5  # At least 5% CPU utilization improvement
        )
        
        summary['overall_system_performance_good'] = performance_good
        summary['overall_system_efficiency_good'] = efficiency_good
        summary['system_optimization_effective'] = performance_good and efficiency_good
        
        return summary


def run_system_benchmarks():
    """Run all system-level benchmarks"""
    config = SystemBenchmarkConfig()
    benchmarker = SystemBenchmarker(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = benchmarker.run_all_system_benchmarks(device)
    
    # Save results
    Path("benchmark_results").mkdir(parents=True, exist_ok=True)
    with open("benchmark_results/system_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_system_benchmarks()