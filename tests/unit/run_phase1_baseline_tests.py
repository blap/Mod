"""
Script to run Phase 1 baseline tests for Qwen3-VL-2B-Instruct model
This script will run both CPU and GPU baseline tests and save results in a structured format
"""
import json
import torch
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmarks.cpu.baseline_cpu import run_cpu_baseline_tests
from benchmarks.gpu.baseline_gpu import run_gpu_baseline_tests
from benchmarks.multimodal_benchmark import run_multimodal_benchmarks


def run_all_baseline_tests():
    """
    Run all baseline tests and save results to a structured file
    """
    print("Starting Phase 1 Baseline Tests for Qwen3-VL-2B-Instruct Model")
    print("=" * 70)
    
    # Collect system information
    system_info = {
        'timestamp': datetime.now().isoformat(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'python_version': sys.version,
    }
    
    if torch.cuda.is_available():
        system_info.update({
            'cuda_version': torch.version.cuda,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
            'gpu_compute_capability': torch.cuda.get_device_capability(0),
        })
    
    print(f"System Info: {system_info}")
    
    results = {
        'system_info': system_info,
        'cpu_results': None,
        'gpu_results': None,
        'multimodal_results': None
    }
    
    # Run CPU baseline tests
    print("\n" + "="*50)
    print("RUNNING CPU BASELINE TESTS")
    print("="*50)
    try:
        cpu_results = run_cpu_baseline_tests()
        results['cpu_results'] = cpu_results
        print("CPU baseline tests completed successfully!")
    except Exception as e:
        print(f"Error running CPU baseline tests: {e}")
        import traceback
        traceback.print_exc()
    
    # Run GPU baseline tests if available
    print("\n" + "="*50)
    print("RUNNING GPU BASELINE TESTS")
    print("="*50)
    if torch.cuda.is_available():
        try:
            gpu_results = run_gpu_baseline_tests()
            results['gpu_results'] = gpu_results
            print("GPU baseline tests completed successfully!")
        except Exception as e:
            print(f"Error running GPU baseline tests: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("CUDA not available. Skipping GPU tests.")
    
    # Run multimodal benchmark tests
    print("\n" + "="*50)
    print("RUNNING MULTIMODAL BENCHMARK TESTS")
    print("="*50)
    try:
        multimodal_results = run_multimodal_benchmarks()
        results['multimodal_results'] = multimodal_results
        print("Multimodal benchmark tests completed successfully!")
    except Exception as e:
        print(f"Error running multimodal benchmark tests: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results to file
    output_file = f"phase1_baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n" + "="*70)
    print(f"ALL PHASE 1 BASELINE TESTS COMPLETED!")
    print(f"Results saved to: {output_file}")
    print("="*70)
    
    # Print summary
    print("\nSUMMARY:")
    if results['cpu_results']:
        cpu_inf = results['cpu_results']['inference']
        print(f"  CPU Inference: {cpu_inf['avg_time']:.4f}s avg, {cpu_inf['throughput_samples_per_sec']:.2f} samples/sec")
        
        cpu_multi = results['cpu_results']['multimodal']
        print(f"  CPU Multimodal: {cpu_multi['avg_time']:.4f}s avg")
        
        cpu_mem = results['cpu_results']['memory']
        print(f"  CPU Memory Increase: {cpu_mem['cpu_memory_increase_mb']:.2f} MB")
    
    if results['gpu_results']:
        gpu_inf = results['gpu_results']['inference']
        print(f"  GPU Inference: {gpu_inf['avg_time']:.4f}s avg, {gpu_inf['throughput_samples_per_sec']:.2f} samples/sec")
        
        gpu_multi = results['gpu_results']['multimodal']
        print(f"  GPU Multimodal: {gpu_multi['avg_time']:.4f}s avg")
        
        gpu_mem = results['gpu_results']['memory']
        print(f"  GPU Memory Increase: {gpu_mem['gpu_memory_increase_mb']:.2f} MB")
        print(f"  GPU Peak Memory: {gpu_mem['gpu_peak_memory_mb']:.2f} MB")
    
    if results['multimodal_results']:
        for device, device_results in results['multimodal_results'].items():
            if 'multimodal' in device_results:
                multi_time = device_results['multimodal']['avg_time']
                throughput = device_results['multimodal']['throughput_samples_per_sec']
                print(f"  {device.upper()} Multimodal: {multi_time:.4f}s avg, {throughput:.2f} samples/sec")
    
    return results


if __name__ == "__main__":
    run_all_baseline_tests()