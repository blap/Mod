"""
Main benchmark runner to coordinate different benchmark types.
"""

from typing import Dict, Any
import torch
import os
from datetime import datetime


def run_all_benchmarks(
    model: torch.nn.Module,
    baseline_model: torch.nn.Module,
    config: Dict[str, Any]
):
    """
    Run all types of benchmarks: inference speed, memory usage, optimization impact, and accuracy.
    
    Args:
        model: The model to benchmark
        baseline_model: The baseline model for comparison
        config: Configuration dictionary with benchmark parameters
    
    Returns:
        Dictionary with all benchmark results
    """
    print("Starting comprehensive benchmark suite...")
    
    # Import benchmark modules
    from .inference.inference_speed_benchmark import run_inference_benchmarks, save_inference_benchmark_results
    from .memory.memory_usage_benchmark import run_memory_benchmarks, save_memory_benchmark_results
    from .optimization.optimization_impact_benchmark import run_optimization_impact_benchmarks, save_optimization_impact_results
    from .accuracy.accuracy_benchmark import run_accuracy_benchmarks, save_accuracy_benchmark_results
    
    # Run each benchmark type
    print("Running inference speed benchmarks...")
    inference_results = run_inference_benchmarks(model, config)
    
    print("Running memory usage benchmarks...")
    memory_results = run_memory_benchmarks(model, config)
    
    print("Running optimization impact benchmarks...")
    optimization_results = run_optimization_impact_benchmarks(model, baseline_model, config)
    
    print("Running accuracy benchmarks...")
    accuracy_results = run_accuracy_benchmarks(model, baseline_model, config)
    
    # Combine all results
    all_results = {
        'benchmark_suite': 'comprehensive',
        'timestamp': datetime.now().isoformat(),
        'inference_results': inference_results,
        'memory_results': memory_results,
        'optimization_results': optimization_results,
        'accuracy_results': accuracy_results
    }
    
    return all_results


def save_all_benchmark_results(results: Dict[str, Any], output_dir: str = "benchmark_results"):
    """
    Save all benchmark results to files.
    
    Args:
        results: Results dictionary to save
        output_dir: Directory to save results to
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save combined results
    import json
    json_filename = os.path.join(output_dir, f"comprehensive_benchmark_results_{timestamp}.json")
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"All benchmark results saved to: {json_filename}")
    
    # Also save individual results using the specific save functions
    from .inference.inference_speed_benchmark import save_inference_benchmark_results
    from .memory.memory_usage_benchmark import save_memory_benchmark_results
    from .optimization.optimization_impact_benchmark import save_optimization_impact_results
    from .accuracy.accuracy_benchmark import save_accuracy_benchmark_results
    
    save_inference_benchmark_results(results['inference_results'], output_dir)
    save_memory_benchmark_results(results['memory_results'], output_dir)
    save_optimization_impact_results(results['optimization_results'], output_dir)
    save_accuracy_benchmark_results(results['accuracy_results'], output_dir)


def run_benchmark_suite(model: torch.nn.Module, baseline_model: torch.nn.Module, config: Dict[str, Any]):
    """
    Main function to run the complete benchmark suite.
    
    Args:
        model: The model to benchmark
        baseline_model: The baseline model for comparison
        config: Configuration dictionary with benchmark parameters
    
    Returns:
        Dictionary with all benchmark results
    """
    # Run all benchmarks
    results = run_all_benchmarks(model, baseline_model, config)
    
    # Save results
    save_all_benchmark_results(results)
    
    # Print summary
    print("\nBenchmark Suite Summary:")
    print(f"  - Inference speed: {results['inference_results']['num_configurations_tested']} configurations tested")
    print(f"  - Memory usage: {results['memory_results']['num_configurations_tested']} configurations tested")
    print(f"  - Optimization impact: {results['optimization_results']['num_configurations_tested']} configurations tested")
    print(f"  - Accuracy: Baseline {results['accuracy_results']['accuracy_vs_baseline_result']['baseline_accuracy_percentage']:.2f}% vs Model {results['accuracy_results']['accuracy_vs_baseline_result']['model_accuracy_percentage']:.2f}%")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Example usage of comprehensive benchmark suite")
    
    # Create dummy models for demonstration
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 128)
        
        def forward(self, x):
            return self.linear(x)
    
    model = DummyModel()
    baseline_model = DummyModel()
    
    # Configuration for benchmarks
    config = {
        'input_sizes': [1, 4],
        'sequence_lengths': [64, 128],
        'num_iterations': 50,
        'warmup_iterations': 5
    }
    
    # Run the complete benchmark suite
    results = run_benchmark_suite(model, baseline_model, config)
    
    print("Comprehensive benchmark suite completed.")