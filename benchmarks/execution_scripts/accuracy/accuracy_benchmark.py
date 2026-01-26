"""
Focused benchmark module for accuracy measurements.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple
import json
import csv
from datetime import datetime


def benchmark_accuracy(
    model: torch.nn.Module, 
    test_data: List[Tuple[torch.Tensor, torch.Tensor]], 
    criterion=F.cross_entropy
):
    """
    Benchmark the accuracy of a model on test data.
    
    Args:
        model: The model to benchmark
        test_data: List of (input, target) tuples
        criterion: Loss function to use for evaluation
    
    Returns:
        Dictionary with accuracy benchmark results
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in test_data:
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(test_data)
    accuracy = 100 * correct_predictions / total_samples
    
    results = {
        'benchmark_type': 'accuracy',
        'average_loss': avg_loss,
        'accuracy_percentage': accuracy,
        'correct_predictions': correct_predictions,
        'total_samples': total_samples,
        'num_batches': len(test_data),
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def benchmark_accuracy_with_noise(
    model: torch.nn.Module, 
    test_data: List[Tuple[torch.Tensor, torch.Tensor]], 
    noise_levels: List[float] = [0.0, 0.01, 0.05, 0.1]
):
    """
    Benchmark accuracy under different noise levels.
    
    Args:
        model: The model to benchmark
        test_data: List of (input, target) tuples
        noise_levels: List of noise levels to test
    
    Returns:
        Dictionary with accuracy results for different noise levels
    """
    results = {}
    
    for noise_level in noise_levels:
        noisy_test_data = []
        
        for inputs, targets in test_data:
            # Add Gaussian noise to inputs
            noise = torch.randn_like(inputs) * noise_level
            noisy_inputs = inputs + noise
            noisy_test_data.append((noisy_inputs, targets))
        
        accuracy_result = benchmark_accuracy(model, noisy_test_data)
        accuracy_result['noise_level'] = noise_level
        results[f'noise_{noise_level}'] = accuracy_result
    
    return results


def benchmark_accuracy_vs_baseline(
    model: torch.nn.Module,
    baseline_model: torch.nn.Module,
    test_data: List[Tuple[torch.Tensor, torch.Tensor]]
):
    """
    Compare accuracy of model against a baseline model.
    
    Args:
        model: The model to evaluate
        baseline_model: The baseline model to compare against
        test_data: List of (input, target) tuples
    
    Returns:
        Dictionary with comparison results
    """
    model_accuracy_result = benchmark_accuracy(model, test_data)
    baseline_accuracy_result = benchmark_accuracy(baseline_model, test_data)
    
    accuracy_difference = model_accuracy_result['accuracy_percentage'] - baseline_accuracy_result['accuracy_percentage']
    
    results = {
        'benchmark_type': 'accuracy_vs_baseline',
        'model_accuracy_percentage': model_accuracy_result['accuracy_percentage'],
        'baseline_accuracy_percentage': baseline_accuracy_result['accuracy_percentage'],
        'accuracy_difference_percentage': accuracy_difference,
        'model_average_loss': model_accuracy_result['average_loss'],
        'baseline_average_loss': baseline_accuracy_result['average_loss'],
        'loss_difference': model_accuracy_result['average_loss'] - baseline_accuracy_result['average_loss'],
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def run_accuracy_benchmarks(
    model: torch.nn.Module, 
    baseline_model: torch.nn.Module, 
    config: Dict[str, Any]
):
    """
    Run a series of accuracy benchmarks.
    
    Args:
        model: The model to benchmark
        baseline_model: The baseline model to compare against
        config: Configuration dictionary with benchmark parameters
    
    Returns:
        Dictionary with all benchmark results
    """
    print("Running accuracy benchmarks...")
    
    # Get configuration parameters
    num_samples = config.get('num_samples', 1000)
    input_size = config.get('input_size', 128)
    num_classes = config.get('num_classes', 10)
    noise_levels = config.get('noise_levels', [0.0, 0.01, 0.05, 0.1])
    
    # Generate synthetic test data
    test_data = []
    for _ in range(num_samples // 10):  # 10 batches of 100 samples each
        inputs = torch.randn(100, input_size)
        targets = torch.randint(0, num_classes, (100,))
        test_data.append((inputs, targets))
    
    # Run basic accuracy benchmark
    basic_accuracy_result = benchmark_accuracy(model, test_data)
    
    # Run accuracy with noise benchmark
    noise_accuracy_results = benchmark_accuracy_with_noise(model, test_data, noise_levels)
    
    # Run accuracy vs baseline benchmark
    accuracy_vs_baseline_result = benchmark_accuracy_vs_baseline(model, baseline_model, test_data)
    
    aggregate_results = {
        'benchmark_type': 'accuracy_aggregate',
        'basic_accuracy_result': basic_accuracy_result,
        'noise_accuracy_results': noise_accuracy_results,
        'accuracy_vs_baseline_result': accuracy_vs_baseline_result,
        'timestamp': datetime.now().isoformat()
    }
    
    return aggregate_results


def save_accuracy_benchmark_results(results: Dict[str, Any], output_dir: str = "benchmark_results"):
    """
    Save accuracy benchmark results to files.
    
    Args:
        results: Results dictionary to save
        output_dir: Directory to save results to
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_filename = os.path.join(output_dir, f"accuracy_benchmark_results_{timestamp}.json")
    with open(json_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save CSV summary
    csv_filename = os.path.join(output_dir, f"accuracy_benchmark_summary_{timestamp}.csv")
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'Benchmark Type', 'Model Accuracy %', 'Baseline Accuracy %', 
            'Accuracy Difference %', 'Model Average Loss', 'Baseline Average Loss',
            'Loss Difference', 'Timestamp'
        ])
        
        # Write summary row
        baseline_result = results.get('accuracy_vs_baseline_result', {})
        writer.writerow([
            results.get('benchmark_type', ''),
            baseline_result.get('model_accuracy_percentage', 0),
            baseline_result.get('baseline_accuracy_percentage', 0),
            baseline_result.get('accuracy_difference_percentage', 0),
            baseline_result.get('model_average_loss', 0),
            baseline_result.get('baseline_average_loss', 0),
            baseline_result.get('loss_difference', 0),
            results.get('timestamp', '')
        ])
    
    print(f"Accuracy benchmark results saved to: {json_filename} and {csv_filename}")


if __name__ == "__main__":
    # Example usage
    print("Example usage of accuracy benchmark module")
    
    # Create dummy models for demonstration
    class SimpleClassifier(torch.nn.Module):
        def __init__(self, input_size=128, num_classes=10):
            super().__init__()
            self.fc = torch.nn.Linear(input_size, num_classes)
        
        def forward(self, x):
            return self.fc(x)
    
    model = SimpleClassifier()
    baseline_model = SimpleClassifier()
    
    # Configuration for benchmarks
    config = {
        'num_samples': 500,
        'input_size': 128,
        'num_classes': 10,
        'noise_levels': [0.0, 0.05, 0.1]
    }
    
    # Run benchmarks
    results = run_accuracy_benchmarks(model, baseline_model, config)
    
    # Save results
    save_accuracy_benchmark_results(results)
    
    print("Accuracy benchmark example completed.")