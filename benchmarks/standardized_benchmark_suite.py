"""
Standardized Benchmark Suite for Qwen3-VL Model Components

This module provides a comprehensive, standardized benchmark suite for all Qwen3-VL model components
following best practices for performance measurement and ensuring consistency across all benchmarks.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
import psutil
import GPUtil
import gc
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to the path to import modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.qwen3_vl.models.config import Qwen3VLConfig
from src.qwen3_vl.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from benchmarks.benchmark_utils import benchmark_model_inference, profile_memory_usage, benchmark_generation


class StandardizedBenchmarkSuite:
    """Comprehensive benchmark suite for Qwen3-VL model components."""

    def __init__(self):
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def benchmark_model_performance(self, model: nn.Module, input_ids: torch.Tensor, 
                                   pixel_values: torch.Tensor, num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark model performance metrics."""
        print(f"Benchmarking model performance on {self.device}...")

        # Warm up
        model.eval()
        model = model.to(self.device)
        input_ids = input_ids.to(self.device)
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids=input_ids, pixel_values=pixel_values)

        # Time model execution
        times = []
        for _ in range(num_runs):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
            end_time = time.time()

            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        # Calculate throughput (samples per second)
        throughput = 1.0 / avg_time if avg_time > 0 else 0

        # Profile memory usage
        memory_usage = profile_memory_usage(model, {
            'input_ids': input_ids,
            'pixel_values': pixel_values
        }, device=self.device)

        performance_metrics = {
            'average_time_seconds': avg_time,
            'min_time_seconds': min_time,
            'max_time_seconds': max_time,
            'throughput_samples_per_second': throughput,
            'memory_usage': memory_usage,
            'num_runs': num_runs,
            'device': str(self.device)
        }

        return performance_metrics

    def benchmark_multimodal_performance(self) -> Dict[str, Any]:
        """Benchmark multimodal performance of the model."""
        print("Benchmarking multimodal performance...")

        # Create configuration
        config = Qwen3VLConfig()
        config.hidden_size = 512  # Reduced for benchmarking
        config.num_attention_heads = 8
        config.num_hidden_layers = 4  # Reduced for benchmarking
        config.vision_num_hidden_layers = 4  # Reduced for benchmarking

        # Create model
        model = Qwen3VLForConditionalGeneration(config)
        model = model.to(self.device)
        model.eval()

        # Create test inputs
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(self.device)
        pixel_values = torch.randn(batch_size, 3, 224, 224).to(self.device)

        # Benchmark multimodal processing
        multimodal_metrics = self.benchmark_model_performance(
            model, input_ids, pixel_values, num_runs=5
        )

        # Benchmark text-only processing
        text_metrics = self.benchmark_model_performance(
            model, input_ids, torch.randn(1, 3, 224, 224).expand(batch_size, -1, -1, -1).to(self.device), num_runs=5
        )

        # Benchmark image-only processing
        image_metrics = self.benchmark_model_performance(
            model, torch.randint(0, config.vocab_size, (batch_size, 1)).to(self.device), pixel_values, num_runs=5
        )

        multimodal_results = {
            'multimodal_metrics': multimodal_metrics,
            'text_only_metrics': text_metrics,
            'image_only_metrics': image_metrics,
            'comparison': {
                'multimodal_vs_text_only_speedup': text_metrics['average_time_seconds'] / multimodal_metrics['average_time_seconds'],
                'multimodal_vs_image_only_speedup': image_metrics['average_time_seconds'] / multimodal_metrics['average_time_seconds']
            }
        }

        self.results['multimodal_performance'] = multimodal_results
        return multimodal_results

    def benchmark_memory_efficiency(self) -> Dict[str, Any]:
        """Benchmark memory efficiency improvements."""
        print("Benchmarking memory efficiency...")

        # Create configurations for different optimization levels
        baseline_config = Qwen3VLConfig()
        baseline_config.hidden_size = 256
        baseline_config.num_attention_heads = 4
        baseline_config.num_hidden_layers = 2
        baseline_config.vision_num_hidden_layers = 2

        optimized_config = Qwen3VLConfig()
        optimized_config.hidden_size = 256
        optimized_config.num_attention_heads = 4
        optimized_config.num_hidden_layers = 2
        optimized_config.vision_num_hidden_layers = 2
        # Enable optimizations
        optimized_config.use_sparsity = True
        optimized_config.use_gradient_checkpointing = True

        # Create models
        baseline_model = Qwen3VLForConditionalGeneration(baseline_config)
        optimized_model = Qwen3VLForConditionalGeneration(optimized_config)

        # Copy weights to ensure fair comparison
        optimized_model.load_state_dict(baseline_model.state_dict(), strict=False)

        # Create test inputs
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len)).to(self.device)
        pixel_values = torch.randn(batch_size, 3, 224, 224).to(self.device)

        # Profile memory usage for both models
        baseline_memory = profile_memory_usage(baseline_model, {
            'input_ids': input_ids,
            'pixel_values': pixel_values
        }, device=self.device)

        optimized_memory = profile_memory_usage(optimized_model, {
            'input_ids': input_ids,
            'pixel_values': pixel_values
        }, device=self.device)

        memory_results = {
            'baseline_memory': baseline_memory,
            'optimized_memory': optimized_memory,
            'memory_savings_percentage': (
                (baseline_memory['peak_memory_mb'] - optimized_memory['peak_memory_mb']) / 
                baseline_memory['peak_memory_mb'] * 100
            ) if baseline_memory['peak_memory_mb'] > 0 else 0
        }

        self.results['memory_efficiency'] = memory_results
        return memory_results

    def benchmark_throughput_scalability(self) -> Dict[str, Any]:
        """Benchmark throughput scalability with different batch sizes."""
        print("Benchmarking throughput scalability...")

        # Create model
        config = Qwen3VLConfig()
        config.hidden_size = 256
        config.num_attention_heads = 4
        config.num_hidden_layers = 2
        config.vision_num_hidden_layers = 2

        model = Qwen3VLForConditionalGeneration(config)
        model = model.to(self.device)
        model.eval()

        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        throughput_results = {}

        for batch_size in batch_sizes:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, 32)).to(self.device)
            pixel_values = torch.randn(batch_size, 3, 224, 224).to(self.device)

            # Benchmark performance
            metrics = self.benchmark_model_performance(model, input_ids, pixel_values, num_runs=3)
            throughput_results[f'batch_size_{batch_size}'] = metrics

        scalability_results = {
            'batch_sizes': batch_sizes,
            'throughput_results': throughput_results,
            'scalability_analysis': self._analyze_scalability(throughput_results)
        }

        self.results['scalability'] = scalability_results
        return scalability_results

    def _analyze_scalability(self, throughput_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scalability of throughput results."""
        batch_sizes = []
        throughputs = []

        for key, metrics in throughput_results.items():
            batch_size = int(key.split('_')[2])
            batch_sizes.append(batch_size)
            throughputs.append(metrics['throughput_samples_per_second'])

        # Calculate scaling efficiency
        if len(throughputs) > 1:
            ideal_scaling = [throughputs[0] * bs / batch_sizes[0] for bs in batch_sizes]
            actual_efficiency = [actual / ideal for actual, ideal in zip(throughputs, ideal_scaling)]
        else:
            actual_efficiency = [1.0]

        return {
            'batch_sizes': batch_sizes,
            'throughputs': throughputs,
            'ideal_scaling': ideal_scaling if len(throughputs) > 1 else [],
            'scaling_efficiency': actual_efficiency,
            'best_batch_size': batch_sizes[throughputs.index(max(throughputs))] if throughputs else None
        }

    def benchmark_accuracy_preservation(self) -> Dict[str, Any]:
        """Benchmark accuracy preservation with optimizations."""
        print("Benchmarking accuracy preservation...")

        # Create configurations
        baseline_config = Qwen3VLConfig()
        baseline_config.hidden_size = 128
        baseline_config.num_attention_heads = 4
        baseline_config.num_hidden_layers = 2
        baseline_config.vision_num_hidden_layers = 2

        optimized_config = Qwen3VLConfig()
        optimized_config.hidden_size = 128
        optimized_config.num_attention_heads = 4
        optimized_config.num_hidden_layers = 2
        optimized_config.vision_num_hidden_layers = 2
        # Enable various optimizations
        optimized_config.use_sparsity = True
        optimized_config.use_gradient_checkpointing = True
        optimized_config.use_flash_attention_2 = True

        # Create models
        baseline_model = Qwen3VLForConditionalGeneration(baseline_config)
        optimized_model = Qwen3VLForConditionalGeneration(optimized_config)

        # Copy weights to ensure fair comparison
        optimized_model.load_state_dict(baseline_model.state_dict(), strict=False)

        # Create test inputs
        batch_size, seq_len = 1, 8
        input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len)).to(self.device)
        pixel_values = torch.randn(batch_size, 3, 224, 224).to(self.device)

        # Run both models
        baseline_model = baseline_model.to(self.device)
        optimized_model = optimized_model.to(self.device)
        baseline_model.eval()
        optimized_model.eval()

        with torch.no_grad():
            baseline_output = baseline_model(input_ids=input_ids, pixel_values=pixel_values)
            optimized_output = optimized_model(input_ids=input_ids, pixel_values=pixel_values)

        # Calculate similarity metrics
        cosine_similarity = torch.cosine_similarity(
            baseline_output.flatten(),
            optimized_output.flatten(),
            dim=0
        ).item()

        mse_loss = torch.mean((baseline_output - optimized_output) ** 2).item()
        mae_loss = torch.mean(torch.abs(baseline_output - optimized_output)).item()

        accuracy_results = {
            'cosine_similarity': cosine_similarity,
            'mse_loss': mse_loss,
            'mae_loss': mae_loss,
            'similarity_threshold_met': cosine_similarity > 0.95,  # High similarity threshold
            'accuracy_preserved': cosine_similarity > 0.95 and mse_loss < 0.01
        }

        self.results['accuracy_preservation'] = accuracy_results
        return accuracy_results

    def benchmark_generation_performance(self) -> Dict[str, Any]:
        """Benchmark generation performance."""
        print("Benchmarking generation performance...")

        # Create model
        config = Qwen3VLConfig()
        config.hidden_size = 256
        config.num_attention_heads = 4
        config.num_hidden_layers = 2
        config.vision_num_hidden_layers = 2

        model = Qwen3VLForConditionalGeneration(config)
        model = model.to(self.device)
        model.eval()

        # Create test inputs
        batch_size = 1
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 10)).to(self.device)
        pixel_values = torch.randn(batch_size, 3, 224, 224).to(self.device)

        # Benchmark generation
        generation_results = benchmark_generation(
            model, input_ids, pixel_values, device=self.device
        )

        self.results['generation_performance'] = generation_results
        return generation_results

    def benchmark_capacity_preservation(self) -> Dict[str, Any]:
        """Benchmark that model capacity is preserved."""
        print("Benchmarking capacity preservation...")

        # Create full-capacity configuration
        config = Qwen3VLConfig()
        
        # Verify configuration
        capacity_results = {
            'num_hidden_layers': config.num_hidden_layers,
            'num_attention_heads': config.num_attention_heads,
            'vision_num_hidden_layers': config.vision_num_hidden_layers,
            'vision_num_attention_heads': config.vision_num_attention_heads,
            'hidden_size': config.hidden_size,
            'vocab_size': config.vocab_size,
            'capacity_preserved': (
                config.num_hidden_layers == 32 and 
                config.num_attention_heads == 32 and
                config.vision_num_hidden_layers == 24 and
                config.vision_num_attention_heads == 16
            )
        }

        # Create model and verify architecture
        model = Qwen3VLForConditionalGeneration(config)
        model = model.to(self.device)
        model.eval()

        architecture_results = {
            'language_layers_count': len(model.language_model.layers),
            'vision_layers_count': len(model.vision_tower.layers),
            'architecture_matches_config': (
                len(model.language_model.layers) == config.num_hidden_layers and
                len(model.vision_tower.layers) == config.vision_num_hidden_layers
            )
        }

        capacity_results.update(architecture_results)

        self.results['capacity_preservation'] = capacity_results
        return capacity_results

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        print("="*80)
        print("RUNNING COMPREHENSIVE BENCHMARK SUITE FOR QWEN3-VL MODEL")
        print("="*80)

        # Run all benchmarks
        benchmarks = [
            ("Multimodal Performance", self.benchmark_multimodal_performance),
            ("Memory Efficiency", self.benchmark_memory_efficiency),
            ("Throughput Scalability", self.benchmark_throughput_scalability),
            ("Accuracy Preservation", self.benchmark_accuracy_preservation),
            ("Generation Performance", self.benchmark_generation_performance),
            ("Capacity Preservation", self.benchmark_capacity_preservation)
        ]

        for benchmark_name, benchmark_func in benchmarks:
            print(f"\n{benchmark_name}:")
            print("-" * len(benchmark_name))
            try:
                benchmark_func()
                print(f"  ✓ {benchmark_name} completed successfully")
            except Exception as e:
                print(f"  ✗ ERROR in {benchmark_name}: {str(e)}")

        # Generate summary
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        summary = {
            'total_benchmarks': len(benchmarks),
            'completed_benchmarks': len(self.results),
            'results': self.results
        }

        # Performance summary
        if 'multimodal_performance' in self.results:
            perf = self.results['multimodal_performance']['multimodal_metrics']
            print(f"Multimodal Performance - Avg time: {perf['average_time_seconds']:.4f}s, "
                  f"Throughput: {perf['throughput_samples_per_second']:.2f} samples/s")

        # Memory efficiency summary
        if 'memory_efficiency' in self.results:
            mem = self.results['memory_efficiency']
            print(f"Memory Efficiency - Savings: {mem['memory_savings_percentage']:.2f}%")

        # Accuracy preservation summary
        if 'accuracy_preservation' in self.results:
            acc = self.results['accuracy_preservation']
            print(f"Accuracy - Cosine similarity: {acc['cosine_similarity']:.4f}, "
                  f"Accuracy preserved: {acc['accuracy_preserved']}")

        # Capacity preservation summary
        if 'capacity_preservation' in self.results:
            cap = self.results['capacity_preservation']
            print(f"Capacity - Layers: {cap['num_hidden_layers']}, "
                  f"Heads: {cap['num_attention_heads']}, "
                  f"Preserved: {cap['capacity_preserved']}")

        print(f"\nTotal benchmarks completed: {summary['completed_benchmarks']}/{summary['total_benchmarks']}")

        return summary


def run_standardized_benchmarks():
    """Run the standardized benchmark suite."""
    benchmark_suite = StandardizedBenchmarkSuite()
    results = benchmark_suite.run_comprehensive_benchmark()

    # Save results to file
    import json
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nBenchmark results saved to 'benchmark_results.json'")
    return results


if __name__ == "__main__":
    results = run_standardized_benchmarks()