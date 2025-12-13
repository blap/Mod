"""
Standardized Benchmark Suite for Qwen3-VL Model Components

This module provides a comprehensive, standardized benchmark suite for all Qwen3-VL model components
following best practices for performance measurement and ensuring consistency across all benchmarks.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import psutil
import GPUtil
import gc
import sys
from pathlib import Path

# Add the project root to the path to import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class BenchmarkUtils:
    """Utility functions for standardized benchmarking."""

    @staticmethod
    def benchmark_model_inference(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        num_runs: int = 10,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Standardized benchmark for model inference performance.

        Args:
            model: The model to benchmark
            input_ids: Input token IDs
            pixel_values: Optional pixel values for vision input
            num_runs: Number of runs for averaging
            device: Device to run benchmark on
            **kwargs: Additional arguments for model.forward()

        Returns:
            Dictionary with performance metrics
        """
        if device is None:
            device = next(model.parameters()).device

        model = model.to(device)
        model.eval()

        # Move inputs to device
        input_ids = input_ids.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        # Warm up
        for _ in range(3):
            with torch.no_grad():
                if pixel_values is not None:
                    _ = model(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                else:
                    _ = model(input_ids=input_ids, **kwargs)

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Benchmark
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                if pixel_values is not None:
                    _ = model(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                else:
                    _ = model(input_ids=input_ids, **kwargs)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

        # Memory usage
        memory_allocated = torch.cuda.memory_allocated(device) if torch.cuda.is_available() else 0
        memory_reserved = torch.cuda.memory_reserved(device) if torch.cuda.is_available() else 0

        return {
            'avg_time_seconds': avg_time,
            'std_time_seconds': std_time,
            'times_list': times,
            'memory_allocated_bytes': memory_allocated,
            'memory_reserved_bytes': memory_reserved,
            'device': str(device),
            'num_runs': num_runs
        }

    @staticmethod
    def benchmark_generation(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: int = 20,
        num_runs: int = 5,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Standardized benchmark for model generation performance.

        Args:
            model: The model to benchmark
            input_ids: Input token IDs
            pixel_values: Optional pixel values for vision input
            max_new_tokens: Maximum number of new tokens to generate
            num_runs: Number of runs for averaging
            device: Device to run benchmark on
            **kwargs: Additional arguments for model.generate()

        Returns:
            Dictionary with generation performance metrics
        """
        if device is None:
            device = next(model.parameters()).device

        model = model.to(device)
        model.eval()

        # Move inputs to device
        input_ids = input_ids.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        # Warm up
        for _ in range(2):
            with torch.no_grad():
                if pixel_values is not None:
                    _ = model.generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        max_new_tokens=5,
                        **kwargs
                    )
                else:
                    _ = model.generate(input_ids=input_ids, max_new_tokens=5, **kwargs)

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Benchmark generation
        times = []
        generated_tokens_counts = []

        for _ in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()

            with torch.no_grad():
                if pixel_values is not None:
                    generated = model.generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        max_new_tokens=max_new_tokens,
                        **kwargs
                    )
                else:
                    generated = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, **kwargs)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()

            times.append(end_time - start_time)
            generated_tokens_counts.append(generated.shape[1] - input_ids.shape[1])  # Count newly generated tokens

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        avg_gen_tokens = sum(generated_tokens_counts) / len(generated_tokens_counts)

        return {
            'avg_generation_time_seconds': avg_time,
            'std_generation_time_seconds': std_time,
            'avg_generated_tokens': avg_gen_tokens,
            'generation_times_list': times,
            'generated_tokens_counts': generated_tokens_counts,
            'device': str(device),
            'num_runs': num_runs,
            'max_new_tokens': max_new_tokens
        }

    @staticmethod
    def profile_memory_usage(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Profile memory usage of the model during inference.

        Args:
            model: The model to profile
            input_ids: Input token IDs
            pixel_values: Optional pixel values for vision input
            device: Device to run profiling on

        Returns:
            Dictionary with memory usage metrics
        """
        if device is None:
            device = next(model.parameters()).device

        model = model.to(device)
        model.eval()

        # Move inputs to device
        input_ids = input_ids.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)

        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0

        # Run model
        with torch.no_grad():
            if pixel_values is not None:
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
            else:
                _ = model(input_ids=input_ids)

        # Get final memory stats
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
        else:
            final_memory = 0
            peak_memory = 0

        return {
            'initial_memory_bytes': initial_memory,
            'final_memory_bytes': final_memory,
            'peak_memory_bytes': peak_memory,
            'memory_increase_bytes': final_memory - initial_memory if initial_memory > 0 else final_memory,
            'device': str(device)
        }


class StandardizedBenchmarkSuite:
    """Comprehensive benchmark suite for Qwen3-VL model components."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.benchmark_utils = BenchmarkUtils()

    def benchmark_model_performance(self, config_kwargs: Optional[Dict] = None) -> Dict[str, Any]:
        """Benchmark model performance with standardized inputs."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        # Create configuration
        config = Qwen3VLConfig(**(config_kwargs or {}))
        config.hidden_size = 256  # Reduced for benchmarking
        config.num_attention_heads = 8  # Reduced for benchmarking
        config.num_hidden_layers = 4  # Reduced for benchmarking
        config.vision_hidden_size = 256  # Reduced for benchmarking
        config.vision_num_attention_heads = 8  # Reduced for benchmarking
        config.vision_num_hidden_layers = 4  # Reduced for benchmarking
        config.vocab_size = 1000  # Reduced for benchmarking

        # Create model
        model = Qwen3VLForConditionalGeneration(config)
        model = model.to(self.device)
        model.eval()

        # Create test inputs
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        # Benchmark inference
        inference_results = self.benchmark_utils.benchmark_model_inference(
            model, input_ids, pixel_values, num_runs=5, device=self.device
        )

        # Benchmark generation
        generation_results = self.benchmark_utils.benchmark_generation(
            model, input_ids, pixel_values, max_new_tokens=10, num_runs=3, device=self.device
        )

        # Profile memory usage
        memory_results = self.benchmark_utils.profile_memory_usage(
            model, input_ids, pixel_values, device=self.device
        )

        performance_results = {
            'inference': inference_results,
            'generation': generation_results,
            'memory': memory_results,
            'model_config': {
                'hidden_size': config.hidden_size,
                'num_attention_heads': config.num_attention_heads,
                'num_hidden_layers': config.num_hidden_layers,
                'vocab_size': config.vocab_size
            }
        }

        self.results['model_performance'] = performance_results
        return performance_results

    def benchmark_multimodal_integration(self) -> Dict[str, Any]:
        """Benchmark multimodal integration performance."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        # Create configuration
        config = Qwen3VLConfig()
        config.hidden_size = 128  # Smaller for testing
        config.num_attention_heads = 4  # Smaller for testing
        config.num_hidden_layers = 2  # Smaller for testing
        config.vision_hidden_size = 128  # Smaller for testing
        config.vision_num_attention_heads = 4  # Smaller for testing
        config.vision_num_hidden_layers = 2  # Smaller for testing
        config.vocab_size = 500  # Smaller for testing

        # Create model
        model = Qwen3VLForConditionalGeneration(config)
        model = model.to(self.device)
        model.eval()

        # Create test inputs
        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        # Benchmark multimodal processing
        multimodal_results = self.benchmark_utils.benchmark_model_inference(
            model, input_ids, pixel_values, num_runs=3, device=self.device
        )

        # Benchmark text-only processing
        text_only_results = self.benchmark_utils.benchmark_model_inference(
            model, input_ids, pixel_values=None, num_runs=3, device=self.device
        )

        # Benchmark image-only processing
        image_only_results = self.benchmark_utils.benchmark_model_inference(
            model, torch.randint(0, config.vocab_size, (batch_size, 1)),
            pixel_values=pixel_values, num_runs=3, device=self.device
        )

        integration_results = {
            'multimodal': multimodal_results,
            'text_only': text_only_results,
            'image_only': image_only_results,
            'efficiency_ratios': {
                'multimodal_to_text_ratio': multimodal_results['avg_time_seconds'] / text_only_results['avg_time_seconds'] if text_only_results['avg_time_seconds'] > 0 else 0,
                'multimodal_to_image_ratio': multimodal_results['avg_time_seconds'] / image_only_results['avg_time_seconds'] if image_only_results['avg_time_seconds'] > 0 else 0
            }
        }

        self.results['multimodal_integration'] = integration_results
        return integration_results

    def benchmark_memory_efficiency(self) -> Dict[str, Any]:
        """Benchmark memory efficiency improvements."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        # Create configurations for different optimization levels
        baseline_config = Qwen3VLConfig()
        baseline_config.hidden_size = 128
        baseline_config.num_attention_heads = 4
        baseline_config.num_hidden_layers = 2
        baseline_config.vision_hidden_size = 128
        baseline_config.vision_num_attention_heads = 4
        baseline_config.vision_num_hidden_layers = 2
        baseline_config.vocab_size = 500
        baseline_config.use_gradient_checkpointing = False  # Disable optimizations for baseline

        optimized_config = Qwen3VLConfig()
        optimized_config.hidden_size = 128
        optimized_config.num_attention_heads = 4
        optimized_config.num_hidden_layers = 2
        optimized_config.vision_hidden_size = 128
        optimized_config.vision_num_attention_heads = 4
        optimized_config.vision_num_hidden_layers = 2
        optimized_config.vocab_size = 500
        # Enable optimizations
        optimized_config.use_gradient_checkpointing = True
        optimized_config.use_sparsity = True
        optimized_config.sparsity_ratio = 0.5

        # Create models
        baseline_model = Qwen3VLForConditionalGeneration(baseline_config)
        optimized_model = Qwen3VLForConditionalGeneration(optimized_config)

        # Copy weights to ensure fair comparison
        optimized_model.load_state_dict(baseline_model.state_dict(), strict=False)

        # Move models to device
        baseline_model = baseline_model.to(self.device)
        optimized_model = optimized_model.to(self.device)
        baseline_model.eval()
        optimized_model.eval()

        # Create test inputs
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)

        # Profile memory usage for both models
        baseline_memory = self.benchmark_utils.profile_memory_usage(
            baseline_model, input_ids, pixel_values, device=self.device
        )

        optimized_memory = self.benchmark_utils.profile_memory_usage(
            optimized_model, input_ids, pixel_values, device=self.device
        )

        # Benchmark performance for both models
        baseline_perf = self.benchmark_utils.benchmark_model_inference(
            baseline_model, input_ids, pixel_values, num_runs=3, device=self.device
        )

        optimized_perf = self.benchmark_utils.benchmark_model_inference(
            optimized_model, input_ids, pixel_values, num_runs=3, device=self.device
        )

        memory_results = {
            'baseline_memory': baseline_memory,
            'optimized_memory': optimized_memory,
            'baseline_performance': baseline_perf,
            'optimized_performance': optimized_perf,
            'memory_savings_bytes': baseline_memory['peak_memory_bytes'] - optimized_memory['peak_memory_bytes'],
            'memory_savings_percentage': (
                (baseline_memory['peak_memory_bytes'] - optimized_memory['peak_memory_bytes']) /
                baseline_memory['peak_memory_bytes'] * 100
                if baseline_memory['peak_memory_bytes'] > 0 else 0
            ),
            'performance_impact_seconds': optimized_perf['avg_time_seconds'] - baseline_perf['avg_time_seconds']
        }

        self.results['memory_efficiency'] = memory_results
        return memory_results

    def benchmark_capacity_preservation(self) -> Dict[str, Any]:
        """Benchmark that model capacity is preserved."""
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        # Create full capacity config
        config = Qwen3VLConfig()

        # Verify config parameters
        config_params = {
            'num_hidden_layers': config.num_hidden_layers,
            'num_attention_heads': config.num_attention_heads,
            'vision_num_hidden_layers': config.vision_num_hidden_layers,
            'vision_num_attention_heads': config.vision_num_attention_heads,
            'hidden_size': config.hidden_size,
            'vocab_size': config.vocab_size
        }

        # Create model
        model = Qwen3VLForConditionalGeneration(config)

        # Verify model architecture
        model_architecture = {
            'language_model_layers': len(model.language_model.layers) if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers') else 0,
            'vision_model_layers': len(model.vision_tower.layers) if hasattr(model, 'vision_tower') and hasattr(model.vision_tower, 'layers') else 0,
            'language_model_attention_heads': model.config.num_attention_heads,
            'vision_model_attention_heads': model.config.vision_num_attention_heads
        }

        # Verify capacity preservation
        capacity_preserved = (
            config.num_hidden_layers == 32 and
            config.num_attention_heads == 32 and
            model_architecture['language_model_layers'] == 32 and
            model_architecture['vision_model_layers'] == 24
        )

        capacity_results = {
            'config_params': config_params,
            'model_architecture': model_architecture,
            'capacity_preserved': capacity_preserved,
            'expected_language_layers': 32,
            'expected_language_attention_heads': 32,
            'expected_vision_layers': 24,
            'expected_vision_attention_heads': 16
        }

        self.results['capacity_preservation'] = capacity_results
        return capacity_results

    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        print("="*80)
        print("RUNNING COMPREHENSIVE BENCHMARK SUITE FOR QWEN3-VL MODEL")
        print("="*80)

        # Run all benchmarks
        benchmarks = [
            ("Model Performance", self.benchmark_model_performance),
            ("Multimodal Integration", self.benchmark_multimodal_integration),
            ("Memory Efficiency", self.benchmark_memory_efficiency),
            ("Capacity Preservation", self.benchmark_capacity_preservation)
        ]

        for benchmark_name, benchmark_func in benchmarks:
            print(f"\n{benchmark_name}:")
            print("-" * len(benchmark_name))
            try:
                benchmark_func()
                print(f"  OK {benchmark_name} completed successfully")
            except Exception as e:
                print(f"  X ERROR in {benchmark_name}: {str(e)}")
                import traceback
                traceback.print_exc()

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
        if 'model_performance' in self.results:
            perf = self.results['model_performance']['inference']
            print(f"Performance - Average inference time: {perf['avg_time_seconds']:.4f}s")

        # Memory efficiency summary
        if 'memory_efficiency' in self.results:
            mem = self.results['memory_efficiency']
            print(f"Memory Efficiency - Savings: {mem['memory_savings_percentage']:.2f}%")
            print(f"Performance Impact - Change: {mem['performance_impact_seconds']:.4f}s")

        # Capacity preservation summary
        if 'capacity_preservation' in self.results:
            cap = self.results['capacity_preservation']
            print(f"Capacity - Preserved: {cap['capacity_preserved']}")
            print(f"  Language layers: {cap['model_architecture']['language_model_layers']}/32")
            print(f"  Language attention heads: {cap['model_architecture']['language_model_attention_heads']}/32")
            print(f"  Vision layers: {cap['model_architecture']['vision_model_layers']}/24")

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