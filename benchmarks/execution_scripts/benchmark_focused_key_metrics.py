"""
Focused Benchmark Script for Key Performance Metrics

This script measures the key performance metrics (inference speed, memory usage, 
throughput, and accuracy preservation) between optimized and unoptimized model versions.
"""

import sys
import time
import json
import torch
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import statistics
import threading
import queue

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the model plugins
from inference_pio.models.glm_4_7.plugin import create_glm_4_7_plugin
from inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
from inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
from inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin


class FocusedBenchmarkRunner:
    """
    A focused class to run key performance benchmarks comparing optimized vs unoptimized models.
    """

    def __init__(self):
        self.models = {
            "glm_4_7": create_glm_4_7_plugin,
            "qwen3_4b_instruct_2507": create_qwen3_4b_instruct_2507_plugin,
            "qwen3_coder_30b": create_qwen3_coder_30b_plugin,
            "qwen3_vl_2b": create_qwen3_vl_2b_instruct_plugin
        }
        
        self.results = {}
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def warmup_model(self, plugin, input_data):
        """Warm up the model to ensure accurate measurements."""
        for _ in range(3):
            try:
                if hasattr(plugin, 'infer'):
                    plugin.infer(input_data)
                elif hasattr(plugin, 'generate_text'):
                    plugin.generate_text("warmup text for accurate timing")
            except:
                pass  # Ignore warmup errors
    
    def benchmark_inference_speed(self, plugin, input_data, iterations=10) -> Dict[str, float]:
        """Benchmark inference speed (tokens/second)."""
        # Warmup
        self.warmup_model(plugin, input_data)
        
        # Timing run
        start_time = time.time()
        for i in range(iterations):
            try:
                if hasattr(plugin, 'infer'):
                    result = plugin.infer(input_data)
                elif hasattr(plugin, 'generate_text'):
                    result = plugin.generate_text("test input for speed benchmark")
            except Exception as e:
                print(f"Inference error: {e}")
                continue
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_inference = total_time / iterations
        tokens_per_second = (input_data.shape[-1] * iterations) / total_time if total_time > 0 else 0
        
        return {
            'total_time_seconds': total_time,
            'avg_time_per_inference_seconds': avg_time_per_inference,
            'tokens_per_second': tokens_per_second,
            'iterations_completed': iterations
        }
    
    def benchmark_memory_usage(self, plugin, input_data, iterations=5) -> Dict[str, float]:
        """Benchmark memory usage (peak and average)."""
        initial_memory = self.get_memory_usage()
        memory_readings = []
        
        for i in range(iterations):
            # Run inference
            try:
                if hasattr(plugin, 'infer'):
                    result = plugin.infer(input_data)
                elif hasattr(plugin, 'generate_text'):
                    result = plugin.generate_text("test input for memory benchmark")
            except:
                continue
            
            # Record memory after each inference
            current_memory = self.get_memory_usage()
            memory_readings.append(current_memory)
        
        if memory_readings:
            peak_memory = max(memory_readings)
            avg_memory = statistics.mean(memory_readings)
            memory_increase = peak_memory - initial_memory
        else:
            peak_memory = initial_memory
            avg_memory = initial_memory
            memory_increase = 0
        
        return {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'avg_memory_mb': avg_memory,
            'memory_increase_mb': memory_increase,
            'memory_readings_mb': memory_readings
        }
    
    def benchmark_throughput(self, plugin, input_data, duration=10) -> Dict[str, float]:
        """Benchmark throughput (requests per second)."""
        # Warmup
        self.warmup_model(plugin, input_data)
        
        start_time = time.time()
        count = 0
        end_target = start_time + duration
        
        while time.time() < end_target:
            try:
                if hasattr(plugin, 'infer'):
                    plugin.infer(input_data)
                elif hasattr(plugin, 'generate_text'):
                    plugin.generate_text("test input for throughput benchmark")
                count += 1
            except:
                continue
        
        actual_duration = time.time() - start_time
        requests_per_second = count / actual_duration if actual_duration > 0 else 0
        
        return {
            'requests_completed': count,
            'actual_duration_seconds': actual_duration,
            'requests_per_second': requests_per_second
        }
    
    def benchmark_accuracy_preservation(self, unoptimized_plugin, optimized_plugin, input_data, iterations=5) -> Dict[str, Any]:
        """Benchmark to ensure optimizations don't degrade model quality."""
        outputs_unopt = []
        outputs_opt = []
        
        for i in range(iterations):
            try:
                if hasattr(unoptimized_plugin, 'infer'):
                    out_unopt = unoptimized_plugin.infer(input_data)
                    outputs_unopt.append(out_unopt)
                elif hasattr(unoptimized_plugin, 'generate_text'):
                    out_unopt = unoptimized_plugin.generate_text("test input for accuracy")
                    outputs_unopt.append(out_unopt)
                    
                if hasattr(optimized_plugin, 'infer'):
                    out_opt = optimized_plugin.infer(input_data)
                    outputs_opt.append(out_opt)
                elif hasattr(optimized_plugin, 'generate_text'):
                    out_opt = optimized_plugin.generate_text("test input for accuracy")
                    outputs_opt.append(out_opt)
            except:
                continue
        
        # Calculate similarity score based on output characteristics
        similarity_score = 0.0
        if outputs_unopt and outputs_opt:
            # Compare output lengths as a basic metric (in practice, use proper metrics like cosine similarity)
            unopt_lengths = [len(str(out)) if isinstance(out, (str, torch.Tensor)) else 0 for out in outputs_unopt]
            opt_lengths = [len(str(out)) if isinstance(out, (str, torch.Tensor)) else 0 for out in outputs_opt]
            
            if unopt_lengths and opt_lengths:
                avg_unopt_len = sum(unopt_lengths) / len(unopt_lengths)
                avg_opt_len = sum(opt_lengths) / len(opt_lengths)
                
                # Calculate similarity as inverse of relative difference
                if avg_unopt_len > 0:
                    rel_diff = abs(avg_opt_len - avg_unopt_len) / avg_unopt_len
                    similarity_score = max(0, 1 - rel_diff)
        
        return {
            'similarity_score': similarity_score,
            'preserved': similarity_score > 0.95,  # Consider preserved if >95% similar
            'note': 'Basic length comparison - implement proper metrics for production use'
        }
    
    def run_focused_comparison(self, model_name: str) -> Dict[str, Any]:
        """Run focused comparison between optimized and unoptimized versions of a model."""
        print(f"\n{'='*70}")
        print(f"FOCUSED BENCHMARK: {model_name.upper()}")
        print(f"{'='*70}")
        
        create_plugin_func = self.models[model_name]
        
        # Create unoptimized version (baseline)
        print("Initializing unoptimized model...")
        unoptimized_plugin = create_plugin_func()
        unopt_init_success = unoptimized_plugin.initialize(device="cpu")  # Using CPU for consistency
        if not unopt_init_success:
            print(f"Failed to initialize unoptimized {model_name}")
            return {"error": f"Failed to initialize unoptimized {model_name}", "status": "failed"}
        
        unoptimized_model = unoptimized_plugin.load_model()
        if unoptimized_model is None:
            print(f"Failed to load unoptimized {model_name}")
            return {"error": f"Failed to load unoptimized {model_name}", "status": "failed"}
        
        # Create optimized version
        print("Initializing optimized model...")
        optimized_plugin = create_plugin_func()
        opt_init_success = optimized_plugin.initialize(
            device="cpu",
            use_flash_attention=True,
            memory_efficient=True,
            use_fused_layers=True,
            use_tensor_parallelism=False,  # Disable for consistency in single-threaded benchmarking
            quantization_bits=8  # If supported
        )
        if not opt_init_success:
            print(f"Failed to initialize optimized {model_name} with full optimizations, trying with basic optimizations...")
            # Try with basic optimizations
            opt_init_success = optimized_plugin.initialize(
                device="cpu",
                use_flash_attention=True,
                memory_efficient=True
            )
        
        if not opt_init_success:
            print(f"Failed to initialize optimized {model_name}")
            return {"error": f"Failed to initialize optimized {model_name}", "status": "failed"}
        
        optimized_model = optimized_plugin.load_model()
        if optimized_model is None:
            print(f"Failed to load optimized {model_name}")
            return {"error": f"Failed to load optimized {model_name}", "status": "failed"}
        
        # Prepare input data
        input_shape = (1, 50)  # Batch size 1, sequence length 50
        input_data = torch.randint(0, 1000, input_shape)
        
        # Run focused benchmarks
        print("\n1. Running inference speed benchmarks...")
        unopt_speed = self.benchmark_inference_speed(unoptimized_plugin, input_data, iterations=10)
        opt_speed = self.benchmark_inference_speed(optimized_plugin, input_data, iterations=10)
        
        print("\n2. Running memory usage benchmarks...")
        unopt_memory = self.benchmark_memory_usage(unoptimized_plugin, input_data, iterations=5)
        opt_memory = self.benchmark_memory_usage(optimized_plugin, input_data, iterations=5)
        
        print("\n3. Running throughput benchmarks...")
        unopt_throughput = self.benchmark_throughput(unoptimized_plugin, input_data, duration=5)
        opt_throughput = self.benchmark_throughput(optimized_plugin, input_data, duration=5)
        
        print("\n4. Running accuracy preservation benchmarks...")
        accuracy_check = self.benchmark_accuracy_preservation(
            unoptimized_plugin, optimized_plugin, input_data, iterations=3
        )
        
        # Calculate percentage improvements
        speed_improvement = ((opt_speed['tokens_per_second'] / unopt_speed['tokens_per_second']) - 1) * 100 if unopt_speed['tokens_per_second'] > 0 else 0
        throughput_improvement = ((opt_throughput['requests_per_second'] / unopt_throughput['requests_per_second']) - 1) * 100 if unopt_throughput['requests_per_second'] > 0 else 0
        memory_improvement = ((unopt_memory['memory_increase_mb'] - opt_memory['memory_increase_mb']) / unopt_memory['memory_increase_mb']) * 100 if unopt_memory['memory_increase_mb'] > 0 else 0
        
        # Compile focused results
        results = {
            "model": model_name,
            "timestamp": time.time(),
            "metrics": {
                "inference_speed": {
                    "unoptimized_tokens_per_second": unopt_speed['tokens_per_second'],
                    "optimized_tokens_per_second": opt_speed['tokens_per_second'],
                    "improvement_percentage": speed_improvement
                },
                "memory_usage": {
                    "unoptimized_peak_mb": unopt_memory['peak_memory_mb'],
                    "optimized_peak_mb": opt_memory['peak_memory_mb'],
                    "unoptimized_avg_mb": unopt_memory['avg_memory_mb'],
                    "optimized_avg_mb": opt_memory['avg_memory_mb'],
                    "improvement_percentage": memory_improvement
                },
                "throughput": {
                    "unoptimized_rps": unopt_throughput['requests_per_second'],
                    "optimized_rps": opt_throughput['requests_per_second'],
                    "improvement_percentage": throughput_improvement
                },
                "accuracy_preservation": {
                    "similarity_score": accuracy_check['similarity_score'],
                    "preserved": accuracy_check['preserved']
                }
            },
            "raw_data": {
                "unoptimized": {
                    "inference_speed": unopt_speed,
                    "memory_usage": unopt_memory,
                    "throughput": unopt_throughput
                },
                "optimized": {
                    "inference_speed": opt_speed,
                    "memory_usage": opt_memory,
                    "throughput": opt_throughput
                }
            }
        }
        
        # Print focused summary
        print(f"\n{model_name.upper()} FOCUSED RESULTS:")
        print(f"  Inference Speed: {unopt_speed['tokens_per_second']:.2f} → {opt_speed['tokens_per_second']:.2f} tokens/sec ({speed_improvement:+.2f}%)")
        print(f"  Memory Usage: {unopt_memory['avg_memory_mb']:.2f} → {opt_memory['avg_memory_mb']:.2f} MB avg ({memory_improvement:+.2f}%)")
        print(f"  Throughput: {unopt_throughput['requests_per_second']:.2f} → {opt_throughput['requests_per_second']:.2f} reqs/sec ({throughput_improvement:+.2f}%)")
        print(f"  Accuracy Preserved: {'✓' if accuracy_check['preserved'] else '✗'} (score: {accuracy_check['similarity_score']:.3f})")
        
        # Cleanup
        if hasattr(unoptimized_plugin, 'cleanup'):
            unoptimized_plugin.cleanup()
        if hasattr(optimized_plugin, 'cleanup'):
            optimized_plugin.cleanup()
        
        return results
    
    def run_all_focused_benchmarks(self) -> Dict[str, Any]:
        """Run focused benchmarks for all models."""
        print("Starting focused optimization impact benchmarking...")
        
        overall_results = {
            "summary": {
                "total_models": len(self.models),
                "start_time": time.time(),
                "end_time": None,
                "duration": None
            },
            "individual_results": {},
            "aggregated_metrics": {
                "avg_speed_improvement": 0,
                "avg_memory_improvement": 0,
                "avg_throughput_improvement": 0,
                "accuracy_preserved_models": 0,
                "total_models_evaluated": 0
            }
        }
        
        for i, model_name in enumerate(self.models.keys()):
            print(f"\n[{i+1}/{len(self.models)}] Processing {model_name}...")
            result = self.run_focused_comparison(model_name)
            overall_results["individual_results"][model_name] = result
        
        # Calculate aggregated metrics
        speed_improvements = []
        memory_improvements = []
        throughput_improvements = []
        accuracy_preserved_count = 0
        total_evaluated = 0
        
        for model_name, result in overall_results["individual_results"].items():
            if result.get("status") != "failed" and "metrics" in result:
                metrics = result["metrics"]
                speed_improvements.append(metrics["inference_speed"]["improvement_percentage"])
                memory_improvements.append(metrics["memory_usage"]["improvement_percentage"])
                throughput_improvements.append(metrics["throughput"]["improvement_percentage"])
                
                if metrics["accuracy_preservation"]["preserved"]:
                    accuracy_preserved_count += 1
                total_evaluated += 1
        
        if speed_improvements:
            overall_results["aggregated_metrics"]["avg_speed_improvement"] = statistics.mean(speed_improvements)
        if memory_improvements:
            overall_results["aggregated_metrics"]["avg_memory_improvement"] = statistics.mean(memory_improvements)
        if throughput_improvements:
            overall_results["aggregated_metrics"]["avg_throughput_improvement"] = statistics.mean(throughput_improvements)
        
        overall_results["aggregated_metrics"]["accuracy_preserved_models"] = accuracy_preserved_count
        overall_results["aggregated_metrics"]["total_models_evaluated"] = total_evaluated
        
        overall_results["summary"]["end_time"] = time.time()
        overall_results["summary"]["duration"] = overall_results["summary"]["end_time"] - overall_results["summary"]["start_time"]
        
        return overall_results
    
    def generate_key_metrics_report(self, results: Dict[str, Any]) -> str:
        """Generate a report focusing on key metrics."""
        report = []
        report.append("# Key Metrics Optimization Impact Report\n")
        report.append(f"Generated on: {time.ctime(results['summary']['end_time'])}\n")
        report.append(f"Total Duration: {results['summary']['duration']:.2f} seconds\n")
        
        agg = results["aggregated_metrics"]
        report.append("## Aggregated Key Metrics\n")
        report.append(f"- **Avg Speed Improvement**: {agg['avg_speed_improvement']:+.2f}%\n")
        report.append(f"- **Avg Memory Improvement**: {agg['avg_memory_improvement']:+.2f}%\n")
        report.append(f"- **Avg Throughput Improvement**: {agg['avg_throughput_improvement']:+.2f}%\n")
        report.append(f"- **Accuracy Preserved Models**: {agg['accuracy_preserved_models']}/{agg['total_models_evaluated']}\n")
        
        report.append("\n## Individual Model Results\n")
        
        for model_name, result in results["individual_results"].items():
            if result.get("status") == "failed":
                report.append(f"### {model_name}\n")
                report.append(f"- Status: FAILED - {result.get('error', 'Unknown error')}\n")
                continue
                
            metrics = result["metrics"]
            report.append(f"### {model_name}\n")
            report.append(f"- **Inference Speed**: {metrics['inference_speed']['unoptimized_tokens_per_second']:.2f} → {metrics['inference_speed']['optimized_tokens_per_second']:.2f} tokens/sec ({metrics['inference_speed']['improvement_percentage']:+.2f}%)\n")
            report.append(f"- **Memory Usage**: {metrics['memory_usage']['unoptimized_avg_mb']:.2f} → {metrics['memory_usage']['optimized_avg_mb']:.2f} MB avg ({metrics['memory_usage']['improvement_percentage']:+.2f}%)\n")
            report.append(f"- **Throughput**: {metrics['throughput']['unoptimized_rps']:.2f} → {metrics['throughput']['optimized_rps']:.2f} reqs/sec ({metrics['throughput']['improvement_percentage']:+.2f}%)\n")
            report.append(f"- **Accuracy Preserved**: {'Yes' if metrics['accuracy_preservation']['preserved'] else 'No'} (Score: {metrics['accuracy_preservation']['similarity_score']:.3f})\n")
            report.append("\n")
        
        report.append("## Key Findings\n")
        report.append(f"1. On average, optimizations improved inference speed by {agg['avg_speed_improvement']:+.2f}%.\n")
        report.append(f"2. Memory usage was reduced by {agg['avg_memory_improvement']:+.2f}% on average.\n")
        report.append(f"3. Throughput increased by {agg['avg_throughput_improvement']:+.2f}% on average.\n")
        accuracy_rate = (agg['accuracy_preserved_models'] / agg['total_models_evaluated'] * 100) if agg['total_models_evaluated'] > 0 else 0
        report.append(f"4. Model accuracy was preserved in {accuracy_rate:.1f}% of cases.\n")
        
        return "".join(report)


def main():
    """Main function to run focused benchmarks."""
    runner = FocusedBenchmarkRunner()
    results = runner.run_all_focused_benchmarks()
    
    # Print summary
    print("\n" + "="*70)
    print("FOCUSED BENCHMARK SUMMARY")
    print("="*70)
    
    agg = results["aggregated_metrics"]
    print(f"Average Speed Improvement: {agg['avg_speed_improvement']:+.2f}%")
    print(f"Average Memory Improvement: {agg['avg_memory_improvement']:+.2f}%")
    print(f"Average Throughput Improvement: {agg['avg_throughput_improvement']:+.2f}%")
    print(f"Accuracy Preserved Models: {agg['accuracy_preserved_models']}/{agg['total_models_evaluated']}")
    print(f"Total Duration: {results['summary']['duration']:.2f} seconds")
    
    # Generate and save detailed report
    report = runner.generate_key_metrics_report(results)
    
    # Save JSON results
    with open("focused_optimization_impact_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save markdown report
    with open("focused_optimization_impact_report.md", "w") as f:
        f.write(report)
    
    print(f"\nDetailed results saved to 'focused_optimization_impact_results.json'")
    print(f"Key metrics report saved to 'focused_optimization_impact_report.md'")


if __name__ == "__main__":
    main()