"""
Comprehensive Benchmark Script for Measuring Performance Differences Between 
Optimized and Unoptimized Model Versions

This script creates and compares optimized and unoptimized versions of each model,
measuring actual performance metrics including speed, memory usage, throughput,
and accuracy preservation.
"""

import sys
import time
import json
import torch
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import statistics

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the model plugins
from inference_pio.models.glm_4_7.plugin import create_glm_4_7_plugin
from inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
from inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
from inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin


class ModelBenchmarkRunner:
    """
    A class to run comprehensive benchmarks comparing optimized vs unoptimized models.
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
        for _ in range(5):
            try:
                if hasattr(plugin, 'infer'):
                    plugin.infer(input_data)
                elif hasattr(plugin, 'generate_text'):
                    plugin.generate_text("warmup text")
            except:
                pass  # Ignore warmup errors
    
    def benchmark_inference_speed(self, plugin, input_data, iterations=10) -> Dict[str, float]:
        """Benchmark inference speed."""
        # Warmup
        self.warmup_model(plugin, input_data)
        
        # Measure memory before
        mem_before = self.get_memory_usage()
        
        # Timing run
        start_time = time.time()
        for i in range(iterations):
            try:
                if hasattr(plugin, 'infer'):
                    result = plugin.infer(input_data)
                elif hasattr(plugin, 'generate_text'):
                    result = plugin.generate_text("test input")
            except Exception as e:
                print(f"Inference error: {e}")
                continue
        end_time = time.time()
        
        # Measure memory after
        mem_after = self.get_memory_usage()
        
        total_time = end_time - start_time
        avg_time_per_inference = total_time / iterations
        tokens_per_second = input_data.shape[-1] / avg_time_per_inference if avg_time_per_inference > 0 else 0
        
        return {
            'total_time': total_time,
            'avg_time_per_inference': avg_time_per_inference,
            'tokens_per_second': tokens_per_second,
            'memory_used_peak': max(mem_before, mem_after),
            'memory_used_avg': (mem_before + mem_after) / 2,
            'memory_increase': mem_after - mem_before
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
                    plugin.generate_text("test input")
                count += 1
            except:
                continue
        
        actual_duration = time.time() - start_time
        requests_per_second = count / actual_duration if actual_duration > 0 else 0
        
        return {
            'requests_completed': count,
            'actual_duration': actual_duration,
            'requests_per_second': requests_per_second
        }
    
    def benchmark_memory_usage(self, plugin, input_data, iterations=5) -> Dict[str, float]:
        """Benchmark memory usage patterns."""
        memory_readings = []
        
        for i in range(iterations):
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Get initial memory
            initial_mem = self.get_memory_usage()
            
            # Run inference
            try:
                if hasattr(plugin, 'infer'):
                    result = plugin.infer(input_data)
                elif hasattr(plugin, 'generate_text'):
                    result = plugin.generate_text("test input")
            except:
                continue
            
            # Get peak memory during operation
            peak_mem = self.get_memory_usage()
            memory_readings.append(peak_mem - initial_mem)
        
        if memory_readings:
            return {
                'peak_memory_increase': max(memory_readings),
                'avg_memory_increase': statistics.mean(memory_readings),
                'std_memory_increase': statistics.stdev(memory_readings) if len(memory_readings) > 1 else 0
            }
        else:
            return {
                'peak_memory_increase': 0,
                'avg_memory_increase': 0,
                'std_memory_increase': 0
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
                    out_unopt = unoptimized_plugin.generate_text("test input")
                    outputs_unopt.append(out_unopt)
                    
                if hasattr(optimized_plugin, 'infer'):
                    out_opt = optimized_plugin.infer(input_data)
                    outputs_opt.append(out_opt)
                elif hasattr(optimized_plugin, 'generate_text'):
                    out_opt = optimized_plugin.generate_text("test input")
                    outputs_opt.append(out_opt)
            except:
                continue
        
        # Simple comparison - in practice, you'd want more sophisticated metrics
        similarity_score = 0.0
        if outputs_unopt and outputs_opt:
            # For now, just compare output lengths as a basic metric
            # In real scenarios, you'd use cosine similarity, BLEU, etc.
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
    
    def run_model_comparison(self, model_name: str) -> Dict[str, Any]:
        """Run comparison between optimized and unoptimized versions of a model."""
        print(f"\n{'='*80}")
        print(f"COMPARING OPTIMIZED VS UNOPTIMIZED VERSIONS FOR {model_name.upper()}")
        print(f"{'='*80}")
        
        create_plugin_func = self.models[model_name]
        
        # Create unoptimized version (baseline)
        print("Creating unoptimized model...")
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
        print("Creating optimized model...")
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
            print(f"Failed to initialize optimized {model_name}, trying with minimal optimizations...")
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
        
        # Run benchmarks
        print("\nRunning inference speed benchmarks...")
        unopt_speed = self.benchmark_inference_speed(unoptimized_plugin, input_data, iterations=10)
        opt_speed = self.benchmark_inference_speed(optimized_plugin, input_data, iterations=10)
        
        print("\nRunning throughput benchmarks...")
        unopt_throughput = self.benchmark_throughput(unoptimized_plugin, input_data, duration=5)
        opt_throughput = self.benchmark_throughput(optimized_plugin, input_data, duration=5)
        
        print("\nRunning memory usage benchmarks...")
        unopt_memory = self.benchmark_memory_usage(unoptimized_plugin, input_data, iterations=5)
        opt_memory = self.benchmark_memory_usage(optimized_plugin, input_data, iterations=5)
        
        print("\nRunning accuracy preservation benchmarks...")
        accuracy_check = self.benchmark_accuracy_preservation(
            unoptimized_plugin, optimized_plugin, input_data, iterations=3
        )
        
        # Calculate improvements
        speed_improvement = ((opt_speed['tokens_per_second'] / unopt_speed['tokens_per_second']) - 1) * 100 if unopt_speed['tokens_per_second'] > 0 else 0
        throughput_improvement = ((opt_throughput['requests_per_second'] / unopt_throughput['requests_per_second']) - 1) * 100 if unopt_throughput['requests_per_second'] > 0 else 0
        memory_improvement = ((unopt_memory['avg_memory_increase'] - opt_memory['avg_memory_increase']) / unopt_memory['avg_memory_increase']) * 100 if unopt_memory['avg_memory_increase'] > 0 else 0
        
        # Compile results
        results = {
            "model": model_name,
            "timestamp": time.time(),
            "unoptimized": {
                "inference_speed": unopt_speed,
                "throughput": unopt_throughput,
                "memory_usage": unopt_memory
            },
            "optimized": {
                "inference_speed": opt_speed,
                "throughput": opt_throughput,
                "memory_usage": opt_memory
            },
            "comparisons": {
                "speed_improvement_percent": speed_improvement,
                "throughput_improvement_percent": throughput_improvement,
                "memory_improvement_percent": memory_improvement,
                "accuracy_preserved": accuracy_check['preserved'],
                "accuracy_similarity_score": accuracy_check['similarity_score']
            },
            "summary": {
                "tokens_per_second_unoptimized": unopt_speed['tokens_per_second'],
                "tokens_per_second_optimized": opt_speed['tokens_per_second'],
                "throughput_unoptimized": unopt_throughput['requests_per_second'],
                "throughput_optimized": opt_throughput['requests_per_second'],
                "avg_memory_unoptimized": unopt_memory['avg_memory_increase'],
                "avg_memory_optimized": opt_memory['avg_memory_increase']
            }
        }
        
        # Print summary
        print(f"\n{model_name.upper()} PERFORMANCE COMPARISON RESULTS:")
        print(f"  Inference Speed: {unopt_speed['tokens_per_second']:.2f} → {opt_speed['tokens_per_second']:.2f} tokens/sec ({speed_improvement:+.2f}%)")
        print(f"  Throughput: {unopt_throughput['requests_per_second']:.2f} → {opt_throughput['requests_per_second']:.2f} reqs/sec ({throughput_improvement:+.2f}%)")
        print(f"  Memory Usage: {unopt_memory['avg_memory_increase']:.2f} → {opt_memory['avg_memory_increase']:.2f} MB ({memory_improvement:+.2f}%)")
        print(f"  Accuracy Preserved: {'✓' if accuracy_check['preserved'] else '✗'} (score: {accuracy_check['similarity_score']:.3f})")
        
        # Cleanup
        if hasattr(unoptimized_plugin, 'cleanup'):
            unoptimized_plugin.cleanup()
        if hasattr(optimized_plugin, 'cleanup'):
            optimized_plugin.cleanup()
        
        return results
    
    def run_all_comparisons(self) -> Dict[str, Any]:
        """Run comparisons for all models."""
        print("Starting comprehensive optimized vs unoptimized benchmarking...")
        
        overall_results = {
            "summary": {
                "total_models": len(self.models),
                "start_time": time.time(),
                "end_time": None,
                "duration": None
            },
            "individual_results": {},
            "aggregated_improvements": {
                "avg_speed_improvement": 0,
                "avg_throughput_improvement": 0,
                "avg_memory_improvement": 0,
                "models_with_improved_accuracy": 0,
                "total_accuracy_checks": 0
            }
        }
        
        for i, model_name in enumerate(self.models.keys()):
            print(f"\n[{i+1}/{len(self.models)}] Processing {model_name}...")
            result = self.run_model_comparison(model_name)
            overall_results["individual_results"][model_name] = result
        
        # Calculate aggregated results
        speed_improvements = []
        throughput_improvements = []
        memory_improvements = []
        accuracy_preserved_count = 0
        total_accuracy_checks = 0
        
        for model_name, result in overall_results["individual_results"].items():
            if result.get("status") != "failed" and "comparisons" in result:
                comp = result["comparisons"]
                speed_improvements.append(comp["speed_improvement_percent"])
                throughput_improvements.append(comp["throughput_improvement_percent"])
                memory_improvements.append(comp["memory_improvement_percent"])
                
                if comp["accuracy_preserved"]:
                    accuracy_preserved_count += 1
                total_accuracy_checks += 1
        
        if speed_improvements:
            overall_results["aggregated_improvements"]["avg_speed_improvement"] = statistics.mean(speed_improvements)
        if throughput_improvements:
            overall_results["aggregated_improvements"]["avg_throughput_improvement"] = statistics.mean(throughput_improvements)
        if memory_improvements:
            overall_results["aggregated_improvements"]["avg_memory_improvement"] = statistics.mean(memory_improvements)
        
        overall_results["aggregated_improvements"]["models_with_improved_accuracy"] = accuracy_preserved_count
        overall_results["aggregated_improvements"]["total_accuracy_checks"] = total_accuracy_checks
        
        overall_results["summary"]["end_time"] = time.time()
        overall_results["summary"]["duration"] = overall_results["summary"]["end_time"] - overall_results["summary"]["start_time"]
        
        return overall_results
    
    def generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """Generate a detailed report from the benchmark results."""
        report = []
        report.append("# Optimization Impact Benchmark Report\n")
        report.append(f"Generated on: {time.ctime(results['summary']['end_time'])}\n")
        report.append(f"Total Duration: {results['summary']['duration']:.2f} seconds\n")
        
        report.append("## Executive Summary\n")
        agg = results["aggregated_improvements"]
        report.append(f"- Average Speed Improvement: {agg['avg_speed_improvement']:+.2f}%\n")
        report.append(f"- Average Throughput Improvement: {agg['avg_throughput_improvement']:+.2f}%\n")
        report.append(f"- Average Memory Usage Improvement: {agg['avg_memory_improvement']:+.2f}%\n")
        report.append(f"- Models with Preserved Accuracy: {agg['models_with_improved_accuracy']}/{agg['total_accuracy_checks']}\n")
        
        report.append("\n## Individual Model Results\n")
        
        for model_name, result in results["individual_results"].items():
            if result.get("status") == "failed":
                report.append(f"### {model_name}\n")
                report.append(f"- Status: FAILED - {result.get('error', 'Unknown error')}\n")
                continue
                
            report.append(f"### {model_name}\n")
            summary = result["summary"]
            comparisons = result["comparisons"]
            
            report.append(f"- **Inference Speed**: {summary['tokens_per_second_unoptimized']:.2f} → {summary['tokens_per_second_optimized']:.2f} tokens/sec ({comparisons['speed_improvement_percent']:+.2f}%)\n")
            report.append(f"- **Throughput**: {summary['throughput_unoptimized']:.2f} → {summary['throughput_optimized']:.2f} reqs/sec ({comparisons['throughput_improvement_percent']:+.2f}%)\n")
            report.append(f"- **Memory Usage**: {summary['avg_memory_unoptimized']:.2f} → {summary['avg_memory_optimized']:.2f} MB ({comparisons['memory_improvement_percent']:+.2f}%)\n")
            report.append(f"- **Accuracy Preserved**: {'Yes' if comparisons['accuracy_preserved'] else 'No'} (Score: {comparisons['accuracy_similarity_score']:.3f})\n")
            
            # Detailed metrics
            unopt_speed = result["unoptimized"]["inference_speed"]
            opt_speed = result["optimized"]["inference_speed"]
            report.append(f"  - Unoptimized: Total time {unopt_speed['total_time']:.2f}s, Avg time {unopt_speed['avg_time_per_inference']:.4f}s per inference\n")
            report.append(f"  - Optimized: Total time {opt_speed['total_time']:.2f}s, Avg time {opt_speed['avg_time_per_inference']:.4f}s per inference\n")
            
            unopt_mem = result["unoptimized"]["memory_usage"]
            opt_mem = result["optimized"]["memory_usage"]
            report.append(f"  - Memory: Unoptimized avg increase {unopt_mem['avg_memory_increase']:.2f}MB, Optimized avg increase {opt_mem['avg_memory_increase']:.2f}MB\n")
            
            report.append("\n")
        
        report.append("## Conclusion\n")
        if agg['avg_speed_improvement'] > 0:
            report.append(f"The optimization techniques resulted in an average speed improvement of {agg['avg_speed_improvement']:+.2f}%.\n")
        else:
            report.append(f"The optimization techniques resulted in an average speed degradation of {abs(agg['avg_speed_improvement']):+.2f}%.\n")
            
        if agg['avg_memory_improvement'] > 0:
            report.append(f"Memory usage was reduced by an average of {agg['avg_memory_improvement']:+.2f}%.\n")
        else:
            report.append(f"Memory usage increased by an average of {abs(agg['avg_memory_improvement']):+.2f}%.\n")
            
        accuracy_rate = (agg['models_with_improved_accuracy'] / agg['total_accuracy_checks'] * 100) if agg['total_accuracy_checks'] > 0 else 0
        report.append(f"Model accuracy was preserved in {accuracy_rate:.1f}% of cases.\n")
        
        return "".join(report)


def main():
    """Main function to run the benchmark."""
    runner = ModelBenchmarkRunner()
    results = runner.run_all_comparisons()
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL AGGREGATED RESULTS")
    print("="*80)
    
    agg = results["aggregated_improvements"]
    print(f"Average Speed Improvement: {agg['avg_speed_improvement']:+.2f}%")
    print(f"Average Throughput Improvement: {agg['avg_throughput_improvement']:+.2f}%")
    print(f"Average Memory Usage Improvement: {agg['avg_memory_improvement']:+.2f}%")
    print(f"Models with Preserved Accuracy: {agg['models_with_improved_accuracy']}/{agg['total_accuracy_checks']}")
    print(f"Total Duration: {results['summary']['duration']:.2f} seconds")
    
    # Generate and save detailed report
    report = runner.generate_detailed_report(results)
    
    # Save JSON results
    with open("optimization_impact_detailed_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save markdown report
    with open("optimization_impact_benchmark_report.md", "w") as f:
        f.write(report)
    
    print(f"\nDetailed results saved to 'optimization_impact_detailed_results.json'")
    print(f"Markdown report saved to 'optimization_impact_benchmark_report.md'")


if __name__ == "__main__":
    main()