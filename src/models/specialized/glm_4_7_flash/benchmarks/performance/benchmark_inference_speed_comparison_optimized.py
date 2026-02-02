"""
Comprehensive Benchmark for Inference Speed Comparison - GLM-4.7

This module benchmarks the inference speed comparison between the original
and modified versions of the GLM-4.7 model with all optimizations enabled.
"""

import csv
import gc
import json
import os
import threading
import time
from datetime import datetime
from statistics import mean, stdev

import GPUtil
import psutil
import schedule
import torch


class GLM47InferenceSpeedComparisonBenchmark:
    """Benchmark class for comparing inference speed between original and modified GLM-4.7 models."""

    def __init__(self):
        self.results = {
            "model_name": "GLM-4.7",
            "timestamp": datetime.now().isoformat(),
            "original_results": {},
            "modified_results": {},
            "comparison": {},
        }
        self.monitoring_active = False

        # Create results directory if it doesn't exist
        self.results_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "benchmark_results"
        )
        os.makedirs(self.results_dir, exist_ok=True)

    def load_original_model(self):
        """Load the original version of the model from current location."""
        from inference_pio.models.glm_4_7.plugin import create_glm_4_7_plugin

        plugin = create_glm_4_7_plugin()

        # Initialize with CPU to ensure consistency
        success = plugin.initialize(
            device="cpu",
            enable_memory_management=False,  # Disable for baseline
            enable_tensor_paging=False,
            enable_adaptive_batching=False,
            enable_kernel_fusion=False,
            enable_tensor_compression=False,
            enable_disk_offloading=False,
            enable_activation_offloading=False,
            enable_model_surgery=False,
            enable_pipeline=False,
        )
        if not success:
            raise RuntimeError("Failed to initialize original model")

        model = plugin.load_model()
        if model is None:
            raise RuntimeError("Failed to load original model")

        return plugin

    def load_modified_model(self):
        """Load the modified version of the model from main directory with optimizations."""
        from inference_pio.models.glm_4_7.plugin import create_glm_4_7_plugin

        plugin = create_glm_4_7_plugin()

        # Initialize with CPU and all optimizations enabled
        success = plugin.initialize(
            device="cpu",
            enable_memory_management=True,
            enable_tensor_paging=True,
            enable_adaptive_batching=True,
            enable_kernel_fusion=True,
            enable_tensor_compression=True,
            enable_disk_offloading=True,
            enable_activation_offloading=True,
            enable_model_surgery=True,
            enable_pipeline=True,
            torch_compile_mode="reduce-overhead",  # Enable torch.compile for optimizations
        )
        if not success:
            raise RuntimeError("Failed to initialize modified model with optimizations")

        model = plugin.load_model()
        if model is None:
            raise RuntimeError("Failed to load modified model")

        return plugin

    def benchmark_model_inference_speed(
        self, plugin, model_label, input_length=50, num_iterations=10
    ):
        """Benchmark inference speed for a specific model with comprehensive metrics."""
        # Prepare input
        input_ids = torch.randint(0, 1000, (1, input_length))

        # Warmup
        for _ in range(5):
            _ = plugin.infer("Hello, how are you?")

        # Clear cache before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Pre-garbage collection
        gc.collect()

        # Timing run
        start_time = time.time()
        inference_times = []

        for i in range(num_iterations):
            iter_start = time.time()
            _ = plugin.infer("Hello, how are you?")
            iter_end = time.time()
            inference_times.append(iter_end - iter_start)

        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_inference = mean(inference_times)
        std_time_per_inference = (
            stdev(inference_times) if len(inference_times) > 1 else 0.0
        )
        tokens_per_second = (
            input_length / avg_time_per_inference
            if avg_time_per_inference > 0
            else float("inf")
        )

        # Collect system metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        gpu_info = []
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]  # Get first GPU
            gpu_info = [
                {
                    "name": gpu.name,
                    "load": gpu.load,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                }
            ]

        # Get memory stats from plugin if available
        memory_stats = {}
        if hasattr(plugin, "get_memory_stats"):
            memory_stats = plugin.get_memory_stats()

        # Get compression stats from plugin if available
        compression_stats = {}
        if hasattr(plugin, "get_compression_stats"):
            compression_stats = plugin.get_compression_stats()

        # Get offloading stats from plugin if available
        offloading_stats = {}
        if hasattr(plugin, "get_offloading_stats"):
            offloading_stats = plugin.get_offloading_stats()

        return {
            "model_label": model_label,
            "total_time": total_time,
            "avg_time_per_inference": avg_time_per_inference,
            "std_time_per_inference": std_time_per_inference,
            "tokens_per_second": tokens_per_second,
            "num_iterations": num_iterations,
            "input_length": input_length,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_info.percent,
            "gpu_info": gpu_info,
            "memory_stats": memory_stats,
            "compression_stats": compression_stats,
            "offloading_stats": offloading_stats,
            "individual_times": inference_times,
        }

    def run_comparison_benchmark(self):
        """Run the comparison benchmark between original and modified models."""
        print("=" * 80)
        print("GLM-4.7 INFERENCE SPEED COMPARISON BENCHMARK WITH OPTIMIZATIONS")
        print("=" * 80)

        # Load original model
        print("\nLoading original model (baseline)...")
        original_plugin = self.load_original_model()
        print("Original model loaded successfully.")

        # Load modified model
        print("\nLoading modified model (with optimizations)...")
        modified_plugin = self.load_modified_model()
        print("Modified model loaded successfully.")

        # Define test parameters
        input_lengths = [20, 50, 100, 200]
        num_iterations = 10

        original_results = []
        modified_results = []

        for length in input_lengths:
            print(f"\nTesting with input length: {length} tokens")

            # Benchmark original model
            print("  Benchmarking original model...")
            orig_result = self.benchmark_model_inference_speed(
                original_plugin, "original", length, num_iterations
            )
            original_results.append(orig_result)
            print(
                f"    Original: {orig_result['tokens_per_second']:.2f} tokens/sec "
                f"(±{orig_result['std_time_per_inference']*1000:.2f}ms)"
            )

            # Benchmark modified model
            print("  Benchmarking modified model...")
            mod_result = self.benchmark_model_inference_speed(
                modified_plugin, "modified", length, num_iterations
            )
            modified_results.append(mod_result)
            print(
                f"    Modified: {mod_result['tokens_per_second']:.2f} tokens/sec "
                f"(±{mod_result['std_time_per_inference']*1000:.2f}ms)"
            )

        # Store results
        self.results["original_results"] = original_results
        self.results["modified_results"] = modified_results

        # Calculate comparison metrics
        comparison_metrics = []
        for i, (orig_res, mod_res) in enumerate(
            zip(original_results, modified_results)
        ):
            # Calculate percentage improvement (positive if modified is faster)
            if orig_res["tokens_per_second"] != float("inf") and mod_res[
                "tokens_per_second"
            ] != float("inf"):
                improvement = (
                    (mod_res["tokens_per_second"] - orig_res["tokens_per_second"])
                    / orig_res["tokens_per_second"]
                ) * 100
            elif orig_res["tokens_per_second"] == float("inf"):
                improvement = (
                    float("inf") if mod_res["tokens_per_second"] != float("inf") else 0
                )
            else:
                improvement = float("-inf")  # Modified is infinitely slower

            comparison_metrics.append(
                {
                    "input_length": orig_res["input_length"],
                    "original_tps": orig_res["tokens_per_second"],
                    "modified_tps": mod_res["tokens_per_second"],
                    "improvement_percentage": improvement,
                    "original_avg_time": orig_res["avg_time_per_inference"],
                    "modified_avg_time": mod_res["avg_time_per_inference"],
                    "original_std_time": orig_res["std_time_per_inference"],
                    "modified_std_time": mod_res["std_time_per_inference"],
                    "original_memory_stats": orig_res["memory_stats"],
                    "modified_memory_stats": mod_res["memory_stats"],
                    "original_compression_stats": orig_res["compression_stats"],
                    "modified_compression_stats": mod_res["compression_stats"],
                    "original_offloading_stats": orig_res["offloading_stats"],
                    "modified_offloading_stats": mod_res["offloading_stats"],
                }
            )

        self.results["comparison"] = comparison_metrics

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY - GLM-4.7 INFERENCE SPEED COMPARISON WITH OPTIMIZATIONS")
        print("=" * 80)
        print(
            f"{'Input Length':<12} {'Orig TPS':<12} {'Mod TPS':<12} {'Improvement %':<15} {'Orig Time':<12} {'Mod Time':<12}"
        )
        print("-" * 80)
        for metric in comparison_metrics:
            improvement_str = (
                f"{metric['improvement_percentage']:.2f}%"
                if isinstance(metric["improvement_percentage"], (int, float))
                else str(metric["improvement_percentage"])
            )
            print(
                f"{metric['input_length']:<12} {metric['original_tps']:<12.2f} {metric['modified_tps']:<12.2f} {improvement_str:<15} "
                f"{metric['original_avg_time']:<12.4f} {metric['modified_avg_time']:<12.4f}"
            )

        # Calculate overall improvement
        orig_avg_tps = mean(
            [
                r["tokens_per_second"]
                for r in original_results
                if r["tokens_per_second"] != float("inf")
            ]
        )
        mod_avg_tps = mean(
            [
                r["tokens_per_second"]
                for r in modified_results
                if r["tokens_per_second"] != float("inf")
            ]
        )

        if orig_avg_tps != 0:
            overall_improvement = ((mod_avg_tps - orig_avg_tps) / orig_avg_tps) * 100
        else:
            overall_improvement = float("inf") if mod_avg_tps > 0 else 0

        print(
            f"\nOverall Average Tokens/Sec - Original: {orig_avg_tps:.2f}, Modified: {mod_avg_tps:.2f}"
        )
        print(f"Overall Improvement: {overall_improvement:.2f}%")

        # Calculate memory efficiency improvements
        orig_avg_memory = mean([r["memory_percent"] for r in original_results])
        mod_avg_memory = mean([r["memory_percent"] for r in modified_results])
        memory_improvement = (
            ((orig_avg_memory - mod_avg_memory) / orig_avg_memory) * 100
            if orig_avg_memory > 0
            else 0
        )

        print(
            f"Memory Efficiency Improvement: {memory_improvement:.2f}% (lower is better)"
        )

        # Cleanup
        original_plugin.cleanup()
        modified_plugin.cleanup()

        return self.results

    def save_results(self):
        """Save benchmark results to JSON and CSV files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_filename = os.path.join(
            self.results_dir,
            f"glm_4_7_inference_speed_comparison_optimized_{timestamp}.json",
        )
        with open(json_filename, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save CSV results
        csv_filename = os.path.join(
            self.results_dir,
            f"glm_4_7_inference_speed_comparison_optimized_{timestamp}.csv",
        )
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "Model Name",
                    "Input Length",
                    "Original TPS",
                    "Modified TPS",
                    "Improvement Percentage",
                    "Original Avg Time",
                    "Modified Avg Time",
                    "Original Std Time",
                    "Modified Std Time",
                    "Memory Improvement %",
                    "Timestamp",
                ]
            )

            # Write data rows
            for metric in self.results["comparison"]:
                writer.writerow(
                    [
                        self.results["model_name"],
                        metric["input_length"],
                        metric["original_tps"],
                        metric["modified_tps"],
                        metric["improvement_percentage"],
                        metric["original_avg_time"],
                        metric["modified_avg_time"],
                        metric["original_std_time"],
                        metric["modified_std_time"],
                        (
                            (
                                (
                                    metric["original_memory_stats"].get(
                                        "system_memory_percent", 0
                                    )
                                    - metric["modified_memory_stats"].get(
                                        "system_memory_percent", 0
                                    )
                                )
                                / metric["original_memory_stats"].get(
                                    "system_memory_percent", 1
                                )
                            )
                            * 100
                            if metric["original_memory_stats"].get(
                                "system_memory_percent", 1
                            )
                            > 0
                            else 0
                        ),
                        self.results["timestamp"],
                    ]
                )

        print(f"\nResults saved to:")
        print(f"  JSON: {json_filename}")
        print(f"  CSV: {csv_filename}")

    def start_monitoring(self):
        """Start monitoring system status every 5 minutes."""
        self.monitoring_active = True

        def monitor_job():
            if self.monitoring_active:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] System Status Monitor"
                )
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                print(f"  CPU: {cpu_percent}% | Memory: {memory_info.percent}%")

                if torch.cuda.is_available():
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        print(
                            f"  GPU {i}: {gpu.load*100:.1f}% load, {gpu.memoryUsed}/{gpu.memoryTotal} MB"
                        )

        # Schedule the monitoring job
        schedule.every(5).minutes.do(monitor_job)

        def run_scheduler():
            while self.monitoring_active:
                schedule.run_pending()
                time.sleep(1)

        # Run scheduler in a separate thread
        monitor_thread = threading.Thread(target=run_scheduler, daemon=True)
        monitor_thread.start()

        print("Monitoring started (every 5 minutes)")

    def stop_monitoring(self):
        """Stop the monitoring process."""
        self.monitoring_active = False
        schedule.clear()
        print("Monitoring stopped")


def run_glm_4_7_benchmark():
    """Run the GLM-4.7 inference speed comparison benchmark."""
    benchmark = GLM47InferenceSpeedComparisonBenchmark()

    try:
        # Start monitoring
        benchmark.start_monitoring()

        # Run the comparison benchmark
        results = benchmark.run_comparison_benchmark()

        # Save results
        benchmark.save_results()

        # Stop monitoring
        benchmark.stop_monitoring()

        return results
    except Exception as e:
        print(f"Error running GLM-4.7 benchmark: {e}")
        # Stop monitoring in case of error
        benchmark.stop_monitoring()
        raise


if __name__ == "__main__":
    run_glm_4_7_benchmark()
