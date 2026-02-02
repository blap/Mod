"""
Comprehensive Benchmark for Inference Speed Comparison - Qwen3-4B-Instruct-2507

This module benchmarks the inference speed comparison between the original
and modified versions of the Qwen3-4B-Instruct-2507 model.
"""

import csv
import json
import os
import threading
import time
from datetime import datetime
from statistics import mean

import GPUtil
import psutil
import schedule
import torch


class Qwen34BInstruct2507InferenceSpeedComparisonBenchmark:
    """Benchmark class for comparing inference speed between original and modified Qwen3-4B-Instruct-2507 models."""

    def __init__(self):
        self.results = {
            "model_name": "Qwen3-4B-Instruct-2507",
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
        """Load the original version of the model from H drive."""
        from inference_pio.models.qwen3_4b_instruct_2507.plugin import (
            create_qwen3_4b_instruct_2507_plugin,
        )

        plugin = create_qwen3_4b_instruct_2507_plugin()

        # Initialize with CPU to ensure consistency, pointing to H drive model
        success = plugin.initialize(
            device="cpu", model_path="H:/Qwen3-4B-Instruct-2507"
        )
        if not success:
            raise RuntimeError("Failed to initialize original model from H drive")

        model = plugin.load_model()
        if model is None:
            raise RuntimeError("Failed to load original model from H drive")

        return plugin

    def load_modified_model(self):
        """Load the modified version of the model from H drive."""
        from inference_pio.models.qwen3_4b_instruct_2507.plugin import (
            create_qwen3_4b_instruct_2507_plugin,
        )

        plugin = create_qwen3_4b_instruct_2507_plugin()

        # Initialize with CPU to ensure consistency, pointing to H drive model
        success = plugin.initialize(
            device="cpu", model_path="H:/Qwen3-4B-Instruct-2507"
        )
        if not success:
            raise RuntimeError("Failed to initialize modified model from H drive")

        model = plugin.load_model()
        if model is None:
            raise RuntimeError("Failed to load modified model from H drive")

        return plugin

    def benchmark_model_inference_speed(
        self, plugin, model_label, input_length=50, num_iterations=10
    ):
        """Benchmark inference speed for a specific model."""
        # Prepare input
        input_ids = torch.randint(0, 1000, (1, input_length))

        # Warmup
        for _ in range(3):
            _ = plugin.infer(input_ids)

        # Timing run
        start_time = time.time()
        for i in range(num_iterations):
            _ = plugin.infer(input_ids)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time_per_inference = total_time / num_iterations
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

        return {
            "model_label": model_label,
            "total_time": total_time,
            "avg_time_per_inference": avg_time_per_inference,
            "tokens_per_second": tokens_per_second,
            "num_iterations": num_iterations,
            "input_length": input_length,
            "cpu_percent": cpu_percent,
            "memory_percent": memory_info.percent,
            "gpu_info": gpu_info,
        }

    def run_comparison_benchmark(self):
        """Run the comparison benchmark between original and modified models."""
        print("=" * 70)
        print("QWEN3-4B-INSTRUCT-2507 INFERENCE SPEED COMPARISON BENCHMARK")
        print("=" * 70)

        # Load original model
        print("\nLoading original model...")
        original_plugin = self.load_original_model()
        print("Original model loaded successfully.")

        # Load modified model
        print("\nLoading modified model...")
        modified_plugin = self.load_modified_model()
        print("Modified model loaded successfully.")

        # Define test parameters
        input_lengths = [20, 50, 100]
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
            print(f"    Original: {orig_result['tokens_per_second']:.2f} tokens/sec")

            # Benchmark modified model
            print("  Benchmarking modified model...")
            mod_result = self.benchmark_model_inference_speed(
                modified_plugin, "modified", length, num_iterations
            )
            modified_results.append(mod_result)
            print(f"    Modified: {mod_result['tokens_per_second']:.2f} tokens/sec")

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
                }
            )

        self.results["comparison"] = comparison_metrics

        # Print summary
        print("\n" + "=" * 70)
        print("SUMMARY - QWEN3-4B-INSTRUCT-2507 INFERENCE SPEED COMPARISON")
        print("=" * 70)
        print(
            f"{'Input Length':<12} {'Orig TPS':<12} {'Mod TPS':<12} {'Improvement %':<15}"
        )
        print("-" * 50)
        for metric in comparison_metrics:
            improvement_str = (
                f"{metric['improvement_percentage']:.2f}%"
                if isinstance(metric["improvement_percentage"], (int, float))
                else str(metric["improvement_percentage"])
            )
            print(
                f"{metric['input_length']:<12} {metric['original_tps']:<12.2f} {metric['modified_tps']:<12.2f} {improvement_str:<15}"
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
            f"qwen3_4b_instruct_2507_inference_speed_comparison_{timestamp}.json",
        )
        with open(json_filename, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save CSV results
        csv_filename = os.path.join(
            self.results_dir,
            f"qwen3_4b_instruct_2507_inference_speed_comparison_{timestamp}.csv",
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


def run_qwen3_4b_instruct_2507_benchmark():
    """Run the Qwen3-4B-Instruct-2507 inference speed comparison benchmark."""
    benchmark = Qwen34BInstruct2507InferenceSpeedComparisonBenchmark()

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
        print(f"Error running Qwen3-4B-Instruct-2507 benchmark: {e}")
        # Stop monitoring in case of error
        benchmark.stop_monitoring()
        raise


if __name__ == "__main__":
    run_qwen3_4b_instruct_2507_benchmark()
