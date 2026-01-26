"""
Standardized Benchmark for Power Efficiency - Qwen3-4B-Instruct-2507

This module benchmarks the power efficiency for the Qwen3-4B-Instruct-2507 model.
"""

import time
import unittest
import torch
import psutil
import threading
from inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin


class BenchmarkQwen34BInstruct2507PowerEfficiency(unittest.TestCase):
    """Benchmark cases for Qwen3-4B-Instruct-2507 power efficiency."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_qwen3_4b_instruct_2507_plugin()
        success = self.plugin.initialize(device="cpu")  # Using CPU for power efficiency testing
        self.assertTrue(success)
        self.model = self.plugin.load_model()
        self.assertTrue(self.model is not None)

    def measure_cpu_utilization(self, duration=5):
        """Measure CPU utilization over a period of time."""
        cpu_percentages = []
        
        def monitor_cpu():
            start_time = time.time()
            while time.time() - start_time < duration:
                cpu_percent = psutil.cpu_percent(interval=0.1, percpu=False)
                cpu_percentages.append(cpu_percent)
                time.sleep(0.1)
        
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Perform some work during monitoring
        start_time = time.time()
        while time.time() - start_time < duration:
            # Perform lightweight operations to keep CPU active
            input_ids = torch.randint(0, 1000, (1, 10))
            _ = self.plugin.infer(input_ids)
        
        monitor_thread.join()
        
        avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0
        max_cpu = max(cpu_percentages) if cpu_percentages else 0
        
        return avg_cpu, max_cpu

    def benchmark_power_efficiency(self, workload_duration=10):
        """Benchmark power efficiency by measuring CPU utilization during workload."""
        # Measure baseline CPU utilization
        baseline_avg, baseline_max = self.measure_cpu_utilization(duration=3)
        
        # Run workload and measure CPU utilization
        workload_avg, workload_max = self.measure_cpu_utilization(duration=workload_duration)
        
        # Calculate efficiency metrics
        cpu_increase = workload_avg - baseline_avg
        efficiency_ratio = workload_avg / (cpu_increase + 1)  # Add 1 to avoid division by zero
        
        return {
            'baseline_avg_cpu': baseline_avg,
            'baseline_max_cpu': baseline_max,
            'workload_avg_cpu': workload_avg,
            'workload_max_cpu': workload_max,
            'cpu_increase': cpu_increase,
            'efficiency_ratio': efficiency_ratio,
            'workload_duration': workload_duration
        }

    def test_cpu_utilization_baseline(self):
        """Test baseline CPU utilization."""
        baseline_avg, baseline_max = self.measure_cpu_utilization(duration=3)
        
        print(f"\nQwen3-4B-Instruct-2507 Baseline CPU Utilization:")
        print(f"  Average: {baseline_avg:.2f}%")
        print(f"  Maximum: {baseline_max:.2f}%")
        
        # Basic sanity check
        self.assertGreaterEqual(baseline_avg, 0)
        self.assertLessEqual(baseline_avg, 100)

    def test_cpu_utilization_under_workload(self):
        """Test CPU utilization under model workload."""
        results = self.benchmark_power_efficiency(workload_duration=10)
        
        print(f"\nQwen3-4B-Instruct-2507 CPU Utilization Under Workload:")
        print(f"  Baseline average: {results['baseline_avg_cpu']:.2f}%")
        print(f"  Workload average: {results['workload_avg_cpu']:.2f}%")
        print(f"  CPU increase: {results['cpu_increase']:.2f}%")
        print(f"  Efficiency ratio: {results['efficiency_ratio']:.2f}")
        
        # Basic sanity checks
        self.assertGreaterEqual(results['workload_avg_cpu'], 0)
        self.assertLessEqual(results['workload_avg_cpu'], 100)

    def test_power_efficiency_with_different_batch_sizes(self):
        """Test power efficiency with different batch sizes."""
        batch_sizes = [1, 2, 4, 8]
        
        efficiency_results = []
        
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                # Measure baseline
                baseline_avg, _ = self.measure_cpu_utilization(duration=2)
                
                # Run workload with specific batch size
                start_time = time.time()
                duration = 5
                operations = 0
                
                while time.time() - start_time < duration:
                    input_ids = torch.randint(0, 1000, (batch_size, 20))
                    _ = self.plugin.infer(input_ids)
                    operations += 1
                
                workload_time = time.time() - start_time
                workload_avg, _ = self.measure_cpu_utilization(duration=1)  # Quick measurement
                
                cpu_increase = workload_avg - baseline_avg
                ops_per_second = operations / workload_time
                efficiency = ops_per_second / (cpu_increase + 1) if cpu_increase > 0 else ops_per_second
                
                efficiency_results.append(efficiency)
                
                print(f"  Batch size {batch_size}: {efficiency:.2f} ops/(CPU%+1)")

    def test_power_efficiency_with_different_sequence_lengths(self):
        """Test power efficiency with different sequence lengths."""
        seq_lengths = [10, 50, 100, 200]
        
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                # Measure baseline
                baseline_avg, _ = self.measure_cpu_utilization(duration=2)
                
                # Run workload with specific sequence length
                start_time = time.time()
                duration = 5
                operations = 0
                
                while time.time() - start_time < duration:
                    input_ids = torch.randint(0, 1000, (1, seq_len))
                    _ = self.plugin.infer(input_ids)
                    operations += 1
                
                workload_time = time.time() - start_time
                workload_avg, _ = self.measure_cpu_utilization(duration=1)  # Quick measurement
                
                cpu_increase = workload_avg - baseline_avg
                ops_per_second = operations / workload_time
                efficiency = ops_per_second / (cpu_increase + 1) if cpu_increase > 0 else ops_per_second
                
                print(f"  Sequence length {seq_len}: {efficiency:.2f} ops/(CPU%+1)")

    def test_power_efficiency_during_generation(self):
        """Test power efficiency during text generation."""
        # Measure baseline
        baseline_avg, _ = self.measure_cpu_utilization(duration=2)
        
        # Run text generation workload
        start_time = time.time()
        duration = 8
        operations = 0
        
        prompts = ["The future of AI", "Machine learning", "Deep learning", "NLP models"]
        
        while time.time() - start_time < duration:
            prompt = prompts[operations % len(prompts)]
            _ = self.plugin.generate_text(prompt, max_new_tokens=20)
            operations += 1
        
        workload_time = time.time() - start_time
        workload_avg, workload_max = self.measure_cpu_utilization(duration=1)
        
        cpu_increase = workload_avg - baseline_avg
        ops_per_second = operations / workload_time
        efficiency = ops_per_second / (cpu_increase + 1) if cpu_increase > 0 else ops_per_second
        
        print(f"\nQwen3-4B-Instruct-2507 Power Efficiency During Generation:")
        print(f"  Operations: {operations}")
        print(f"  Duration: {workload_time:.2f}s")
        print(f"  Baseline CPU: {baseline_avg:.2f}%")
        print(f"  Workload CPU: {workload_avg:.2f}%")
        print(f"  Efficiency: {efficiency:.2f} ops/(CPU%+1)")

    def test_power_efficiency_with_optimizations(self):
        """Test power efficiency with different optimization settings."""
        # Test without optimizations
        plugin_no_opt = create_qwen3_4b_instruct_2507_plugin()
        success = plugin_no_opt.initialize(device="cpu", use_flash_attention=False, memory_efficient=False)
        self.assertTrue(success)
        model_no_opt = plugin_no_opt.load_model()
        self.assertIsNotNone(model_no_opt)
        
        # Measure efficiency without optimizations
        baseline_avg, _ = self.measure_cpu_utilization(duration=2)
        
        start_time = time.time()
        duration = 5
        operations = 0
        while time.time() - start_time < duration:
            input_ids = torch.randint(0, 1000, (1, 30))
            _ = plugin_no_opt.infer(input_ids)
            operations += 1
        
        workload_time = time.time() - start_time
        workload_avg_no_opt, _ = self.measure_cpu_utilization(duration=1)
        
        cpu_increase_no_opt = workload_avg_no_opt - baseline_avg
        ops_per_second_no_opt = operations / workload_time
        efficiency_no_opt = ops_per_second_no_opt / (cpu_increase_no_opt + 1)
        
        # Cleanup
        if hasattr(plugin_no_opt, 'cleanup'):
            plugin_no_opt.cleanup()
        
        # Now test with optimizations (if supported)
        plugin_opt = create_qwen3_4b_instruct_2507_plugin()
        success = plugin_opt.initialize(device="cpu", use_flash_attention=True, memory_efficient=True)
        self.assertTrue(success)
        model_opt = plugin_opt.load_model()
        self.assertIsNotNone(model_opt)
        
        # Measure efficiency with optimizations
        baseline_avg, _ = self.measure_cpu_utilization(duration=2)
        
        start_time = time.time()
        operations = 0
        while time.time() - start_time < duration:
            input_ids = torch.randint(0, 1000, (1, 30))
            _ = plugin_opt.infer(input_ids)
            operations += 1
        
        workload_time = time.time() - start_time
        workload_avg_opt, _ = self.measure_cpu_utilization(duration=1)
        
        cpu_increase_opt = workload_avg_opt - baseline_avg
        ops_per_second_opt = operations / workload_time
        efficiency_opt = ops_per_second_opt / (cpu_increase_opt + 1)
        
        print(f"\nQwen3-4B-Instruct-2507 Power Efficiency Comparison:")
        print(f"  Without optimizations: {efficiency_no_opt:.2f} ops/(CPU%+1)")
        print(f"  With optimizations: {efficiency_opt:.2f} ops/(CPU%+1)")
        
        # Cleanup
        if hasattr(plugin_opt, 'cleanup'):
            plugin_opt.cleanup()

    def test_power_efficiency_memory_correlation(self):
        """Test correlation between power efficiency and memory usage."""
        import os
        
        # Get baseline memory
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run workload and measure both CPU and memory
        baseline_cpu, _ = self.measure_cpu_utilization(duration=2)
        
        start_time = time.time()
        duration = 8
        operations = 0
        
        while time.time() - start_time < duration:
            input_ids = torch.randint(0, 1000, (2, 40))  # Larger batch to increase resource usage
            _ = self.plugin.infer(input_ids)
            operations += 1
        
        workload_time = time.time() - start_time
        workload_cpu, _ = self.measure_cpu_utilization(duration=1)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        cpu_increase = workload_cpu - baseline_cpu
        ops_per_second = operations / workload_time
        efficiency = ops_per_second / (cpu_increase + 1)
        
        print(f"\nQwen3-4B-Instruct-2507 Power-Memory Efficiency Correlation:")
        print(f"  CPU efficiency: {efficiency:.2f} ops/(CPU%+1)")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        print(f"  Operations per second: {ops_per_second:.2f}")

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, 'cleanup') and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == '__main__':
    unittest.main()