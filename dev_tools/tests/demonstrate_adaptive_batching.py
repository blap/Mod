"""
Demonstration script for Adaptive Micro-batching functionality.

This script demonstrates how the adaptive batching system works with model plugins.
"""

import time

import torch

from src.inference_pio.common.adaptive_batch_manager import AdaptiveBatchManager
from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin


def demonstrate_adaptive_batching():
    """Demonstrate the adaptive batching functionality."""
    print("=== Adaptive Micro-batching Demonstration ===\n")

    # Create an adaptive batch manager
    print("1. Creating Adaptive Batch Manager...")
    batch_manager = AdaptiveBatchManager(
        initial_batch_size=4,
        min_batch_size=1,
        max_batch_size=16,
        memory_threshold_ratio=0.85,
        performance_window_size=5,
        adjustment_factor=0.2,
        cooldown_period=2.0,
    )

    print(f"   Initial batch size: {batch_manager.current_batch_size}")
    print(
        f"   Batch size range: [{batch_manager.min_batch_size}, {batch_manager.max_batch_size}]"
    )
    print(f"   Memory threshold: {batch_manager.memory_threshold_ratio}\n")

    # Simulate processing with varying performance
    print("2. Simulating batch processing with varying performance...\n")

    for i in range(10):
        # Simulate different processing scenarios
        if i < 3:
            # Good performance scenario
            processing_time = 100.0  # ms
            tokens_processed = 100
            scenario = "Good performance"
        elif i < 6:
            # Degraded performance scenario
            processing_time = 500.0  # ms
            tokens_processed = 20
            scenario = "Poor performance"
        else:
            # Recovering performance scenario
            processing_time = 150.0  # ms
            tokens_processed = 80
            scenario = "Recovering performance"

        print(f"   Iteration {i+1}: {scenario}")
        print(f"     Processing {tokens_processed} tokens in {processing_time}ms")

        # Get optimal batch size based on performance
        optimal_size = batch_manager.get_optimal_batch_size(
            processing_time, tokens_processed
        )

        # Get current status
        status = batch_manager.get_status_report()
        current_size = status["current_batch_size"]

        print(f"     Recommended batch size: {optimal_size}")
        print(f"     Current batch size: {current_size}")
        if "latest_metrics" in status and status["latest_metrics"]:
            print(
                f"     Memory pressure: {status['latest_metrics']['memory_pressure_ratio']:.2f}"
            )
            print(
                f"     Performance score: {status['latest_metrics']['performance_score']:.2f}"
            )
        else:
            print("     Memory pressure: N/A")
            print("     Performance score: N/A")
        print()

        time.sleep(0.1)  # Brief pause to simulate processing

    print("3. Final status:")
    final_status = batch_manager.get_status_report()
    print(f"   Final batch size: {final_status['current_batch_size']}")
    print(
        f"   Average processing time: {final_status['average_processing_time_ms']:.2f}ms"
    )
    print(
        f"   Average throughput: {final_status['average_throughput_tokens_per_sec']:.2f} tokens/sec"
    )


def demonstrate_plugin_integration():
    """Demonstrate how adaptive batching integrates with model plugins."""
    print("\n=== Plugin Integration Demonstration ===\n")

    # Create a plugin
    print("1. Creating GLM-4.7 Plugin...")
    plugin = GLM_4_7_Plugin()
    print(f"   Plugin name: {plugin.metadata.name}")
    print(
        f"   Has adaptive batching methods: {hasattr(plugin, 'setup_adaptive_batching')}\n"
    )

    # Initialize the plugin with adaptive batching enabled
    print("2. Initializing plugin with adaptive batching...")
    init_success = plugin.initialize(enable_adaptive_batching=True)
    print(f"   Initialization success: {init_success}\n")

    # Show batching status
    print("3. Checking batching status...")
    status = plugin.get_batching_status()
    print(f"   Adaptive batching enabled: {status['adaptive_batching_enabled']}")
    print(f"   Current batch size: {status['current_batch_size']}")
    print(f"   Memory pressure ratio: {status['memory_pressure_ratio']:.2f}")
    print(f"   Performance score: {status['performance_score']:.2f}\n")

    # Simulate getting optimal batch size during inference
    print("4. Simulating batch size recommendation during inference...")
    for i in range(3):
        # Simulate different inference scenarios
        processing_time = 200.0 - (i * 30)  # Getting faster
        tokens_processed = 50 + (i * 10)  # Processing more tokens

        optimal_size = plugin.get_optimal_batch_size(processing_time, tokens_processed)
        print(
            f"   Scenario {i+1}: Processed {tokens_processed} tokens in {processing_time}ms -> Recommended batch size: {optimal_size}"
        )

    print("\n5. Demonstrating batch size adjustment...")
    new_size, was_adjusted, reason = plugin.adjust_batch_size()
    print(f"   New batch size: {new_size}")
    print(f"   Was adjusted: {was_adjusted}")
    print(f"   Reason: {reason}")


if __name__ == "__main__":
    demonstrate_adaptive_batching()
    demonstrate_plugin_integration()

    print("\n=== Summary ===")
    print(
        "The adaptive micro-batching system successfully monitors performance and memory"
    )
    print(
        "metrics to dynamically adjust batch sizes for optimal efficiency. The system"
    )
    print(
        "can detect performance degradation and memory pressure, responding by reducing"
    )
    print("batch sizes, and can increase batch sizes when conditions improve.")
