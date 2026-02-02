"""
Benchmark Migration Script

This script migrates existing benchmarks to the standardized structure.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List


def find_existing_benchmarks(
    base_path: str = "src/inference_pio/models",
) -> Dict[str, List[str]]:
    """Find all existing benchmark files in model directories."""
    base_path = Path(base_path)
    model_benchmarks = {}

    if not base_path.exists():
        print(f"Base path does not exist: {base_path}")
        return model_benchmarks

    for model_dir in base_path.iterdir():
        if model_dir.is_dir() and not model_dir.name.startswith("."):
            # Look for existing benchmark files
            benchmark_files = []
            for benchmark_dir in model_dir.glob("benchmarks/*"):
                if benchmark_dir.is_dir():
                    for benchmark_file in benchmark_dir.glob("*.py"):
                        benchmark_files.append(str(benchmark_file))

            if benchmark_files:
                model_benchmarks[model_dir.name] = benchmark_files

    return model_benchmarks


def create_standardized_structure(model_dir: str):
    """Create the standardized benchmark directory structure for a model."""
    model_path = Path(model_dir)

    # Create benchmark directories
    benchmark_base = model_path / "benchmarks"
    benchmark_base.mkdir(exist_ok=True)

    unit_dir = benchmark_base / "unit"
    unit_dir.mkdir(exist_ok=True)

    integration_dir = benchmark_base / "integration"
    integration_dir.mkdir(exist_ok=True)

    performance_dir = benchmark_base / "performance"
    performance_dir.mkdir(exist_ok=True)


def migrate_existing_benchmarks():
    """Migrate existing benchmarks to the standardized structure."""
    print("=" * 80)
    print("BENCHMARK MIGRATION TO STANDARDIZED STRUCTURE")
    print("=" * 80)

    # Find existing benchmarks
    existing_benchmarks = find_existing_benchmarks()

    if not existing_benchmarks:
        print("No existing benchmarks found to migrate.")
        return

    print(f"Found benchmarks in {len(existing_benchmarks)} models:")
    for model, files in existing_benchmarks.items():
        print(f"  {model}: {len(files)} files")
        for file in files:
            print(f"    - {file}")

    print(f"\nCreating standardized structure for all models...")

    # Create standardized structure for each model
    base_path = Path("src/inference_pio/models")
    for model_name in existing_benchmarks.keys():
        model_dir = base_path / model_name
        if model_dir.exists():
            create_standardized_structure(str(model_dir))
            print(f"  Created structure for {model_name}")

    print(
        f"\nMigration completed! All models now have the standardized benchmark structure."
    )
    print(f"\nNote: Existing benchmark files were not moved to avoid losing work.")
    print(
        f"You may want to review and update existing benchmarks to use the new interface."
    )


def create_example_benchmarks(existing_benchmarks):
    """Create example benchmark files using the standardized interface."""
    print(f"\nCreating example benchmark files...")

    base_path = Path("src/inference_pio/models")

    # Example accuracy benchmark
    accuracy_benchmark_content = '''"""
Standardized Accuracy Benchmark for {model_name}

This module benchmarks the accuracy for the {model_name} model using the standardized interface.
"""

import unittest
import torch
from inference_pio.models.{model_dir_name}.plugin import create_{plugin_func_name}
from inference_pio.common.benchmark_interface import AccuracyBenchmark


class {model_class_name}AccuracyBenchmark(unittest.TestCase):
    """Benchmark cases for {model_name} accuracy using standardized interface."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_{plugin_func_name}()
        success = self.plugin.initialize(device="cpu", use_mock_model=True)
        self.assertTrue(success)
        self.model_name = "{model_name}"

    def test_accuracy_with_standard_interface(self):
        """Test accuracy using the standardized benchmark interface."""
        benchmark = AccuracyBenchmark(self.plugin, self.model_name)
        result = benchmark.run()

        print(f"{{self.model_name}} Accuracy: {{result.value}} {{result.unit}}")

        # Basic validation
        self.assertIsNotNone(result.value)
        self.assertGreaterEqual(result.value, 0)  # Accuracy should be non-negative

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, 'cleanup') and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == '__main__':
    unittest.main()
'''

    # Example performance benchmark
    performance_benchmark_content = '''"""
Standardized Performance Benchmark for {model_name}

This module benchmarks the performance for the {model_name} model using the standardized interface.
"""

import unittest
import torch
from inference_pio.models.{model_dir_name}.plugin import create_{plugin_func_name}
from inference_pio.common.benchmark_interface import (
    InferenceSpeedBenchmark,
    MemoryUsageBenchmark,
    BatchProcessingBenchmark
)


class {model_class_name}PerformanceBenchmark(unittest.TestCase):
    """Benchmark cases for {model_name} performance using standardized interface."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        self.plugin = create_{plugin_func_name}()
        success = self.plugin.initialize(device="cpu", use_mock_model=True)
        self.assertTrue(success)
        self.model_name = "{model_name}"

    def test_inference_speed_with_standard_interface(self):
        """Test inference speed using the standardized benchmark interface."""
        # Test different input lengths
        for input_length in [20, 50, 100]:
            benchmark = InferenceSpeedBenchmark(self.plugin, self.model_name, input_length=input_length)
            result = benchmark.run()

            print(f"{{self.model_name}} Inference Speed ({{input_length}} tokens): {{result.value}} {{result.unit}}")

            # Basic validation
            self.assertIsNotNone(result.value)
            if result.value != float('inf'):  # Handle infinite values
                self.assertGreater(result.value, 0)

    def test_memory_usage_with_standard_interface(self):
        """Test memory usage using the standardized benchmark interface."""
        benchmark = MemoryUsageBenchmark(self.plugin, self.model_name)
        result = benchmark.run()

        print(f"{{self.model_name}} Memory Usage: {{result.value}} {{result.unit}}")

        # Basic validation
        self.assertIsNotNone(result.value)
        self.assertGreaterEqual(result.value, 0)

    def test_batch_processing_with_standard_interface(self):
        """Test batch processing using the standardized benchmark interface."""
        benchmark = BatchProcessingBenchmark(self.plugin, self.model_name)
        result = benchmark.run()

        print(f"{{self.model_name}} Batch Processing Throughput: {{result.value}} {{result.unit}}")

        # Basic validation
        self.assertIsNotNone(result.value)
        if result.value != float('inf'):  # Handle infinite values
            self.assertGreaterEqual(result.value, 0)

    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self.plugin, 'cleanup') and self.plugin.is_loaded:
            self.plugin.cleanup()


if __name__ == '__main__':
    unittest.main()
'''

    # Create example benchmarks for each model that has existing benchmarks
    for model_name in existing_benchmarks.keys():
        # Convert model name to appropriate formats
        model_dir_name = model_name
        plugin_func_name = model_name.replace("-", "_").replace(".", "_").lower()
        model_class_name = "".join(
            word.capitalize() for word in plugin_func_name.split("_")
        )

        model_path = base_path / model_name

        # Create example accuracy benchmark if it doesn't exist
        accuracy_file = (
            model_path / "benchmarks" / "unit" / "benchmark_accuracy_standardized.py"
        )
        if not accuracy_file.exists():
            content = accuracy_benchmark_content.format(
                model_name=model_name,
                model_dir_name=model_dir_name,
                plugin_func_name=plugin_func_name,
                model_class_name=model_class_name,
            )
            accuracy_file.write_text(content)
            print(f"  Created example accuracy benchmark: {accuracy_file}")

        # Create example performance benchmark if it doesn't exist
        performance_file = (
            model_path
            / "benchmarks"
            / "performance"
            / "benchmark_performance_standardized.py"
        )
        if not performance_file.exists():
            content = performance_benchmark_content.format(
                model_name=model_name,
                model_dir_name=model_dir_name,
                plugin_func_name=plugin_func_name,
                model_class_name=model_class_name,
            )
            performance_file.write_text(content)
            print(f"  Created example performance benchmark: {performance_file}")


def main():
    """Main function to run the migration."""
    # Find existing benchmarks
    existing_benchmarks = find_existing_benchmarks()

    migrate_existing_benchmarks()
    create_example_benchmarks(existing_benchmarks)

    print(f"\n{'='*80}")
    print("MIGRATION COMPLETE")
    print(f"{'='*80}")
    print("Next steps:")
    print("1. Review the created standardized directory structure")
    print("2. Update existing benchmarks to use the new standardized interface")
    print("3. Use the example benchmark files as templates for your implementations")
    print("4. Run the verification script to ensure compliance")


if __name__ == "__main__":
    main()
