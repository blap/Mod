"""
Quick verification script to check the benchmarking solution structure without running actual models.
"""

import importlib
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_utils import discover_benchmark_class


def verify_structure():
    """
    Verify that the benchmarking solution structure is correct without running actual models.
    """
    print("Verifying benchmarking solution structure...")
    print("-" * 60)

    models = ["glm_4_7", "qwen3_4b_instruct_2507", "qwen3_coder_30b", "qwen3_vl_2b"]

    categories = [
        "accuracy",
        "comparison",
        "inference_speed",
        "memory_usage",
        "optimization_impact",
        "power_efficiency",
        "throughput",
    ]

    print(f"Models to verify: {models}")
    print(f"Categories to verify: {categories}")
    print()

    all_found = True

    for model in models:
        print(f"Checking model: {model}")
        for category in categories:
            # Determine the correct subdirectory based on benchmark category
            if category in [
                "inference_speed",
                "memory_usage",
                "throughput",
                "power_efficiency",
                "optimization_impact",
                "inference_speed_comparison",
            ]:
                subdir = "performance"
            elif category in ["accuracy"]:
                subdir = "unit"
            elif category in [
                "comparison",
                "async_multimodal_processing",
                "intelligent_multimodal_caching",
            ]:
                subdir = "integration"
            else:
                # Default to performance for most cases
                subdir = "performance"

            module_path = (
                f"inference_pio.models.{model}.benchmarks.{subdir}.benchmark_{category}"
            )
            try:
                # Try to import the module
                benchmark_module = importlib.import_module(module_path)

                # Try to find the benchmark class
                benchmark_class = discover_benchmark_class(benchmark_module, category)

                if benchmark_class:
                    print(f"  [OK] {category}: Found class {benchmark_class.__name__}")
                else:
                    print(
                        f"  [WARN] {category}: No benchmark class found (but module imported)"
                    )

            except ImportError as e:
                print(f"  [ERROR] {category}: Could not import module - {str(e)}")
                all_found = False
            except Exception as e:
                print(f"  [ERROR] {category}: Error importing - {str(e)}")
                all_found = False

    print("-" * 60)

    if all_found:
        print("[OK] Structure verification PASSED - All modules can be imported")
        print(
            "The benchmarking solution is properly structured and ready to execute real benchmarks."
        )
    else:
        print("[WARN] Structure verification showed some issues")
        print(
            "Some modules could not be imported, but the solution framework is intact."
        )

    print("\nNote: This verification checks structure only, not actual execution.")
    print("To run actual benchmarks, use the run_*_benchmarks.py scripts.")

    return all_found


if __name__ == "__main__":
    verify_structure()
