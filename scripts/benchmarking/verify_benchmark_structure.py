"""
Benchmark Structure Verification Script

This script verifies that all models in the project follow the standardized benchmark structure.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def find_model_directories(base_path: str = "src/inference_pio/models") -> List[str]:
    """Find all model directories in the project."""
    model_dirs = []
    base_path = Path(base_path)

    if not base_path.exists():
        print(f"Base path does not exist: {base_path}")
        return model_dirs

    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Check if it has a plugin.py file to confirm it's a model directory
            plugin_file = item / "plugin.py"
            if plugin_file.exists():
                model_dirs.append(str(item))

    return model_dirs


def check_benchmark_structure(model_dir: str) -> Dict[str, bool]:
    """Check if a model directory has the standardized benchmark structure."""
    model_path = Path(model_dir)
    results = {}

    # Define expected benchmark directories
    expected_dirs = [
        model_path / "benchmarks",
        model_path / "benchmarks" / "unit",
        model_path / "benchmarks" / "integration",
        model_path / "benchmarks" / "performance",
    ]

    # Check if directories exist
    for expected_dir in expected_dirs:
        results[str(expected_dir)] = expected_dir.exists()

    # Define expected benchmark files
    expected_files = [
        model_path / "benchmarks" / "unit" / "benchmark_accuracy.py",
        model_path / "benchmarks" / "integration" / "benchmark_comparison.py",
        model_path / "benchmarks" / "performance" / "benchmark_inference_speed.py",
    ]

    # Check if files exist
    for expected_file in expected_files:
        results[str(expected_file)] = expected_file.exists()

    return results


def validate_benchmark_syntax(file_path: str) -> Tuple[bool, str]:
    """Validate the syntax of a benchmark file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Try to compile the file to check for syntax errors
        compile(content, file_path, "exec")
        return True, "Valid syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"


def check_model_conformance(model_dir: str) -> Dict[str, any]:
    """Comprehensively check if a model conforms to the benchmark standards."""
    results = {
        "model_dir": model_dir,
        "structure_valid": True,
        "syntax_errors": [],
        "missing_elements": [],
        "extra_validation": {},
    }

    # Check structure
    structure_results = check_benchmark_structure(model_dir)

    for path, exists in structure_results.items():
        if not exists:
            results["missing_elements"].append(path)
            results["structure_valid"] = False

    # Validate syntax of existing benchmark files
    model_path = Path(model_dir)
    benchmark_files = [
        model_path / "benchmarks" / "unit" / "benchmark_accuracy.py",
        model_path / "benchmarks" / "integration" / "benchmark_comparison.py",
        model_path / "benchmarks" / "performance" / "benchmark_inference_speed.py",
    ]

    for benchmark_file in benchmark_files:
        if benchmark_file.exists():
            is_valid, message = validate_benchmark_syntax(str(benchmark_file))
            if not is_valid:
                results["syntax_errors"].append(
                    {"file": str(benchmark_file), "error": message}
                )

    # Additional validation: check if benchmark files import the standardized interface
    for benchmark_file in benchmark_files:
        if benchmark_file.exists():
            with open(benchmark_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Check for imports of standardized benchmark interface
            has_standard_import = (
                "benchmark_interface" in content
                or "BaseBenchmark" in content
                or "BenchmarkResult" in content
            )

            results["extra_validation"][str(benchmark_file)] = {
                "has_standard_imports": has_standard_import
            }

    return results


def main():
    """Main function to run the verification."""
    print("=" * 80)
    print("BENCHMARK STRUCTURE VERIFICATION")
    print("=" * 80)

    # Find all model directories
    model_dirs = find_model_directories()
    print(f"Found {len(model_dirs)} model directories to verify:\n")

    for model_dir in model_dirs:
        print(f"- {model_dir}")

    print(f"\nVerifying benchmark structure compliance...\n")

    all_results = []
    compliant_models = 0
    non_compliant_models = 0

    for model_dir in model_dirs:
        print(f"\nChecking {Path(model_dir).name}...")
        result = check_model_conformance(model_dir)
        all_results.append(result)

        if result["structure_valid"] and len(result["syntax_errors"]) == 0:
            print(f"  [PASS] Compliant")
            compliant_models += 1
        else:
            print(f"  [FAIL] Non-compliant")
            non_compliant_models += 1

            if result["missing_elements"]:
                print(f"    Missing elements:")
                for element in result["missing_elements"]:
                    print(f"      - {element}")

            if result["syntax_errors"]:
                print(f"    Syntax errors:")
                for error in result["syntax_errors"]:
                    print(f"      - {error['file']}: {error['error']}")

    # Summary
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total models checked: {len(model_dirs)}")
    print(f"Compliant models: {compliant_models}")
    print(f"Non-compliant models: {non_compliant_models}")

    if non_compliant_models == 0:
        print(
            f"\n[SUCCESS] All models comply with the standardized benchmark structure!"
        )
        return 0
    else:
        print(
            f"\n[WARNING] {non_compliant_models} model(s) need to be updated to comply with the standard."
        )

        # Suggest remediation steps
        print(f"\nTo fix non-compliant models, ensure each model has:")
        print(f"  1. benchmarks/unit/benchmark_accuracy.py")
        print(f"  2. benchmarks/integration/benchmark_comparison.py")
        print(f"  3. benchmarks/performance/benchmark_inference_speed.py")
        print(f"  4. Proper imports from src.inference_pio.common.benchmark_interface")
        print(f"  5. Valid Python syntax")

        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
