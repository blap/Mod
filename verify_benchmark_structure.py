"""
Validation script to verify the benchmark structure for Qwen3 models.

This script validates that the benchmark structure is correctly implemented
for both qwen3_0_6b and qwen3_coder_next models.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def validate_benchmark_structure():
    """Validate the benchmark structure for both models."""
    print("Validating benchmark structure for Qwen3 models...")
    
    # Test imports for qwen3_0_6b
    try:
        from src.inference_pio.models.qwen3_0_6b.benchmarks import (
            Qwen306BUnitBenchmark,
            Qwen306BIntegrationBenchmark,
            Qwen306BPerformanceBenchmark
        )
        print("[OK] qwen3_0_6b benchmark imports successful")
    except ImportError as e:
        print(f"[ERROR] Failed to import qwen3_0_6b benchmarks: {e}")
        return False

    # Test imports for qwen3_coder_next
    try:
        from src.inference_pio.models.qwen3_coder_next.benchmarks import (
            Qwen3CoderNextUnitBenchmark,
            Qwen3CoderNextIntegrationBenchmark,
            Qwen3CoderNextPerformanceBenchmark
        )
        print("[OK] qwen3_coder_next benchmark imports successful")
    except ImportError as e:
        print(f"[ERROR] Failed to import qwen3_coder_next benchmarks: {e}")
        return False
    
    # Validate directory structure for qwen3_0_6b
    qwen3_0_6b_benchmarks_dir = Path("src/inference_pio/models/qwen3_0_6b/benchmarks")
    expected_dirs = ["unit", "integration", "performance"]
    
    for dir_name in expected_dirs:
        dir_path = qwen3_0_6b_benchmarks_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"[OK] qwen3_0_6b/{dir_name}/ directory exists")
        else:
            print(f"[ERROR] qwen3_0_6b/{dir_name}/ directory missing")
            return False

    # Validate directory structure for qwen3_coder_next
    qwen3_coder_next_benchmarks_dir = Path("src/inference_pio/models/qwen3_coder_next/benchmarks")

    for dir_name in expected_dirs:
        dir_path = qwen3_coder_next_benchmarks_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"[OK] qwen3_coder_next/{dir_name}/ directory exists")
        else:
            print(f"[ERROR] qwen3_coder_next/{dir_name}/ directory missing")
            return False

    # Validate __init__.py files exist
    init_files = [
        qwen3_0_6b_benchmarks_dir / "__init__.py",
        qwen3_0_6b_benchmarks_dir / "unit" / "__init__.py",
        qwen3_0_6b_benchmarks_dir / "integration" / "__init__.py",
        qwen3_0_6b_benchmarks_dir / "performance" / "__init__.py",
        qwen3_coder_next_benchmarks_dir / "__init__.py",
        qwen3_coder_next_benchmarks_dir / "unit" / "__init__.py",
        qwen3_coder_next_benchmarks_dir / "integration" / "__init__.py",
        qwen3_coder_next_benchmarks_dir / "performance" / "__init__.py"
    ]

    for init_file in init_files:
        if init_file.exists():
            print(f"[OK] {init_file} exists")
        else:
            print(f"[ERROR] {init_file} missing")
            return False

    # Validate benchmark files exist
    benchmark_files = [
        qwen3_0_6b_benchmarks_dir / "unit" / "benchmark_accuracy_standardized.py",
        qwen3_0_6b_benchmarks_dir / "integration" / "benchmark_integration_standardized.py",
        qwen3_0_6b_benchmarks_dir / "performance" / "benchmark_performance_standardized.py",
        qwen3_coder_next_benchmarks_dir / "unit" / "benchmark_accuracy_standardized.py",
        qwen3_coder_next_benchmarks_dir / "integration" / "benchmark_integration_standardized.py",
        qwen3_coder_next_benchmarks_dir / "performance" / "benchmark_performance_standardized.py"
    ]

    for benchmark_file in benchmark_files:
        if benchmark_file.exists():
            print(f"[OK] {benchmark_file} exists")
        else:
            print(f"[ERROR] {benchmark_file} missing")
            return False

    print("\n[SUCCESS] All validations passed! Benchmark structure is correctly implemented.")
    return True

if __name__ == "__main__":
    success = validate_benchmark_structure()
    if not success:
        sys.exit(1)