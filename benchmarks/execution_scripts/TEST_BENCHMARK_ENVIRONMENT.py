"""
Quick Test Script for Benchmark Environment

This script performs a quick test to ensure the benchmark environment
is properly configured and can run basic benchmark operations.
"""

import sys
import os
from pathlib import Path
import importlib.util

def test_imports():
    """Test that critical modules can be imported."""
    print("Testing critical imports...")
    
    modules_to_test = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("numpy", "NumPy"),
        ("psutil", "PS Util"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib")
    ]
    
    success_count = 0
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"[OK] {display_name} ({module_name}) - OK")
            success_count += 1
        except ImportError as e:
            print(f"[ERROR] {display_name} ({module_name}) - ERROR: {e}")

    print(f"\nImport tests: {success_count}/{len(modules_to_test)} successful")
    return success_count == len(modules_to_test)


def test_model_structure():
    """Test that the model structure is as expected."""
    print("\nTesting model structure...")
    
    src_path = Path("src/inference_pio/models")
    if not src_path.exists():
        print("✗ Model directory not found")
        return False
    
    expected_models = [
        "glm_4_7",
        "qwen3_4b_instruct_2507", 
        "qwen3_coder_30b",
        "qwen3_vl_2b"
    ]
    
    success_count = 0
    for model in expected_models:
        model_path = src_path / model
        if model_path.exists():
            print(f"[OK] Model {model} - OK")
            success_count += 1
        else:
            print(f"[ERROR] Model {model} - NOT FOUND")

    print(f"\nModel structure tests: {success_count}/{len(expected_models)} successful")
    return success_count == len(expected_models)


def test_benchmark_structure():
    """Test that benchmark structure is as expected."""
    print("\nTesting benchmark structure...")
    
    src_path = Path("src/inference_pio/models")
    expected_models = [
        "glm_4_7",
        "qwen3_4b_instruct_2507", 
        "qwen3_coder_30b",
        "qwen3_vl_2b"
    ]
    
    expected_categories = [
        "accuracy",
        "comparison", 
        "inference_speed",
        "memory_usage",
        "optimization_impact",
        "power_efficiency",
        "throughput"
    ]
    
    success_count = 0
    total_expected = len(expected_models) * len(expected_categories)
    
    for model in expected_models:
        for category in expected_categories:
            benchmark_path = src_path / model / "benchmarks" / f"benchmark_{category}.py"
            if benchmark_path.exists():
                success_count += 1
            else:
                print(f"✗ Missing: {model}/benchmarks/benchmark_{category}.py")
    
    print(f"\nBenchmark structure tests: {success_count}/{total_expected} files found")
    return success_count == total_expected


def test_scripts_exist():
    """Test that our newly created scripts exist."""
    print("\nTesting that created scripts exist...")
    
    expected_scripts = [
        "COMPREHENSIVE_BENCHMARK_CONTROLLER.py",
        "SETUP_BENCHMARK_ENVIRONMENT.py", 
        "DUAL_STATE_BENCHMARK_RUNNER.py",
        "MASTER_BENCHMARK_EXECUTOR.py"
    ]
    
    success_count = 0
    for script in expected_scripts:
        if Path(script).exists():
            print(f"[OK] {script} - OK")
            success_count += 1
        else:
            print(f"[ERROR] {script} - NOT FOUND")

    print(f"\nScript existence tests: {success_count}/{len(expected_scripts)} successful")
    return success_count == len(expected_scripts)


def main():
    """Main test function."""
    print("Inference-PIO Benchmark Environment Quick Test")
    print("="*50)
    
    print("\n1. Testing imports...")
    imports_ok = test_imports()
    
    print("\n2. Testing model structure...")
    structure_ok = test_model_structure()
    
    print("\n3. Testing benchmark structure...")
    benchmarks_ok = test_benchmark_structure()
    
    print("\n4. Testing created scripts...")
    scripts_ok = test_scripts_exist()
    
    print("\n" + "="*50)
    print("QUICK TEST SUMMARY")
    print("="*50)
    print(f"Imports: {'[OK]' if imports_ok else '[ERROR]'}")
    print(f"Model Structure: {'[OK]' if structure_ok else '[ERROR]'}")
    print(f"Benchmark Structure: {'[OK]' if benchmarks_ok else '[ERROR]'}")
    print(f"Created Scripts: {'[OK]' if scripts_ok else '[ERROR]'}")

    overall_success = all([imports_ok, structure_ok, benchmarks_ok, scripts_ok])

    if overall_success:
        print(f"\n[SUCCESS] All tests passed! Benchmark environment is ready.")
        print(f"You can now run the MASTER_BENCHMARK_EXECUTOR.py script to start the full benchmarking process.")
    else:
        print(f"\n[WARNING] Some tests failed. Please check the output above.")
        print(f"The benchmark environment may not be fully ready.")

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)