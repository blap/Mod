"""
Test and Benchmark Runner for Qwen3-VL Model Components

This script runs all standardized tests and benchmarks for the Qwen3-VL model components
to ensure quality, performance, and consistency across the codebase.
"""

import sys
import os
import subprocess
import pytest
from pathlib import Path
import torch

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_pytest_tests():
    """Run pytest-based standardized tests."""
    print("="*80)
    print("RUNNING STANDARDIZED PYTEST TESTS")
    print("="*80)
    
    test_file = project_root / "tests" / "standardized_test_suite.py"
    
    if test_file.exists():
        # Run pytest with verbose output and coverage
        result = pytest.main([
            str(test_file),
            "-v",  # Verbose
            "--tb=short",  # Short traceback
            "-x",  # Stop on first failure
            "--disable-warnings"  # Suppress warnings
        ])
        
        print(f"\nPytest result: {'PASSED' if result == 0 else 'FAILED'} (exit code: {result})")
        return result == 0
    else:
        print(f"Test file not found: {test_file}")
        return False


def run_unit_tests():
    """Run unit tests from the unit test directory."""
    print("\n" + "="*80)
    print("RUNNING UNIT TESTS")
    print("="*80)
    
    unit_test_dir = project_root / "tests" / "unit"
    
    if unit_test_dir.exists():
        result = pytest.main([
            str(unit_test_dir),
            "-v",  # Verbose
            "--tb=short",  # Short traceback
            "-x",  # Stop on first failure
            "-k", "test_"  # Only run functions starting with test_
        ])
        
        print(f"\nUnit tests result: {'PASSED' if result == 0 else 'FAILED'} (exit code: {result})")
        return result == 0
    else:
        print(f"Unit test directory not found: {unit_test_dir}")
        return False


def run_integration_tests():
    """Run integration tests."""
    print("\n" + "="*80)
    print("RUNNING INTEGRATION TESTS")
    print("="*80)
    
    integration_test_dir = project_root / "tests" / "integration"
    
    if integration_test_dir.exists():
        result = pytest.main([
            str(integration_test_dir),
            "-v",  # Verbose
            "--tb=short",  # Short traceback
            "-x",  # Stop on first failure
            "-k", "test_"  # Only run functions starting with test_
        ])
        
        print(f"\nIntegration tests result: {'PASSED' if result == 0 else 'FAILED'} (exit code: {result})")
        return result == 0
    else:
        print(f"Integration test directory not found: {integration_test_dir}")
        return False


def run_benchmarks():
    """Run benchmark tests."""
    print("\n" + "="*80)
    print("RUNNING BENCHMARKS")
    print("="*80)
    
    benchmark_script = project_root / "benchmarks" / "standardized_benchmark_suite.py"
    
    if benchmark_script.exists():
        try:
            # Import and run the benchmark suite directly
            import benchmarks.standardized_benchmark_suite as benchmark_suite
            
            # Run the benchmarks
            results = benchmark_suite.run_standardized_benchmarks()
            
            print("\nBenchmark execution completed successfully!")
            return True
        except Exception as e:
            print(f"Error running benchmarks: {e}")
            return False
    else:
        print(f"Benchmark script not found: {benchmark_script}")
        return False


def run_model_specific_tests():
    """Run model-specific tests."""
    print("\n" + "="*80)
    print("RUNNING MODEL-SPECIFIC TESTS")
    print("="*80)
    
    model_test_dirs = [
        project_root / "tests" / "models",
        project_root / "tests" / "language", 
        project_root / "tests" / "vision",
        project_root / "tests" / "multimodal"
    ]
    
    all_passed = True
    
    for test_dir in model_test_dirs:
        if test_dir.exists():
            print(f"\nRunning tests in {test_dir.name} directory...")
            result = pytest.main([
                str(test_dir),
                "-v",  # Verbose
                "--tb=short",  # Short traceback
                "-x",  # Stop on first failure
                "-k", "test_"  # Only run functions starting with test_
            ])
            
            dir_passed = result == 0
            all_passed = all_passed and dir_passed
            print(f"{test_dir.name} tests: {'PASSED' if dir_passed else 'FAILED'} (exit code: {result})")
        else:
            print(f"Model test directory not found: {test_dir}")
    
    return all_passed


def check_environment():
    """Check that the environment is properly set up."""
    print("\n" + "="*80)
    print("CHECKING ENVIRONMENT")
    print("="*80)
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check if required packages are installed
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not found - this may cause some tests to fail")
    
    try:
        import datasets
        print(f"Datasets version: {datasets.__version__}")
    except ImportError:
        print("Datasets not found - this may cause some tests to fail")
    
    return True


def main():
    """Main function to run all tests and benchmarks."""
    print("Starting Qwen3-VL Model Components Test and Benchmark Runner")
    print("="*80)
    
    # Check environment
    env_ok = check_environment()
    if not env_ok:
        print("Environment check failed - continuing anyway...")
    
    # Run all test categories
    results = {}
    
    results['standardized_tests'] = run_pytest_tests()
    results['unit_tests'] = run_unit_tests()
    results['integration_tests'] = run_integration_tests()
    results['model_specific_tests'] = run_model_specific_tests()
    results['benchmarks'] = run_benchmarks()
    
    # Summary
    print("\n" + "="*80)
    print("TEST AND BENCHMARK RUNNER SUMMARY")
    print("="*80)
    
    for category, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{category.replace('_', ' ').title()}: {status}")
    
    # Overall result
    all_passed = all(results.values())
    print(f"\nOverall Result: {'ALL TESTS AND BENCHMARKS PASSED' if all_passed else 'SOME TESTS OR BENCHMARKS FAILED'}")
    
    # Count passed/failed
    passed_count = sum(results.values())
    total_count = len(results)
    print(f"Passed: {passed_count}/{total_count}")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)