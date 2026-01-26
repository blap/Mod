"""
Test script to verify the enhanced benchmark system functionality.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
from pathlib import Path
import json
from datetime import datetime

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_enhanced_benchmark_utils():
    """Test the enhanced benchmark utilities."""
    print("Testing enhanced benchmark utilities...")
    
    try:
        from enhanced_benchmark_utils import (
            calculate_percentage_difference,
            setup_environment,
            apply_modifications,
            remove_modifications
        )
        
        # Test percentage difference calculation
        assert calculate_percentage_difference(100, 120) == 20.0, "Percentage difference calculation failed"
        assert calculate_percentage_difference(100, 80) == -20.0, "Negative percentage difference calculation failed"
        assert calculate_percentage_difference(0, 0) == 0.0, "Zero division case failed"
        
        print("[PASS] Enhanced benchmark utilities test passed")
        return True
    except Exception as e:
        print(f"[FAIL] Enhanced benchmark utilities test failed: {e}")
        return False


def test_file_existence():
    """Test that all required files exist."""
    print("Testing file existence...")
    
    required_files = [
        "enhanced_benchmark_utils.py",
        "ENHANCED_BENCHMARK_CONTROLLER.py",
        "enhanced_benchmark_accuracy.py",
        "enhanced_benchmark_inference_speed.py",
        "enhanced_benchmark_memory_usage.py",
        "enhanced_benchmark_optimization_impact.py",
        "MASTER_ENHANCED_BENCHMARK_EXECUTOR.py",
        "ENHANCED_BENCHMARK_SYSTEM_README.md"
    ]
    
    all_exist = True
    for file in required_files:
        if not Path(file).exists():
            print(f"[FAIL] Missing file: {file}")
            all_exist = False
        else:
            print(f"[PASS] Found file: {file}")

    if all_exist:
        print("[PASS] All required files exist")
    else:
        print("[FAIL] Some required files are missing")

    return all_exist


def main():
    """Main test function."""
    print("Testing Enhanced Benchmark System")
    print("="*50)
    
    tests = [
        test_enhanced_benchmark_utils,
        test_file_existence
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Enhanced benchmark system is ready.")
        return True
    else:
        print("[ERROR] Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)