"""
Test script to verify the benchmarking solution works correctly.
This script runs a quick test to ensure the benchmarking infrastructure is properly set up.
"""

import sys
import time
from pathlib import Path

from tests.utils.test_utils import (
    assert_equal,
    assert_false,
    assert_greater,
    assert_in,
    assert_is_instance,
    assert_is_none,
    assert_is_not_none,
    assert_less,
    assert_not_equal,
    assert_not_in,
    assert_raises,
    assert_true,
    run_tests,
)

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark_utils import run_benchmark_for_model_category


def test_benchmark_infrastructure():
    """
    Test that the benchmarking infrastructure works correctly.
    """
    print("Testing benchmark infrastructure...")
    print("-" * 50)

    # Test with a single model and category to verify the setup
    model_name = "glm_4_7"
    category = "accuracy"

    print(f"Testing {category} benchmark for {model_name}")

    start_time = time.time()
    result = run_benchmark_for_model_category(model_name, category)
    end_time = time.time()

    print(f"Benchmark completed in {end_time - start_time:.2f} seconds")

    # Check if the basic structure is correct
    expected_keys = ["model", "category", "timestamp", "results", "status"]
    missing_keys = [key for key in expected_keys if key not in result]

    if missing_keys:
        print(f"❌ Missing keys in result: {missing_keys}")
        return False
    else:
        print(f"✅ All expected keys present in result")

    # Check if model and category match expectations
    if result["model"] != model_name:
        print(f"❌ Model mismatch: expected {model_name}, got {result['model']}")
        return False
    else:
        print(f"✅ Model name correct: {result['model']}")

    if result["category"] != category:
        print(f"❌ Category mismatch: expected {category}, got {result['category']}")
        return False
    else:
        print(f"✅ Category name correct: {result['category']}")

    # Check status field
    if result["status"] in ["success", "partial", "failed", "no_tests_found"]:
        print(f"✅ Status field valid: {result['status']}")
    else:
        print(f"❌ Invalid status: {result['status']}")
        return False

    # Check results field
    if isinstance(result["results"], list):
        print(f"✅ Results field is a list with {len(result['results'])} items")
    else:
        print(f"❌ Results field is not a list: {type(result['results'])}")
        return False

    print("-" * 50)
    print("✅ Benchmark infrastructure test PASSED")
    return True


def test_multiple_categories():
    """
    Test multiple categories to ensure they can be executed.
    """
    print("\nTesting multiple categories...")
    print("-" * 50)

    model_name = "glm_4_7"
    categories = ["accuracy", "inference_speed"]

    all_passed = True
    for category in categories:
        print(f"Testing {category}...")
        result = run_benchmark_for_model_category(model_name, category)

        if "model" in result and result["model"] == model_name:
            print(f"✅ {category} test passed")
        else:
            print(f"❌ {category} test failed: {result.get('error', 'Unknown error')}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("Running benchmark infrastructure tests...\n")

    test1_passed = test_benchmark_infrastructure()
    test2_passed = test_multiple_categories()

    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 All benchmark infrastructure tests PASSED!")
        print("The benchmarking solution is ready to use.")
    else:
        print("❌ Some tests FAILED!")
        print("Please check the benchmarking solution setup.")
    print("=" * 60)
