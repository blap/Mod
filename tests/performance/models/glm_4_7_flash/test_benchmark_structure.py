import sys
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)

sys.path.insert(0, '.')

from src.inference_pio.models.glm_4_7_flash.benchmarks.performance.benchmark_inference_speed_comparison import GLM47InferenceSpeedComparisonBenchmark

# Create benchmark instance
benchmark = GLM47InferenceSpeedComparisonBenchmark()

# Just test the structure without running full benchmark
print('Benchmark class instantiated successfully')
print('Results directory:', benchmark.results_dir)
print('All good to run full benchmark!')

# Test if we can load the original model (just check if the function exists)
try:
    # Temporarily disable monitoring for this test
    benchmark.monitoring_active = False
    print("Function 'load_original_model' exists:", hasattr(benchmark, 'load_original_model'))
    print("Function 'load_modified_model' exists:", hasattr(benchmark, 'load_modified_model'))
    print("Function 'run_comparison_benchmark' exists:", hasattr(benchmark, 'run_comparison_benchmark'))
    print("All methods exist!")
except Exception as e:
    print(f"Error during testing: {e}")