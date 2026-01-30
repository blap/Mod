import sys
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)

sys.path.insert(0, '.')

from src.inference_pio.models.glm_4_7_flash.benchmarks.performance.benchmark_inference_speed_comparison import GLM47InferenceSpeedComparisonBenchmark

# Create benchmark instance
benchmark = GLM47InferenceSpeedComparisonBenchmark()

# Temporarily disable monitoring for this test
benchmark.monitoring_active = False

print('Testing load_original_model function...')

try:
    # Try to load the original model
    print('Attempting to load original model...')
    original_plugin = benchmark.load_original_model()
    print('SUCCESS: Original model loaded successfully!')
    print(f'Model name: {original_plugin.metadata.name}')
    print(f'Model version: {original_plugin.metadata.version}')

    # Cleanup
    original_plugin.cleanup()
    print('Cleanup completed.')

except Exception as e:
    print(f'ERROR loading original model: {e}')
    import traceback
    traceback.print_exc()

print('\nTesting load_modified_model function...')

try:
    # Try to load the modified model
    print('Attempting to load modified model...')
    modified_plugin = benchmark.load_modified_model()
    print('SUCCESS: Modified model loaded successfully!')
    print(f'Model name: {modified_plugin.metadata.name}')
    print(f'Model version: {modified_plugin.metadata.version}')

    # Cleanup
    modified_plugin.cleanup()
    print('Cleanup completed.')

except Exception as e:
    print(f'ERROR loading modified model: {e}')
    import traceback
    traceback.print_exc()

print('\nAll tests completed!')