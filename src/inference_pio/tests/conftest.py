"""
Main test configuration file for the project.

This file contains the main pytest configuration and imports shared fixtures
and utilities to make them available across all test modules.
"""

from tests.base.benchmark_test_base import (
    BaseBenchmarkTest,
    ModelBenchmarkTest,
    SystemBenchmarkTest,
)
from tests.base.functional_test_base import (
    BaseFunctionalTest,
    ModelFunctionalTest,
    SystemFunctionalTest,
)
from tests.base.integration_test_base import (
    BaseIntegrationTest,
    ModelIntegrationTest,
    PipelineIntegrationTest,
)
from tests.base.regression_test_base import (
    BaseRegressionTest,
    FeatureRegressionTest,
    ModelRegressionTest,
)

# Import base test classes to make them available globally
from tests.base.unit_test_base import BaseUnitTest, ModelUnitTest, PluginUnitTest

# Import example fixtures for advanced scenarios
from tests.shared.fixtures.example_fixtures import (
    complex_test_environment,
    concurrency_test_setup,
    device_and_precision_config,
    expensive_resource,
    integration_test_components,
    mocked_network_operations,
    performance_test_data,
    plugin_factory,
    specialized_plugin_metadata,
    validated_test_state,
)

# Import additional fixtures from original module
from tests.shared.fixtures.plugin_fixtures import mock_plugin_interface_methods

# Import shared fixtures to make them available globally
from tests.shared.fixtures.standardized_fixtures import (  # Basic fixtures; Mock objects; Plugin-specific fixtures; Advanced fixtures
    mock_plugin_dependencies,
    mock_plugin_with_error_handling,
    mock_torch_model,
    parametrized_tensor_data,
    plugin_config_with_gpu,
    realistic_test_plugin,
    sample_config,
    sample_metadata,
    sample_plugin_manifest,
    sample_tensor_data,
    sample_text_data,
    temp_dir,
    temp_file,
)

# Import shared assertions to make them available globally
from tests.shared.utils.assertions import (
    assert_dict_contains_keys,
    assert_list_elements_type,
    assert_plugin_initialized,
    assert_plugin_interface_implemented,
    assert_response_format,
    assert_tensor_properties,
)

# Import shared utilities to make them available globally
from tests.shared.utils.test_utils import (
    assert_tensor_shape,
    assert_tensor_values_close,
    calculate_statistics,
    cleanup_temp_directory,
    compare_dicts,
    create_mock_model,
    create_mock_plugin_instance,
    create_sample_tensor_data,
    create_sample_text_data,
    create_temp_directory,
    ensure_directory_exists,
    extract_model_name_from_path,
    format_bytes,
    format_duration,
    generate_cache_key,
    get_timestamp,
    is_cache_valid,
    load_json_file,
    measure_execution_time,
    normalize_path_separators,
    sanitize_filename,
    save_json_file,
)


def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line(
        "markers", "model_specific: marks tests as specific to a particular model"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )
    config.addinivalue_line("markers", "regression: marks tests as regression tests")
    config.addinivalue_line("markers", "functional: marks tests as functional tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line(
        "markers", "performance: marks tests that evaluate performance"
    )
    config.addinivalue_line(
        "markers", "concurrency: marks tests that evaluate concurrent operations"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    # This hook can be used to modify test collection behavior
    # For now, we'll just log that the hook was called
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Modified {len(items)} test items during collection")
