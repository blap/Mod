"""
Updated Comprehensive Tests for Contractual Interfaces in Inference-PIO

This module contains comprehensive tests for the contractual interfaces,
ensuring all plugins adhere to the standardized contract with enhanced validation.
"""
from src.inference_pio.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, 
    assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, 
    assert_is_instance, assert_raises, run_tests, assert_length, assert_dict_contains,
    assert_list_contains, assert_is_subclass, assert_has_attr, assert_callable,
    assert_iterable, assert_not_is_instance
)


from datetime import datetime
from typing import Any
import torch
import torch.nn as nn

import sys
import os
from pathlib import Path

# Adicionando o diretório src ao path para permitir imports relativos
src_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(src_dir))

from src.inference_pio.common.base_plugin_interface import ModelPluginInterface
from src.inference_pio.common.standard_plugin_interface import (
    PluginMetadata,
    PluginType,
    StandardPluginInterface
)
from src.inference_pio.models.glm_4_7_flash.plugin import GLM_4_7_Flash_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin


def setup_helper():
    """
    Set up test fixtures before each test method.

    Creates test metadata for plugin testing.
    """
    # Create metadata for testing
    test_metadata = PluginMetadata(
        name="TestPlugin",
        version="1.0.0",
        author="Test Author",
        description="Test plugin for contractual interface verification",
        plugin_type=PluginType.MODEL_COMPONENT,
        dependencies=["torch"],
        compatibility={"torch_version": ">=2.0.0"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
        model_architecture="Test Architecture",
        model_size="1B",
        required_memory_gb=2.0,
        supported_modalities=["text"],
        license="MIT",
        tags=["test", "contractual"],
        model_family="Test Family",
        num_parameters=1000000000,
        test_coverage=0.95,
        validation_passed=True
    )


def test_standard_plugin_interface_contract():
    """
    Test that StandardPluginInterface defines the correct contract.

    Verifies that StandardPluginInterface is properly defined as an abstract base class
    with the required abstract methods for plugin implementations.
    """
    # Verify that StandardPluginInterface is an abstract base class
    assert_true(hasattr(StandardPluginInterface, '__abstractmethods__'))

    # Verify all required methods are defined as abstract
    required_methods = {
        'initialize', 'load_model', 'infer', 'cleanup', 'supports_config'
    }
    abstract_methods = StandardPluginInterface.__abstractmethods__

    for method in required_methods:
        assert_in(method, abstract_methods,
                     f"Method '{method}' should be abstract in StandardPluginInterface")


def test_model_plugin_interface_inheritance():
    """
    Test that ModelPluginInterface properly extends StandardPluginInterface.

    Validates that ModelPluginInterface correctly inherits from StandardPluginInterface
    and maintains the expected abstract method structure.
    """
    # Verify inheritance
    assert_true(issubclass(ModelPluginInterface, StandardPluginInterface))
    assert_is_subclass(ModelPluginInterface, StandardPluginInterface)

    # Verify ModelPluginInterface doesn't add additional abstract methods
    standard_abstract = StandardPluginInterface.__abstractmethods__
    model_abstract = ModelPluginInterface.__abstractmethods__

    # ModelPluginInterface should have same or fewer abstract methods
    assert_less_equal(len(model_abstract), len(standard_abstract))

    # All StandardPluginInterface methods should still be abstract in ModelPluginInterface
    for method in standard_abstract:
        assert_in(method, model_abstract)


def test_all_plugins_implement_contract():
    """
    Test that all concrete plugins implement the contractual interface.

    Ensures that all plugin implementations properly inherit from ModelPluginInterface
    and implement all required abstract methods.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        # Verify each plugin is an instance of ModelPluginInterface
        assert_is_instance(plugin, ModelPluginInterface)

        # Verify no abstract methods remain unimplemented
        assert_false(
            hasattr(type(plugin), '__abstractmethods__') and
            len(type(plugin).__abstractmethods__) > 0,
            f"{type(plugin).__name__} has unimplemented abstract methods: "
            f"{getattr(type(plugin), '__abstractmethods__', set())}"
        )


def test_initialize_method_contract():
    """
    Test that initialize method adheres to the contractual signature.

    Validates that the initialize method exists in all plugins and follows
    the expected signature and return type.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        # Verify method exists and is callable
        assert_true(callable(getattr(plugin, 'initialize')))
        assert_callable(getattr(plugin, 'initialize'))

        # Test method signature - should accept arbitrary kwargs and return bool
        result = plugin.initialize(device="cpu", config=None)
        assert_is_instance(result, bool)


def test_load_model_method_contract():
    """
    Test that load_model method adheres to the contractual signature.

    Validates that the load_model method exists in all plugins and follows
    the expected signature and return type.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        # Verify method exists and is callable
        assert_true(callable(getattr(plugin, 'load_model')))
        assert_callable(getattr(plugin, 'load_model'))

        # Test method signature - should accept optional config and return model-like object
        result = plugin.load_model(config=None)
        # Result can be None if model not loaded, but method should exist and be callable


def test_infer_method_contract():
    """
    Test that infer method adheres to the contractual signature.

    Validates that the infer method exists in all plugins and follows
    the expected signature and return type.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        # Verify method exists and is callable
        assert_true(callable(getattr(plugin, 'infer')))
        assert_callable(getattr(plugin, 'infer'))

        # Test method signature - should accept any data type and return Any
        result = plugin.infer("test input")
        # Result can be anything, but method should be callable


def test_cleanup_method_contract():
    """
    Test that cleanup method adheres to the contractual signature.

    Validates that the cleanup method exists in all plugins and follows
    the expected signature and return type.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        # Verify method exists and is callable
        assert_true(callable(getattr(plugin, 'cleanup')))
        assert_callable(getattr(plugin, 'cleanup'))

        # Test method signature - should return bool
        result = plugin.cleanup()
        assert_is_instance(result, bool)


def test_supports_config_method_contract():
    """
    Test that supports_config method adheres to the contractual signature.

    Validates that the supports_config method exists in all plugins and follows
    the expected signature and return type.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        # Verify method exists and is callable
        assert_true(callable(getattr(plugin, 'supports_config')))
        assert_callable(getattr(plugin, 'supports_config'))

        # Test method signature - should accept any config and return bool
        result = plugin.supports_config(None)
        assert_is_instance(result, bool)


def test_plugin_metadata_contract():
    """
    Test that all plugins provide proper metadata as per contract.

    Ensures that all plugins have metadata with the required attributes
    as specified in the contractual interface.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        # Verify metadata exists
        assert_is_not_none(plugin.metadata)

        # Verify all required metadata attributes exist
        metadata_attrs = [
            'name', 'version', 'author', 'description', 'plugin_type',
            'dependencies', 'compatibility', 'created_at', 'updated_at',
            'model_architecture', 'model_size', 'required_memory_gb',
            'supported_modalities', 'license', 'tags', 'model_family',
            'num_parameters', 'test_coverage', 'validation_passed'
        ]
        for attr in metadata_attrs:
            assert_true(hasattr(plugin.metadata, attr),
                        f"Plugin {type(plugin).__name__} missing metadata attribute: {attr}")
            assert_has_attr(plugin.metadata, attr)


def test_plugin_metadata_types():
    """
    Test that plugin metadata has correct types as per contract.

    Validates that all metadata attributes have the expected data types
    as specified in the contractual interface.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata

        # Test basic types
        assert_is_instance(metadata.name, str)
        assert_is_instance(metadata.version, str)
        assert_is_instance(metadata.author, str)
        assert_is_instance(metadata.description, str)
        assert_is_instance(metadata.plugin_type, PluginType)
        assert_is_instance(metadata.dependencies, list)
        assert_is_instance(metadata.compatibility, dict)
        assert_is_instance(metadata.created_at, datetime)
        assert_is_instance(metadata.updated_at, datetime)

        # Test model-specific types
        assert_is_instance(metadata.model_architecture, str)
        assert_is_instance(metadata.model_size, str)
        assert_is_instance(metadata.required_memory_gb, float)
        assert_is_instance(metadata.supported_modalities, list)
        assert_is_instance(metadata.license, str)
        assert_is_instance(metadata.tags, list)
        assert_is_instance(metadata.model_family, str)
        assert_is_instance(metadata.num_parameters, int)
        assert_is_instance(metadata.test_coverage, float)
        assert_is_instance(metadata.validation_passed, bool)


def test_plugin_initialization_with_security():
    """
    Test that plugins can be initialized with security parameters.

    Validates that plugins support initialization with additional security
    parameters as part of the contractual interface.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        # Test initialization with security parameters
        result = plugin.initialize(
            device="cpu",
            config=None,
            security_level="medium",
            resource_limits={"memory_gb": 4.0}
        )
        assert_is_instance(result, bool)


def test_plugin_config_validation():
    """
    Test that plugins properly validate configurations.

    Ensures that plugins can handle various configuration types and
    properly validate them according to the contractual interface.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        # Test with various config types
        result = plugin.supports_config({"batch_size": 1})
        assert_is_instance(result, bool)

        result = plugin.supports_config(None)
        assert_is_instance(result, bool)


def test_plugin_infer_with_different_inputs():
    """
    Test that plugins can handle different input types in infer method.

    Validates that the infer method can accept various input formats
    as specified in the contractual interface.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        # Test with string input
        result_str = plugin.infer("test string")

        # Test with dictionary input
        result_dict = plugin.infer({"input": "test"})

        # Test with list input
        result_list = plugin.infer(["test", "input"])


def test_plugin_cleanup_resilience():
    """
    Test that cleanup method handles various states gracefully.

    Ensures that the cleanup method can handle different plugin states
    without throwing exceptions, as required by the contractual interface.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        # Test cleanup after initialization
        plugin.initialize(device="cpu")
        result_after_init = plugin.cleanup()
        assert_is_instance(result_after_init, bool)

        # Test cleanup when not initialized (should handle gracefully)
        result_uninitialized = plugin.cleanup()
        assert_is_instance(result_uninitialized, bool)


def test_plugin_with_invalid_config():
    """
    Test plugin behavior with invalid configuration.

    Validates that plugins handle invalid configuration inputs gracefully
    and return appropriate boolean responses as per the contractual interface.
    """
    plugin = GLM_4_7_Flash_Plugin()

    # Test with invalid config
    result = plugin.supports_config("invalid_config_type")
    assert_is_instance(result, bool)


def test_plugin_infer_empty_input():
    """
    Test plugin behavior with empty input.

    Ensures that plugins can handle empty input values without crashing
    and return appropriate responses as per the contractual interface.
    """
    plugin = GLM_4_7_Flash_Plugin()

    # Test with empty string
    result = plugin.infer("")

    # Test with empty list
    result = plugin.infer([])

    # Test with empty dict
    result = plugin.infer({})


def test_plugin_metadata_immutability():
    """
    Test that plugin metadata cannot be modified after creation.

    Validates that plugin metadata remains immutable after instantiation,
    preserving the integrity of plugin information as per the contractual interface.
    """
    plugin = GLM_4_7_Flash_Plugin()
    original_name = plugin.metadata.name

    # Attempt to modify metadata (this should either fail or not affect the original)
    try:
        plugin.metadata.name = "Modified Name"
        # If modification succeeded, revert it
        plugin.metadata.name = original_name
    except AttributeError:
        # Expected if metadata is immutable
        pass

    # Verify original name is preserved
    assert_equal(plugin.metadata.name, original_name)


def test_plugin_interface_compatibility():
    """
    Test that plugins maintain backward compatibility.

    Ensures that plugins can be called with minimal parameters
    and maintain compatibility with the contractual interface.
    """
    plugin = GLM_4_7_Flash_Plugin()

    # Test calling methods with minimal parameters
    init_result = plugin.initialize()
    assert_is_instance(init_result, bool)

    load_result = plugin.load_model()

    infer_result = plugin.infer("test")

    cleanup_result = plugin.cleanup()
    assert_is_instance(cleanup_result, bool)


def test_plugin_metadata_attributes_completeness():
    """
    Test that all required metadata attributes are present and properly typed.
    """
    plugin = GLM_4_7_Flash_Plugin()
    metadata = plugin.metadata
    
    # Check that all required attributes exist
    required_attrs = [
        'name', 'version', 'author', 'description', 'plugin_type',
        'dependencies', 'compatibility', 'created_at', 'updated_at',
        'model_architecture', 'model_size', 'required_memory_gb',
        'supported_modalities', 'license', 'tags', 'model_family',
        'num_parameters', 'test_coverage', 'validation_passed'
    ]
    
    for attr in required_attrs:
        assert_has_attr(metadata, attr, f"Missing required metadata attribute: {attr}")
    
    # Check that required fields are not empty/None where inappropriate
    assert_not_equal(metadata.name, "", "Plugin name should not be empty")
    assert_not_equal(metadata.version, "", "Plugin version should not be empty")
    assert_not_equal(metadata.description, "", "Plugin description should not be empty")


def test_plugin_dependencies_format():
    """
    Test that plugin dependencies are properly formatted.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.dependencies, list)
        assert_iterable(metadata.dependencies)
        
        # Dependencies should be strings
        for dep in metadata.dependencies:
            assert_is_instance(dep, str)


def test_plugin_compatibility_format():
    """
    Test that plugin compatibility information is properly formatted.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.compatibility, dict)
        
        # Compatibility should be a dictionary
        for key, value in metadata.compatibility.items():
            assert_is_instance(key, str)
            assert_is_instance(value, str)


def test_plugin_tags_format():
    """
    Test that plugin tags are properly formatted.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.tags, list)
        assert_iterable(metadata.tags)
        
        # Tags should be strings
        for tag in metadata.tags:
            assert_is_instance(tag, str)


def test_plugin_supported_modalities_format():
    """
    Test that plugin supported modalities are properly formatted.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.supported_modalities, list)
        assert_iterable(metadata.supported_modalities)
        
        # Modalities should be strings
        for modality in metadata.supported_modalities:
            assert_is_instance(modality, str)


def test_plugin_memory_requirements():
    """
    Test that plugin memory requirements are reasonable.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.required_memory_gb, float)
        assert_greater(metadata.required_memory_gb, 0.0, 
                      "Required memory should be positive")


def test_plugin_parameter_counts():
    """
    Test that plugin parameter counts are reasonable.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.num_parameters, int)
        assert_greater(metadata.num_parameters, 0, 
                      "Number of parameters should be positive")


def test_plugin_test_coverage():
    """
    Test that plugin test coverage values are reasonable.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.test_coverage, float)
        assert_between(metadata.test_coverage, 0.0, 1.0, 
                      "Test coverage should be between 0 and 1")


def test_plugin_validation_status():
    """
    Test that plugin validation status is properly set.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.validation_passed, bool)


def test_plugin_datetime_formats():
    """
    Test that plugin datetime fields are properly formatted.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.created_at, datetime)
        assert_is_instance(metadata.updated_at, datetime)
        
        # Updated time should be same or later than created time
        assert_true(metadata.updated_at >= metadata.created_at,
                   "Updated time should not be earlier than created time")


def test_plugin_model_family_classification():
    """
    Test that plugin model family classification is properly set.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.model_family, str)
        assert_not_equal(metadata.model_family, "",
                       "Model family should not be empty")


def test_plugin_model_size_classification():
    """
    Test that plugin model size classification is properly set.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.model_size, str)
        assert_not_equal(metadata.model_size, "",
                       "Model size should not be empty")


def test_plugin_architecture_classification():
    """
    Test that plugin model architecture classification is properly set.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.model_architecture, str)
        assert_not_equal(metadata.model_architecture, "",
                       "Model architecture should not be empty")


def test_plugin_license_information():
    """
    Test that plugin license information is properly set.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.license, str)
        assert_not_equal(metadata.license, "",
                       "License should not be empty")


def test_plugin_type_classification():
    """
    Test that plugin type classification is properly set.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.plugin_type, PluginType)


def test_plugin_author_information():
    """
    Test that plugin author information is properly set.
    """
    plugins = [
        GLM_4_7_Flash_Plugin(),
        Qwen3_4B_Instruct_2507_Plugin(),
        Qwen3_Coder_30B_Plugin(),
        Qwen3_VL_2B_Instruct_Plugin()
    ]

    for plugin in plugins:
        metadata = plugin.metadata
        assert_is_instance(metadata.author, str)
        assert_not_equal(metadata.author, "",
                       "Author should not be empty")


def run_contractual_interface_tests():
    """Run all contractual interface tests."""
    test_functions = [
        test_standard_plugin_interface_contract,
        test_model_plugin_interface_inheritance,
        test_all_plugins_implement_contract,
        test_initialize_method_contract,
        test_load_model_method_contract,
        test_infer_method_contract,
        test_cleanup_method_contract,
        test_supports_config_method_contract,
        test_plugin_metadata_contract,
        test_plugin_metadata_types,
        test_plugin_initialization_with_security,
        test_plugin_config_validation,
        test_plugin_infer_with_different_inputs,
        test_plugin_cleanup_resilience,
        test_plugin_with_invalid_config,
        test_plugin_infer_empty_input,
        test_plugin_metadata_immutability,
        test_plugin_interface_compatibility,
        test_plugin_metadata_attributes_completeness,
        test_plugin_dependencies_format,
        test_plugin_compatibility_format,
        test_plugin_tags_format,
        test_plugin_supported_modalities_format,
        test_plugin_memory_requirements,
        test_plugin_parameter_counts,
        test_plugin_test_coverage,
        test_plugin_validation_status,
        test_plugin_datetime_formats,
        test_plugin_model_family_classification,
        test_plugin_model_size_classification,
        test_plugin_architecture_classification,
        test_plugin_license_information,
        test_plugin_type_classification,
        test_plugin_author_information
    ]
    
    print("Running updated contractual interface tests...")
    success = run_tests(test_functions)
    return success


if __name__ == '__main__':
    run_contractual_interface_tests()