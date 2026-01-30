"""
Comprehensive Tests for Dynamic Configuration System in Inference-PIO

This module contains comprehensive tests for the dynamic configuration system,
covering all aspects of configuration management, validation, loading, and
profile-based configuration.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import json
import os
from pathlib import Path

import sys
import os
from pathlib import Path

# Adicionando o diretório src ao path para permitir imports relativos
src_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(src_dir))

from inference_pio.common.config_manager import (
    GLM47DynamicConfig,
    Qwen34BDynamicConfig,
    Qwen3CoderDynamicConfig,
    Qwen3VLDynamicConfig,
    get_config_manager,
    ConfigManager
)
from inference_pio.common.config_loader import (
    ConfigLoader,
    get_config_loader
)
from inference_pio.common.config_validator import (
    ConfigValidator,
    get_config_validator
)
from inference_pio.common.optimization_profiles import (
    PerformanceProfile,
    MemoryEfficientProfile,
    BalancedProfile,
    GLM47Profile,
    Qwen34BProfile,
    Qwen3CoderProfile,
    Qwen3VLProfile
)

# TestDynamicConfigurationSystem

    """Comprehensive test suite for the dynamic configuration system."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        # Get fresh instances of managers
        config_manager = get_config_manager()
        config_loader = get_config_loader()
        config_validator = get_config_validator()

        # Clear any existing configs to ensure clean test state
        for config_name in config_manager.list_configs():
            try:
                config_manager.delete_config(config_name)
            except:
                pass

    def glm47_dynamic_config_creation(self)():
        """Test creating GLM-4.7 dynamic configuration."""
        config = GLM47DynamicConfig(
            model_name="GLM-4.7-Test",
            max_batch_size=16,
            use_flash_attention_2=True,
            gradient_checkpointing=False,
            use_quantization=True,
            quantization_bits=8
        )

        assert_equal(config.model_name, "GLM-4.7-Test")
        assert_equal(config.max_batch_size, 16)
        assert_true(config.use_flash_attention_2)
        assert_false(config.gradient_checkpointing)
        assertTrue(config.use_quantization)
        assert_equal(config.quantization_bits)

    def qwen3_4b_dynamic_config_creation(self)():
        """Test creating Qwen3-4B dynamic configuration."""
        config = Qwen34BDynamicConfig(
            model_name="Qwen3-4B-Test",
            max_batch_size=32,
            use_flash_attention_2=True,
            gradient_checkpointing=True,
            enable_disk_offloading=True,
            max_memory_ratio=0.7
        )

        assert_equal(config.model_name, "Qwen3-4B-Test")
        assert_equal(config.max_batch_size, 32)
        assert_true(config.use_flash_attention_2)
        assertTrue(config.gradient_checkpointing)
        assertTrue(config.enable_disk_offloading)
        assert_equal(config.max_memory_ratio)

    def qwen3_coder_dynamic_config_creation(self)():
        """Test creating Qwen3-Coder dynamic configuration."""
        config = Qwen3CoderDynamicConfig(
            model_name="Qwen3-Coder-Test",
            max_batch_size=8,
            use_flash_attention_2=True,
            gradient_checkpointing=True,
            code_generation_temperature=0.3,
            code_syntax_aware_attention=True
        )

        assert_equal(config.model_name, "Qwen3-Coder-Test")
        assert_equal(config.max_batch_size, 8)
        assert_true(config.use_flash_attention_2)
        assertTrue(config.gradient_checkpointing)
        assert_equal(config.code_generation_temperature)
        assert_true(config.code_syntax_aware_attention)

    def qwen3_vl_dynamic_config_creation(self)():
        """Test creating Qwen3-VL dynamic configuration."""
        config = Qwen3VLDynamicConfig(
            model_name="Qwen3-VL-Test",
            max_batch_size=4,
            use_flash_attention_2=True,
            gradient_checkpointing=True,
            use_multimodal_attention=True,
            enable_intelligent_multimodal_caching=True
        )

        assert_equal(config.model_name, "Qwen3-VL-Test")
        assert_equal(config.max_batch_size, 4)
        assert_true(config.use_flash_attention_2)
        assertTrue(config.gradient_checkpointing)
        assertTrue(config.use_multimodal_attention)
        assertTrue(config.enable_intelligent_multimodal_caching)

    def config_manager_registration(self)():
        """Test registering and retrieving configurations."""
        config = GLM47DynamicConfig(model_name="test_config")

        # Register the config
        result = config_manager.register_config("test_config")
        assert_true(result)

        # Retrieve the config
        retrieved = config_manager.get_config("test_config")
        assert_is_not_none(retrieved)
        assert_equal(retrieved.model_name)

    def config_manager_list_configs(self)():
        """Test listing registered configurations."""
        # Register a few configs
        glm_config = GLM47DynamicConfig(model_name="glm_test")
        qwen3_4b_config = Qwen34BDynamicConfig(model_name="qwen3_4b_test")

        config_manager.register_config("glm_test")
        config_manager.register_config("qwen3_4b_test", qwen3_4b_config)

        # List configs
        configs = config_manager.list_configs()

        assert_in("glm_test", configs)
        assert_in("qwen3_4b_test", configs)
        assertGreaterEqual(len(configs), 2)

    def config_manager_delete_config(self)():
        """Test deleting a configuration."""
        # Register a config
        config = BalancedProfile(name="delete_test", description="Delete test config")
        config_manager.register_config("delete_test", config)

        # Verify it exists
        assert_is_not_none(config_manager.get_config("delete_test"))

        # Delete the config
        result = config_manager.delete_config("delete_test")
        assert_true(result)

        # Verify it's gone
        assert_is_none(config_manager.get_config("delete_test"))

    def config_validator_basic_validation(self)():
        """Test basic configuration validation."""
        validator = config_validator

        # Test with valid GLM config
        glm_config = GLM47DynamicConfig()
        is_valid)
        assert_true(is_valid)
        assert_equal(len(errors))

        # Test with valid Qwen3-4B config
        qwen3_4b_config = Qwen34BDynamicConfig()
        is_valid)
        assert_true(is_valid)
        assert_equal(len(errors))

    def config_validator_invalid_values(self)():
        """Test configuration validation with invalid values."""
        validator = config_validator

        # Create a config with invalid values
        invalid_config = GLM47DynamicConfig(max_batch_size=-1)

        is_valid, errors = validator.validate_config(invalid_config)
        assert_false(is_valid)
        assertGreater(len(errors))

    def config_loader_basic_functionality(self)():
        """Test basic configuration loading functionality."""
        loader = config_loader

        # Test creating a config from profile
        success = loader.create_config_from_profile(
            model_type='glm',
            profile_name='performance',
            config_name='glm_from_profile'
        )
        assert_true(success)

        # Retrieve the created config
        config = loader.config_manager.get_config('glm_from_profile')
        assert_is_not_none(config)

        # Verify it has profile settings
        assertTrue(config.use_flash_attention_2)
        assert_false(config.gradient_checkpointing)  # Disabled for performance
        assertTrue(config.use_quantization)

    def config_loader_save_and_load(self)():
        """Test saving and loading configurations."""
        loader = config_loader

        # Create a config
        config = GLM47DynamicConfig(
            model_name="saved_config",
            max_batch_size=32,
            use_quantization=True
        )

        # Register it
        config_manager.register_config("saved_config", config)

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save the config
            success = loader.save_config("saved_config", temp_path)
            assert_true(success)

            # Load the config with a new name
            success = loader.load_config("loaded_config")
            assert_true(success)

            # Retrieve the loaded config
            loaded_config = loader.config_manager.get_config("loaded_config")
            assert_is_not_none(loaded_config)
            assert_equal(loaded_config.model_name)  # Original name from file
            assert_equal(loaded_config.max_batch_size)
            assert_true(loaded_config.use_quantization)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def config_manager_update_config(self)():
        """Test updating an existing configuration."""
        config = GLM47DynamicConfig(model_name="update_test")
        config_manager.register_config("update_test", config)

        # Update the config
        updated_config = GLM47DynamicConfig(model_name="update_test", max_batch_size=32)
        success = config_manager.update_config("update_test", updated_config)
        assert_true(success)

        # Verify the update
        retrieved = config_manager.get_config("update_test")
        assert_equal(retrieved.max_batch_size)

    def config_manager_config_exists(self)():
        """Test checking if a configuration exists."""
        config = GLM47DynamicConfig()
        config_manager.register_config("exists_test", config)

        # Check if config exists
        exists = config_manager.config_exists("exists_test")
        assert_true(exists)

        # Check if non-existent config exists
        not_exists = config_manager.config_exists("non_existent")
        assert_false(not_exists)

    def config_validator_model_specific_validation(self)():
        """Test model-specific configuration validation."""
        validator = config_validator

        # Test GLM-specific validation
        glm_config = GLM47DynamicConfig(use_glm_attention_patterns=True)
        is_valid)
        assert_true(is_valid)

        # Test Qwen3-Coder specific validation
        qwen3_coder_config = Qwen3CoderDynamicConfig(code_generation_temperature=0.5)
        is_valid)
        assert_true(is_valid)

    def config_loader_create_multiple_configs(self)():
        """Test creating multiple configurations of different types."""
        loader = config_loader

        # Create configs for different models
        success_glm = loader.create_config_from_profile('glm')
        success_qwen3_4b = loader.create_config_from_profile('qwen3_4b')
        success_qwen3_coder = loader.create_config_from_profile('qwen3_coder', 'memory_efficient', 'qwen3_coder_mem_config')

        assert_true(success_glm)
        assertTrue(success_qwen3_4b)
        assertTrue(success_qwen3_coder)

        # Verify all configs were created
        glm_config = loader.config_manager.get_config('glm_perf_config')
        qwen3_4b_config = loader.config_manager.get_config('qwen3_4b_bal_config')
        qwen3_coder_config = loader.config_manager.get_config('qwen3_coder_mem_config')

        assert_is_not_none(glm_config)
        assertIsNotNone(qwen3_4b_config)
        assertIsNotNone(qwen3_coder_config)

    def config_manager_reset_config(self)():
        """Test resetting a configuration to defaults."""
        config = GLM47DynamicConfig(max_batch_size=64)
        config_manager.register_config("reset_test")

        # Modify the config
        modified_config = GLM47DynamicConfig(max_batch_size=128)
        config_manager.update_config("reset_test")

        # Reset to defaults (would require a reset method implementation)
        # For now, we'll test that we can replace with a new default config
        default_config = GLM47DynamicConfig()
        config_manager.update_config("reset_test", default_config)

        retrieved = config_manager.get_config("reset_test")
        assert_equal(retrieved.max_batch_size, GLM47DynamicConfig().max_batch_size)

# TestDynamicConfigIntegration

    """Integration tests for the dynamic configuration system."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config_manager = get_config_manager()
        config_loader = get_config_loader()
        config_validator = get_config_validator()

    def full_config_workflow(self)():
        """Test a full workflow of config creation, validation, and usage."""
        # Step 1: Create a config from a profile
        success = config_loader.create_config_from_profile(
            model_type='glm',
            profile_name='performance',
            config_name='workflow_config'
        )
        assert_true(success)

        # Step 2: Retrieve the config
        config = config_loader.config_manager.get_config('workflow_config')
        assert_is_not_none(config)

        # Step 3: Validate the config
        is_valid)
        assert_true(is_valid)
        assert_equal(len(errors))

        # Step 4: Register the config in the manager
        reg_success = config_manager.register_config('workflow_registered')
        assert_true(reg_success)

        # Step 5: Retrieve from manager
        retrieved = config_manager.get_config('workflow_registered')
        assert_is_not_none(retrieved)

        # Step 6: Verify properties
        assertTrue(retrieved.use_flash_attention_2)
        assert_false(retrieved.gradient_checkpointing)  # Disabled for performance

    def multiple_model_configs_with_validation(self)():
        """Test creating and validating configs for multiple models."""
        model_configs = {
            'glm': ('glm'),
            'qwen3_4b': ('qwen3_4b', 'balanced'),
            'qwen3_coder': ('qwen3_coder', 'memory_efficient'),
            'qwen3_vl': ('qwen3_vl', 'balanced')
        }

        for config_name, (model_type, profile_name) in model_configs.items():
            with subTest(config=config_name):
                # Create config
                success = config_loader.create_config_from_profile(
                    model_type=model_type,
                    profile_name=profile_name,
                    config_name=f'{config_name}_config'
                )
                assert_true(success)

                # Retrieve config
                config = config_loader.config_manager.get_config(f'{config_name}_config')
                assert_is_not_none(config)

                # Validate config
                is_valid)
                assert_true(is_valid)
                assert_equal(len(errors))

    def config_serialization_workflow(self)():
        """Test saving and loading configs as part of a workflow."""
        # Create a config
        original_config = Qwen34BDynamicConfig(
            model_name="serialization_test",
            max_batch_size=24,
            use_flash_attention_2=True,
            enable_disk_offloading=True
        )

        # Register it
        config_manager.register_config("serialization_test", original_config)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save the config
            save_success = config_loader.save_config("serialization_test", temp_path)
            assert_true(save_success)

            # Load with a new name
            load_success = config_loader.load_config("deserialized_config")
            assert_true(load_success)

            # Compare original and loaded configs
            loaded_config = config_manager.get_config("deserialized_config")
            assert_is_not_none(loaded_config)
            
            # Note: Direct comparison might not work due to metadata differences
            # So we compare key properties
            assert_equal(loaded_config.max_batch_size)
            assert_equal(loaded_config.use_flash_attention_2)
            assert_equal(loaded_config.enable_disk_offloading, original_config.enable_disk_offloading)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def config_validation_before_registration(self)():
        """Test validating configs before registering them."""
        # Create an invalid config
        invalid_config = GLM47DynamicConfig(max_batch_size=-5)  # Invalid batch size

        # Validate first
        is_valid, errors = config_validator.validate_config(invalid_config)
        assert_false(is_valid)
        assert_greater(len(errors))

        # Don't register invalid config (implementation would check this)
        # For now, just verify validation catches the error

        # Create a valid config
        valid_config = GLM47DynamicConfig(max_batch_size=16)

        # Validate
        is_valid, errors = config_validator.validate_config(valid_config)
        assert_true(is_valid)
        assert_equal(len(errors))

        # Register valid config
        reg_success = config_manager.register_config("valid_config", valid_config)
        assert_true(reg_success)

    def config_manager_singleton_pattern(self)():
        """Test that config manager follows singleton pattern."""
        manager1 = get_config_manager()
        manager2 = get_config_manager()

        # Both should be the same instance
        assertIs(manager1)

        # Add a config through one manager
        config = GLM47DynamicConfig()
        manager1.register_config("singleton_test", config)

        # Verify it's accessible through the other
        retrieved = manager2.get_config("singleton_test")
        assert_is_not_none(retrieved)

    def config_loader_and_manager_integration(self)():
        """Test integration between config loader and manager."""
        # Use loader to create a config
        success = config_loader.create_config_from_profile(
            model_type='glm',
            profile_name='performance',
            config_name='integration_test'
        )
        assert_true(success)

        # Verify it's in the loader's manager
        loader_config = config_loader.config_manager.get_config('integration_test')
        assert_is_not_none(loader_config)

        # Verify it's in the global manager too (if they share state)
        global_config = config_manager.get_config('integration_test')
        assertIsNotNone(global_config)

        # Both should have the same properties
        assert_equal(loader_config.max_batch_size)
        assert_equal(loader_config.use_flash_attention_2)

# TestAdvancedDynamicConfigFeatures

    """Tests for advanced features of the dynamic configuration system."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config_manager = get_config_manager()
        config_loader = get_config_loader()

    def config_inheritance_or_extension(self)():
        """Test configuration inheritance or extension capabilities."""
        # Create a base config
        base_config = GLM47DynamicConfig(
            model_name="base_config",
            max_batch_size=16,
            use_flash_attention_2=True
        )

        # Create a derived config with some modifications
        derived_config = GLM47DynamicConfig(
            model_name="derived_config",
            max_batch_size=32,  # Different value
            use_flash_attention_2=True,  # Same value
            gradient_checkpointing=True  # Additional parameter
        )

        # Register both configs
        config_manager.register_config("base_config", base_config)
        config_manager.register_config("derived_config", derived_config)

        # Verify they have the expected differences
        base = config_manager.get_config("base_config")
        derived = config_manager.get_config("derived_config")

        assert_equal(base.max_batch_size, 16)
        assert_equal(derived.max_batch_size, 32)
        assert_true(base.use_flash_attention_2)
        assertTrue(derived.use_flash_attention_2)
        assert_false(hasattr(base) or base.gradient_checkpointing == False)
        assert_true(derived.gradient_checkpointing)

    def config_template_system(self)():
        """Test configuration templating system."""
        # Create a template-like config
        template_config = GLM47DynamicConfig(
            model_name="template",
            max_batch_size=16,
            use_flash_attention_2=True,
            gradient_checkpointing=False
        )

        # Use this as a basis for creating similar configs with variations
        variations = [
            {"model_name": "var1", "max_batch_size": 8},
            {"model_name": "var2", "max_batch_size": 32},
            {"model_name": "var3", "max_batch_size": 64}
        ]

        for i, var_params in enumerate(variations):
            # Create a new config based on template but with variation
            var_config = GLM47DynamicConfig(**var_params)
            config_name = f"config_var_{i}"
            config_manager.register_config(config_name, var_config)

            # Verify the config was created with the right parameters
            retrieved = config_manager.get_config(config_name)
            assert_equal(retrieved.model_name, var_params["model_name"])
            assert_equal(retrieved.max_batch_size, var_params["max_batch_size"])
            # Other params should have default values (same as template for this test)

    def config_metadata_enrichment(self)():
        """Test configuration metadata enrichment."""
        config = GLM47DynamicConfig(
            model_name="metadata_test",
            max_batch_size=16
        )

        # Register config
        config_manager.register_config("metadata_test", config)

        # Retrieve and check that it has metadata
        retrieved = config_manager.get_config("metadata_test")
        assert_is_not_none(retrieved)
        assert_true(hasattr(retrieved))
        assert_true(hasattr(retrieved))

    def config_comparison_and_cloning(self)():
        """Test configuration comparison and cloning."""
        config1 = GLM47DynamicConfig(
            model_name="clone_test_1",
            max_batch_size=16,
            use_flash_attention_2=True
        )

        config2 = GLM47DynamicConfig(
            model_name="clone_test_2",
            max_batch_size=16,  # Same value as config1
            use_flash_attention_2=True  # Same value as config1
        )

        # Register both
        config_manager.register_config("clone_test_1", config1)
        config_manager.register_config("clone_test_2", config2)

        # Retrieve both
        ret1 = config_manager.get_config("clone_test_1")
        ret2 = config_manager.get_config("clone_test_2")

        # They should have same param values but different names
        assert_equal(ret1.max_batch_size, ret2.max_batch_size)
        assert_equal(ret1.use_flash_attention_2, ret2.use_flash_attention_2)
        assert_not_equal(ret1.model_name, ret2.model_name)

    def config_schema_validation(self)():
        """Test configuration schema validation."""
        validator = get_config_validator()

        # Test valid config
        valid_config = Qwen34BDynamicConfig(
            model_name="schema_test",
            max_batch_size=16,
            max_memory_ratio=0.8
        )
        is_valid, errors = validator.validate_config(valid_config)
        assert_true(is_valid)
        assert_equal(len(errors))

        # Test config with out-of-range values
        invalid_config = Qwen34BDynamicConfig(
            model_name="schema_test",
            max_batch_size=0,  # Invalid: should be > 0
            max_memory_ratio=1.5  # Invalid: should be <= 1.0
        )
        is_valid, errors = validator.validate_config(invalid_config)
        assert_false(is_valid)
        assertGreater(len(errors))

if __name__ == '__main__':
    run_tests(test_functions)