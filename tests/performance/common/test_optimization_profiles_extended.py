"""
Comprehensive Tests for Optimization Profile System in Inference-PIO

This module contains comprehensive tests for the optimization profile system,
covering all aspects of profile creation, management, application, and integration.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import os
import json
from pathlib import Path

import sys
import os
from pathlib import Path

# Adicionando o diretório src ao path para permitir imports relativos
src_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(src_dir))

from inference_pio.common.optimization_profiles import (
    ProfileManager,
    PerformanceProfile,
    MemoryEfficientProfile,
    BalancedProfile,
    GLM47Profile,
    Qwen34BProfile,
    Qwen3CoderProfile,
    Qwen3VLProfile,
    get_profile_manager
)
from inference_pio.common.config_manager import (
    GLM47DynamicConfig,
    Qwen34BDynamicConfig,
    Qwen3CoderDynamicConfig,
    Qwen3VLDynamicConfig
)

# TestOptimizationProfileSystem

    """Comprehensive test suite for the optimization profile system."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        profile_manager = get_profile_manager()

        # Clear any existing profiles to ensure clean test state
        for profile_name in profile_manager.list_profiles():
            try:
                profile_manager.delete_profile(profile_name)
            except:
                pass

    def performance_profile_creation(self)():
        """Test creating a performance profile."""
        profile = PerformanceProfile(
            name="test_performance", 
            description="Test performance profile",
            max_batch_size=64,
            use_flash_attention_2=True,
            gradient_checkpointing=False,
            use_quantization=True,
            quantization_bits=8
        )

        assert_equal(profile.name, "test_performance")
        assert_equal(profile.description, "Test performance profile")
        assert_true(profile.use_flash_attention_2)
        assert_false(profile.gradient_checkpointing)  # Disabled for performance
        assertTrue(profile.use_quantization)
        assert_equal(profile.quantization_bits)
        assert_equal(profile.max_batch_size)

    def memory_efficient_profile_creation(self)():
        """Test creating a memory efficient profile."""
        profile = MemoryEfficientProfile(
            name="test_memory_efficient", 
            description="Test memory efficient profile",
            max_memory_ratio=0.5,
            gradient_checkpointing=True,
            enable_disk_offloading=True
        )

        assert_equal(profile.name, "test_memory_efficient")
        assert_true(profile.use_flash_attention_2)
        assertTrue(profile.gradient_checkpointing)  # Enabled for memory savings
        assert_equal(profile.max_memory_ratio)
        assert_true(profile.enable_disk_offloading)

    def balanced_profile_creation(self)():
        """Test creating a balanced profile."""
        profile = BalancedProfile(
            name="test_balanced")

        assert_equal(profile.name, "test_balanced")
        assert_true(profile.use_flash_attention_2)
        assertTrue(profile.gradient_checkpointing)  # Enabled for balance
        assert_equal(profile.max_memory_ratio)
        assert_true(profile.enable_adaptive_batching)
        assert_equal(profile.max_batch_size)

    def glm47_profile_creation(self)():
        """Test creating a GLM-4.7 specific profile."""
        profile = GLM47Profile(
            name="test_glm47", 
            description="Test GLM-4.7 profile",
            use_glm_attention_patterns=True,
            use_glm_ffn_optimization=True
        )

        assert_equal(profile.name, "test_glm47")
        assert_true(profile.use_glm_attention_patterns)
        assertTrue(profile.use_glm_ffn_optimization)
        assertTrue(profile.use_glm_memory_efficient_kv)

    def qwen3_4b_profile_creation(self)():
        """Test creating a Qwen3-4B specific profile."""
        profile = Qwen34BProfile(
            name="test_qwen3_4b")

        assert_equal(profile.name, "test_qwen3_4b")
        assert_true(profile.use_qwen3_attention_optimizations)
        assertTrue(profile.use_qwen3_kv_cache_optimizations)
        assertTrue(profile.use_qwen3_instruction_optimizations)

    def qwen3_coder_profile_creation(self)():
        """Test creating a Qwen3-Coder specific profile."""
        profile = Qwen3CoderProfile(
            name="test_qwen3_coder")

        assert_equal(profile.name, "test_qwen3_coder")
        assert_true(profile.use_qwen3_coder_attention_optimizations)
        assertTrue(profile.use_qwen3_coder_code_optimizations)
        assert_equal(profile.code_generation_temperature)
        assert_true(profile.code_syntax_aware_attention)

    def qwen3_vl_profile_creation(self)():
        """Test creating a Qwen3-VL specific profile."""
        profile = Qwen3VLProfile(
            name="test_qwen3_vl")

        assert_equal(profile.name, "test_qwen3_vl")
        assert_true(profile.use_qwen3_vl_attention_optimizations)
        assertTrue(profile.use_multimodal_attention)
        assertTrue(profile.use_cross_modal_fusion)

    def profile_manager_registration(self)():
        """Test registering and retrieving profiles."""
        profile = PerformanceProfile(
            name="perf_test")

        # Register the profile
        result = profile_manager.register_profile("perf_test", profile)
        assert_true(result)

        # Retrieve the profile
        retrieved = profile_manager.get_profile("perf_test")
        assert_is_not_none(retrieved)
        assert_equal(retrieved.name)
        assert_equal(retrieved.description)
        assert_equal(retrieved.max_batch_size, 48)

    def profile_application_to_config(self)():
        """Test applying a profile to a model configuration."""
        # Create a profile
        profile = PerformanceProfile(
            name="apply_test", 
            description="Apply test profile",
            max_batch_size=64,
            use_flash_attention_2=True,
            gradient_checkpointing=False,
            use_quantization=True
        )

        # Register the profile
        profile_manager.register_profile("apply_test", profile)

        # Create a GLM-4.7 config
        config = GLM47DynamicConfig()

        # Apply the profile to the config
        result = profile_manager.apply_profile_to_config("apply_test", config)
        assert_true(result)

        # Check that profile settings were applied
        assertTrue(config.use_flash_attention_2)
        assert_false(config.gradient_checkpointing)  # Should be disabled per profile
        assertTrue(config.use_quantization)
        assert_equal(config.max_batch_size)  # Should match profile

    def profile_manager_list_profiles(self)():
        """Test listing registered profiles."""
        # Register a few profiles
        perf_profile = PerformanceProfile(
            name="perf_list")
        mem_profile = MemoryEfficientProfile(
            name="mem_list", 
            description="Memory list test",
            max_memory_ratio=0.4
        )

        profile_manager.register_profile("perf_list", perf_profile)
        profile_manager.register_profile("mem_list", mem_profile)

        # List profiles
        profiles = profile_manager.list_profiles()

        assert_in("perf_list", profiles)
        assert_in("mem_list", profiles)
        assertGreaterEqual(len(profiles), 2)

    def profile_manager_delete_profile(self)():
        """Test deleting a profile."""
        # Register a profile
        profile = BalancedProfile(
            name="delete_test", 
            description="Delete test profile",
            max_batch_size=24
        )
        profile_manager.register_profile("delete_test", profile)

        # Verify it exists
        assert_is_not_none(profile_manager.get_profile("delete_test"))

        # Delete the profile
        result = profile_manager.delete_profile("delete_test")
        assert_true(result)

        # Verify it's gone
        assert_is_none(profile_manager.get_profile("delete_test"))

    def create_profile_from_template(self)():
        """Test creating a profile from a template."""
        # Use an existing template to create a new profile
        success = profile_manager.create_profile_from_template(
            "performance",
            "derived_perf_profile",
            {"max_batch_size": 128, "description": "Derived from template"}
        )

        assert_true(success)
        new_profile = profile_manager.get_profile("derived_perf_profile")
        assert_is_not_none(new_profile)
        assert_equal(new_profile.name)
        assert_equal(new_profile.description)
        assert_equal(new_profile.max_batch_size, 128)  # Overridden value
        assert_true(new_profile.use_flash_attention_2)  # Default from template

    def profile_metadata(self)():
        """Test getting profile metadata."""
        profile = GLM47Profile(
            name="meta_test")
        profile_manager.register_profile("meta_test", profile)

        metadata = profile_manager.get_profile_metadata("meta_test")

        assert_is_not_none(metadata)
        assert_equal(metadata["name"])
        assert_equal(metadata["type"], "GLM47Profile")
        assert_equal(metadata["description"], "Metadata test profile")
        assert_equal(metadata["version"], "2.0")
        assert_in("test", metadata["tags"])
        assert_in("glm", metadata["tags"])
        assert_equal(metadata["max_batch_size"], 32)

    def multiple_model_configs_with_profiles(self)():
        """Test applying profiles to different model configurations."""
        # Create and register a performance profile
        perf_profile = PerformanceProfile(
            name="multi_model_perf", 
            description="Multi-model performance profile",
            max_batch_size=48
        )
        profile_manager.register_profile("multi_model_perf", perf_profile)

        # Create configs for different models
        glm_config = GLM47DynamicConfig()
        qwen3_4b_config = Qwen34BDynamicConfig()
        qwen3_coder_config = Qwen3CoderDynamicConfig()
        qwen3_vl_config = Qwen3VLDynamicConfig()

        # Apply the same profile to all configs
        profile_manager.apply_profile_to_config("multi_model_perf", glm_config)
        profile_manager.apply_profile_to_config("multi_model_perf", qwen3_4b_config)
        profile_manager.apply_profile_to_config("multi_model_perf", qwen3_coder_config)
        profile_manager.apply_profile_to_config("multi_model_perf", qwen3_vl_config)

        # Verify that common settings were applied to all
        for config in [glm_config, qwen3_4b_config, qwen3_coder_config, qwen3_vl_config]:
            assert_true(config.use_flash_attention_2)
            assert_false(config.gradient_checkpointing)  # Disabled for performance
            assertTrue(config.use_quantization)
            assert_equal(config.max_batch_size)  # From profile

    def profile_serialization_json(self)():
        """Test serializing and deserializing profiles to/from JSON."""
        original_profile = PerformanceProfile(
            name="serialize_test",
            description="Test profile serialization",
            max_batch_size=32,
            use_flash_attention_2=True,
            gradient_checkpointing=False
        )

        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save the profile
            success = profile_manager.save_profile("serialize_test", temp_path, "json")
            # Since the profile isn't registered yet, we need to register it first
            profile_manager.register_profile("serialize_test", original_profile)
            success = profile_manager.save_profile("serialize_test", temp_path, "json")
            assert_true(success)

            # Create a new profile manager instance to simulate loading in a new session
            new_profile_manager = get_profile_manager()

            # Load the profile with a new name
            success = new_profile_manager.load_profile("loaded_profile")
            assert_true(success)

            # Retrieve the loaded profile
            loaded_profile = new_profile_manager.get_profile("loaded_profile")
            assert_is_not_none(loaded_profile)
            assert_equal(loaded_profile.name)
            assert_equal(loaded_profile.max_batch_size)
            assert_true(loaded_profile.use_flash_attention_2)
            assert_false(loaded_profile.gradient_checkpointing)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def profile_manager_update_profile(self)():
        """Test updating an existing profile."""
        profile = PerformanceProfile(
            name="update_test")
        profile_manager.register_profile("update_test")

        # Update the profile with new parameters
        updated_profile = PerformanceProfile(
            name="update_test", 
            description="Updated description",
            max_batch_size=64  # Changed value
        )
        success = profile_manager.update_profile("update_test", updated_profile)
        assert_true(success)

        # Verify the update
        retrieved = profile_manager.get_profile("update_test")
        assert_equal(retrieved.description)
        assert_equal(retrieved.max_batch_size, 64)

# TestOptimizationProfileIntegration

    """Integration tests for optimization profiles with other systems."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        profile_manager = get_profile_manager()

        # Clear any existing profiles to ensure clean test state
        for profile_name in profile_manager.list_profiles():
            try:
                profile_manager.delete_profile(profile_name)
            except:
                pass

    def profile_application_with_config_validation(self)():
        """Test applying profiles with config validation."""
        from src.inference_pio.common.config_validator import get_config_validator
        
        validator = get_config_validator()
        
        # Create a profile
        profile = MemoryEfficientProfile(
            name="validation_test",
            max_memory_ratio=0.6,
            enable_disk_offloading=True
        )
        profile_manager.register_profile("validation_test", profile)

        # Create a config
        config = Qwen34BDynamicConfig()

        # Apply profile to config
        profile_manager.apply_profile_to_config("validation_test", config)

        # Validate the resulting config
        is_valid, errors = validator.validate_config(config)
        assert_true(is_valid)
        assert_equal(len(errors))

    def profile_based_config_creation(self)():
        """Test creating configs based on profiles."""
        # This simulates the workflow of creating a config from a profile
        profile = BalancedProfile(
            name="config_creation_test",
            max_batch_size=24,
            max_memory_ratio=0.8
        )
        profile_manager.register_profile("config_creation_test", profile)

        # Create a config and apply the profile
        config = GLM47DynamicConfig()
        profile_manager.apply_profile_to_config("config_creation_test", config)

        # Verify profile settings were applied
        assert_equal(config.max_batch_size, 24)
        assert_equal(config.max_memory_ratio, 0.8)

    def profile_composition(self)():
        """Test composing multiple profiles (conceptual)."""
        # While direct composition might not be implemented,
        # we can test applying multiple profiles sequentially
        base_profile = PerformanceProfile(
            name="base_profile",
            max_batch_size=32,
            use_quantization=True
        )
        profile_manager.register_profile("base_profile", base_profile)

        override_profile = MemoryEfficientProfile(
            name="override_profile",
            max_memory_ratio=0.5,
            gradient_checkpointing=True
        )
        profile_manager.register_profile("override_profile", override_profile)

        # Create a config
        config = Qwen3CoderDynamicConfig()

        # Apply base profile
        profile_manager.apply_profile_to_config("base_profile", config)
        assert_equal(config.max_batch_size, 32)
        assert_true(config.use_quantization)

        # Apply override profile (some settings will be overridden)
        profile_manager.apply_profile_to_config("override_profile")
        assert_equal(config.max_memory_ratio, 0.5)
        assert_true(config.gradient_checkpointing)

        # Some original settings should remain
        assertTrue(config.use_quantization)  # From base profile

    def profile_manager_singleton_pattern(self)():
        """Test that profile manager follows singleton pattern."""
        manager1 = get_profile_manager()
        manager2 = get_profile_manager()

        # Both should be the same instance
        assertIs(manager1)

        # Add a profile through one manager
        profile = PerformanceProfile(name="singleton_test", max_batch_size=16)
        manager1.register_profile("singleton_test", profile)

        # Verify it's accessible through the other
        retrieved = manager2.get_profile("singleton_test")
        assert_is_not_none(retrieved)
        assert_equal(retrieved.max_batch_size)

    def profile_application_edge_cases(self)():
        """Test profile application with edge cases."""
        # Test with a config that has incompatible settings
        profile = PerformanceProfile(
            name="edge_case_test",
            max_batch_size=128,
            max_memory_ratio=0.9  # High memory ratio for performance
        )
        profile_manager.register_profile("edge_case_test", profile)

        # Create a config
        config = Qwen3VLDynamicConfig()

        # Apply profile - this should work even if settings seem contradictory
        result = profile_manager.apply_profile_to_config("edge_case_test", config)
        assert_true(result)

        # Verify the settings were applied
        assert_equal(config.max_batch_size)
        assert_equal(config.max_memory_ratio, 0.9)

# TestOptimizationProfileAdvancedFeatures

    """Tests for advanced features of the optimization profile system."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        profile_manager = get_profile_manager()

        # Clear any existing profiles to ensure clean test state
        for profile_name in profile_manager.list_profiles():
            try:
                profile_manager.delete_profile(profile_name)
            except:
                pass

    def profile_inheritance_or_derivation(self)():
        """Test profile derivation or inheritance concept."""
        # Create a base profile
        base_profile = BalancedProfile(
            name="base_profile",
            max_batch_size=16,
            max_memory_ratio=0.7,
            use_flash_attention_2=True
        )

        # Create a derived profile with some modifications
        derived_profile = BalancedProfile(
            name="derived_profile",
            max_batch_size=32,  # Different value
            max_memory_ratio=0.7,  # Same value
            use_flash_attention_2=True,  # Same value
            enable_adaptive_batching=True  # Additional parameter
        )

        # Register both profiles
        profile_manager.register_profile("base_profile", base_profile)
        profile_manager.register_profile("derived_profile", derived_profile)

        # Verify they have the expected differences
        base = profile_manager.get_profile("base_profile")
        derived = profile_manager.get_profile("derived_profile")

        assert_equal(base.max_batch_size, 16)
        assert_equal(derived.max_batch_size, 32)
        assert_true(base.use_flash_attention_2)
        assertTrue(derived.use_flash_attention_2)
        assert_false(hasattr(base) or base.enable_adaptive_batching == False)
        assert_true(derived.enable_adaptive_batching)

    def profile_template_system(self)():
        """Test profile templating system."""
        # Create a template-like profile
        template_profile = PerformanceProfile(
            name="template",
            max_batch_size=16,
            use_flash_attention_2=True,
            gradient_checkpointing=False
        )

        # Use this as a basis for creating similar profiles with variations
        variations = [
            {"name": "var1", "max_batch_size": 8, "description": "Variant 1"},
            {"name": "var2", "max_batch_size": 32, "description": "Variant 2"},
            {"name": "var3", "max_batch_size": 64, "description": "Variant 3"}
        ]

        for var_params in variations:
            # Create a new profile based on template but with variation
            var_profile = PerformanceProfile(**var_params)
            profile_name = var_params["name"]
            profile_manager.register_profile(profile_name, var_profile)

            # Verify the profile was created with the right parameters
            retrieved = profile_manager.get_profile(profile_name)
            assert_equal(retrieved.name, var_params["name"])
            assert_equal(retrieved.max_batch_size, var_params["max_batch_size"])
            assert_equal(retrieved.description, var_params["description"])

    def profile_metadata_enrichment(self)():
        """Test profile metadata enrichment."""
        profile = GLM47Profile(
            name="metadata_test",
            max_batch_size=16,
            tags=["test", "glm", "optimization"]
        )

        # Register profile
        profile_manager.register_profile("metadata_test", profile)

        # Retrieve and check metadata
        retrieved = profile_manager.get_profile("metadata_test")
        assert_is_not_none(retrieved)
        assert_true(hasattr(retrieved))
        assert_true(hasattr(retrieved))
        assertIn("test")

    def profile_comparison_and_cloning(self)():
        """Test profile comparison and potential cloning."""
        profile1 = PerformanceProfile(
            name="clone_test_1",
            max_batch_size=16,
            use_flash_attention_2=True
        )

        profile2 = PerformanceProfile(
            name="clone_test_2",
            max_batch_size=16,  # Same value as profile1
            use_flash_attention_2=True  # Same value as profile1
        )

        # Register both
        profile_manager.register_profile("clone_test_1", profile1)
        profile_manager.register_profile("clone_test_2", profile2)

        # Retrieve both
        ret1 = profile_manager.get_profile("clone_test_1")
        ret2 = profile_manager.get_profile("clone_test_2")

        # They should have same param values but different names
        assert_equal(ret1.max_batch_size, ret2.max_batch_size)
        assert_equal(ret1.use_flash_attention_2, ret2.use_flash_attention_2)
        assert_not_equal(ret1.name, ret2.name)

    def profile_schema_validation(self)():
        """Test profile schema validation."""
        # Test valid profile
        valid_profile = PerformanceProfile(
            name="schema_test",
            max_batch_size=16,
            max_memory_ratio=0.8
        )
        profile_manager.register_profile("schema_test", valid_profile)

        # Retrieve and verify
        retrieved = profile_manager.get_profile("schema_test")
        assert_is_not_none(retrieved)
        assert_equal(retrieved.max_batch_size)
        assert_equal(retrieved.max_memory_ratio, 0.8)

    def profile_export_import_functionality(self)():
        """Test profile export/import functionality."""
        # Create and register a profile
        original_profile = BalancedProfile(
            name="export_import_test",
            description="Test export/import functionality",
            max_batch_size=40,
            max_memory_ratio=0.75,
            enable_adaptive_batching=True
        )
        profile_manager.register_profile("export_import_test", original_profile)

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            export_path = tmp_file.name

        try:
            # Export the profile
            export_success = profile_manager.save_profile("export_import_test", export_path, "json")
            assert_true(export_success)

            # Import as a new profile
            import_success = profile_manager.load_profile("imported_test")
            assert_true(import_success)

            # Compare original and imported profiles
            original = profile_manager.get_profile("export_import_test")
            imported = profile_manager.get_profile("imported_test")

            assert_is_not_none(original)
            assertIsNotNone(imported)
            
            # Key properties should match
            assert_equal(original.max_batch_size)
            assert_equal(original.max_memory_ratio)
            assert_equal(original.enable_adaptive_batching, imported.enable_adaptive_batching)
        finally:
            # Clean up
            if os.path.exists(export_path):
                os.remove(export_path)

    def profile_manager_bulk_operations(self)():
        """Test bulk operations on profiles."""
        # Create multiple profiles
        profiles_data = [
            {"cls": PerformanceProfile, "params": {"name": "bulk_perf", "max_batch_size": 16}},
            {"cls": MemoryEfficientProfile, "params": {"name": "bulk_mem", "max_memory_ratio": 0.5}},
            {"cls": BalancedProfile, "params": {"name": "bulk_bal", "max_batch_size": 32}}
        ]

        # Register all profiles
        for profile_data in profiles_data:
            profile_cls = profile_data["cls"]
            params = profile_data["params"]
            profile = profile_cls(**params)
            profile_manager.register_profile(params["name"], profile)

        # Verify all were registered
        all_profiles = profile_manager.list_profiles()
        expected_names = ["bulk_perf", "bulk_mem", "bulk_bal"]
        for name in expected_names:
            assert_in(name, all_profiles)

        # Test bulk listing
        listed_profiles = profile_manager.list_profiles()
        assertGreaterEqual(len(listed_profiles), 3)

        # Test bulk metadata retrieval
        for name in expected_names:
            metadata = profile_manager.get_profile_metadata(name)
            assert_is_not_none(metadata)

# TestOptimizationProfileErrorHandling

    """Tests for error handling in optimization profiles."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        profile_manager = get_profile_manager()

    def profile_creation_with_invalid_params(self)():
        """Test profile creation with invalid parameters."""
        # Test creating a profile with invalid values
        try:
            invalid_profile = PerformanceProfile(
                name="")
            # Depending on validation, this might not raise an error at creation
            # but should be handled properly later
        except Exception:
            # This is acceptable if the constructor validates inputs
            pass

    def applying_profile_to_none_config(self)():
        """Test applying a profile to a None config."""
        profile = PerformanceProfile(name="none_test", max_batch_size=16)
        profile_manager.register_profile("none_test", profile)

        # Applying to None should handle gracefully
        try:
            result = profile_manager.apply_profile_to_config("none_test", None)
            # Result could be False or raise an exception depending on implementation
        except Exception:
            # This is acceptable behavior
            pass

    def loading_profile_from_invalid_file(self)():
        """Test loading a profile from an invalid file."""
        # Try to load from a non-existent file
        try:
            success = profile_manager.load_profile("nonexistent_profile", "/nonexistent/path.json", "json")
            # Should return False or handle gracefully
        except Exception:
            # This is acceptable behavior
            pass

    def register_duplicate_profile_name(self)():
        """Test registering a profile with a duplicate name."""
        profile1 = PerformanceProfile(name="duplicate_test", max_batch_size=16)
        profile2 = MemoryEfficientProfile(name="duplicate_test", max_memory_ratio=0.5)
        
        profile_manager.register_profile("duplicate_test", profile1)
        
        # Registering another profile with the same name might overwrite or raise an error
        # depending on implementation
        try:
            profile_manager.register_profile("duplicate_test", profile2)
            # If it succeeds, the second profile should be retrievable
            retrieved = profile_manager.get_profile("duplicate_test")
            # Which one we get depends on implementation (overwrite vs error)
        except Exception:
            # This is also acceptable behavior if duplicates are not allowed
            pass

    def get_nonexistent_profile(self)():
        """Test getting a nonexistent profile."""
        retrieved = profile_manager.get_profile("nonexistent_profile_12345")
        assert_is_none(retrieved)

    def delete_nonexistent_profile(self)():
        """Test deleting a nonexistent profile."""
        result = profile_manager.delete_profile("nonexistent_profile_12345")
        # Should return False or handle gracefully
        assertIsInstance(result)

if __name__ == '__main__':
    run_tests(test_functions)