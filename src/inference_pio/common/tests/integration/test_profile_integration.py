"""
Integration test for optimization profiles with model configurations.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import os
from src.inference_pio.common.optimization_profiles import (
    get_profile_manager,
    PerformanceProfile,
    MemoryEfficientProfile,
    BalancedProfile,
    GLM47Profile,
    Qwen34BProfile,
    Qwen3CoderProfile,
    Qwen3VLProfile
)
from src.inference_pio.common.config_loader import get_config_loader
from src.inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

# TestProfileIntegration

    """Test integration of optimization profiles with model configurations."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        profile_manager = get_profile_manager()
        config_loader = get_config_loader()
        
        # Clear any existing configs/profiles to ensure clean test state
        for config_name in config_loader.config_manager.list_configs():
            try:
                config_loader.config_manager.delete_config(config_name)
            except:
                pass
        
        for profile_name in profile_manager.list_profiles():
            try:
                profile_manager.delete_profile(profile_name)
            except:
                pass

    def apply_performance_profile_to_glm_config(self)():
        """Test applying performance profile to GLM-4.7 config."""
        # Create a GLM config
        config = GLM47FlashConfig()
        
        # Create and register a performance profile
        perf_profile = PerformanceProfile(name="glm_perf", description="GLM performance profile")
        profile_manager.register_profile("glm_perf", perf_profile)
        
        # Apply profile to config using the config manager
        success = profile_manager.apply_profile_to_config("glm_perf", config)
        assert_true(success)
        
        # Verify profile settings were applied
        assertTrue(config.use_flash_attention_2)
        assert_false(config.gradient_checkpointing)  # Disabled for performance
        assertTrue(config.use_quantization)
        assert_equal(config.quantization_bits)

    def apply_memory_efficient_profile_to_qwen3_4b_config(self)():
        """Test applying memory efficient profile to Qwen3-4B config."""
        # Create a Qwen3-4B config
        config = Qwen34BInstruct2507Config()
        
        # Create and register a memory efficient profile
        mem_profile = MemoryEfficientProfile(name="qwen3_4b_mem")
        profile_manager.register_profile("qwen3_4b_mem", mem_profile)
        
        # Apply profile to config
        success = profile_manager.apply_profile_to_config("qwen3_4b_mem", config)
        assert_true(success)
        
        # Verify profile settings were applied
        assertTrue(config.use_flash_attention_2)
        assertTrue(config.gradient_checkpointing)  # Enabled for memory savings
        assert_equal(config.max_memory_ratio)
        assert_true(config.enable_disk_offloading)
        assertTrue(config.enable_activation_offloading)

    def apply_balanced_profile_to_qwen3_coder_config(self)():
        """Test applying balanced profile to Qwen3-Coder config."""
        # Create a Qwen3-Coder config
        config = Qwen3Coder30BConfig()
        
        # Create and register a balanced profile
        balanced_profile = BalancedProfile(name="qwen3_coder_bal")
        profile_manager.register_profile("qwen3_coder_bal", balanced_profile)
        
        # Apply profile to config
        success = profile_manager.apply_profile_to_config("qwen3_coder_bal", config)
        assert_true(success)
        
        # Verify profile settings were applied
        assertTrue(config.use_flash_attention_2)
        assertTrue(config.gradient_checkpointing)  # Enabled for balance
        assert_equal(config.max_memory_ratio)
        assert_true(config.enable_adaptive_batching)
        assert_equal(config.max_batch_size)

    def apply_model_specific_profile_to_qwen3_vl_config(self)():
        """Test applying Qwen3-VL specific profile to Qwen3-VL config."""
        # Create a Qwen3-VL config
        config = Qwen3VL2BConfig()
        
        # Create and register a Qwen3-VL specific profile
        vl_profile = Qwen3VLProfile(name="qwen3_vl_spec", description="Qwen3-VL specific profile")
        profile_manager.register_profile("qwen3_vl_spec", vl_profile)
        
        # Apply profile to config
        success = profile_manager.apply_profile_to_config("qwen3_vl_spec", config)
        assert_true(success)
        
        # Verify model-specific profile settings were applied
        assertTrue(config.use_qwen3_vl_attention_optimizations)
        assertTrue(config.use_multimodal_attention)
        assertTrue(config.use_cross_modal_fusion)
        assertTrue(config.enable_intelligent_multimodal_caching)
        assertTrue(config.enable_async_multimodal_processing)

    def create_config_from_profile_using_config_loader(self)():
        """Test creating a config from a profile using the config loader."""
        # Use the config loader to create a config from a profile
        success = config_loader.create_config_from_profile(
            model_type='glm',
            profile_name='performance',
            config_name='glm_from_profile'
        )
        assert_true(success)
        
        # Retrieve the created config
        config = config_loader.config_manager.get_config('glm_from_profile')
        assert_is_not_none(config)
        
        # Verify it has profile settings
        assertTrue(config.use_flash_attention_2)
        assert_false(config.gradient_checkpointing)  # Disabled for performance
        assertTrue(config.use_quantization)

    def save_and_load_profile(self)():
        """Test saving and loading optimization profiles."""
        # Create a custom profile
        custom_profile = PerformanceProfile(
            name="saved_profile")
        profile_manager.register_profile("saved_profile")
        
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(mode='w') as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Save the profile
            success = profile_manager.save_profile("saved_profile", temp_path, "json")
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
            assert_true(loaded_profile.use_tensor_parallelism)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def profile_application_with_overrides(self)():
        """Test applying a profile with additional overrides."""
        # Create a base config
        config = GLM47FlashConfig()
        
        # Create and register a profile
        profile = PerformanceProfile(name="override_test")
        profile_manager.register_profile("override_test", profile)
        
        # Apply profile to config
        success = profile_manager.apply_profile_to_config("override_test", config)
        assert_true(success)
        
        # Verify initial profile settings
        assert_equal(config.max_batch_size)  # Default from profile
        
        # Now apply overrides manually to the config
        config.max_batch_size = 64  # Override the profile setting
        config.gradient_checkpointing = True  # Override the profile setting
        
        # Verify overrides took effect
        assert_equal(config.max_batch_size, 64)
        assert_true(config.gradient_checkpointing)

    def multiple_profiles_on_same_config(self)():
        """Test applying multiple profiles to the same config (last wins)."""
        # Create a base config
        config = Qwen34BInstruct2507Config()
        
        # Create and register profiles
        perf_profile = PerformanceProfile(name="perf_first")
        mem_profile = MemoryEfficientProfile(name="mem_second", description="Second memory profile", max_batch_size=8)
        
        profile_manager.register_profile("perf_first", perf_profile)
        profile_manager.register_profile("mem_second", mem_profile)
        
        # Apply first profile
        success1 = profile_manager.apply_profile_to_config("perf_first", config)
        assert_true(success1)
        assert_equal(config.max_batch_size)  # From performance profile
        
        # Apply second profile (should override)
        success2 = profile_manager.apply_profile_to_config("mem_second", config)
        assert_true(success2)
        assert_equal(config.max_batch_size)  # From memory profile (second wins)
        
        # Some settings from the first profile might still remain if not overridden
        assert_true(config.use_flash_attention_2)  # Common to both profiles

    def glm_specific_optimizations(self)():
        """Test GLM-4.7 specific optimizations from profile."""
        config = GLM47FlashConfig()
        
        # Create and register a GLM-specific profile
        glm_profile = GLM47Profile(name="glm_specific")
        profile_manager.register_profile("glm_specific", glm_profile)
        
        # Apply profile to config
        success = profile_manager.apply_profile_to_config("glm_specific", config)
        assert_true(success)
        
        # Verify GLM-specific settings
        assertTrue(config.use_glm_attention_patterns)
        assertTrue(config.use_glm_ffn_optimization)
        assertTrue(config.use_glm_memory_efficient_kv)
        assert_equal(config.glm_kv_cache_compression_ratio)

    def qwen3_coder_specific_optimizations(self)():
        """Test Qwen3-Coder specific optimizations from profile."""
        config = Qwen3Coder30BConfig()
        
        # Create and register a Qwen3-Coder specific profile
        coder_profile = Qwen3CoderProfile(
            name="coder_specific", 
            description="Qwen3-Coder specific optimizations",
            code_generation_temperature=0.1
        )
        profile_manager.register_profile("coder_specific", coder_profile)
        
        # Apply profile to config
        success = profile_manager.apply_profile_to_config("coder_specific", config)
        assert_true(success)
        
        # Verify Qwen3-Coder specific settings
        assertTrue(config.use_qwen3_coder_attention_optimizations)
        assertTrue(config.use_qwen3_coder_code_optimizations)
        assert_equal(config.code_generation_temperature)  # Custom value
        assert_true(config.code_syntax_aware_attention)
        assertTrue(config.code_security_scanning)

    def profile_manager_list_and_metadata(self)():
        """Test profile manager listing and metadata functions."""
        # Register several profiles
        profiles = [
            PerformanceProfile(name="perf_meta"),
            MemoryEfficientProfile(name="mem_meta", description="Memory for metadata test"),
            BalancedProfile(name="bal_meta", description="Balanced for metadata test")
        ]
        
        for profile in profiles:
            profile_manager.register_profile(profile.name, profile)
        
        # Test listing profiles
        listed_profiles = profile_manager.list_profiles()
        assert_in("perf_meta", listed_profiles)
        assert_in("mem_meta", listed_profiles)
        assert_in("bal_meta", listed_profiles)
        
        # Test metadata retrieval
        metadata = profile_manager.get_profile_metadata("perf_meta")
        assert_is_not_none(metadata)
        assert_equal(metadata["name"])
        assert_equal(metadata["description"], "Performance for metadata test")
        assert_equal(metadata["type"], "PerformanceProfile")

if __name__ == '__main__':
    run_tests(test_functions)