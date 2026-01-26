"""
Comprehensive test suite for the optimization profile system in Inference-PIO.

This test suite validates the functionality of the optimization profile system including:
- Creation of various optimization profiles (Performance, Memory Efficient, Balanced)
- Model-specific profiles (GLM-4.7, Qwen3-4B, Qwen3-Coder, Qwen3-VL)
- Profile manager operations (registration, retrieval, listing, deletion)
- Profile application to model configurations
- Profile creation from templates
- Profile metadata management
- Multi-model profile application

The tests ensure that optimization profiles are correctly created, managed, and applied to model configurations.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
from pathlib import Path
from typing import Any, Dict, List

# Adicionando o diretório src ao path para permitir imports relativos
src_dir = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(src_dir))

from inference_pio.test_utils import (
    assert_equal, assert_true, assert_false, assert_is_not_none,
    assert_in, assert_is_none, run_tests
)
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


def test_performance_profile_creation() -> None:
    """
    Test creating a performance optimization profile.

    Validates that a PerformanceProfile is created with the expected default settings
    optimized for maximum performance, including flash attention, disabled gradient
    checkpointing, and enabled quantization.
    """
    profile = PerformanceProfile(name="test_performance", description="Test performance profile")

    assert_equal(profile.name, "test_performance")
    assert_equal(profile.description, "Test performance profile")
    assert_true(profile.use_flash_attention_2)
    assert_false(profile.gradient_checkpointing)  # Disabled for performance
    assert_true(profile.use_quantization)


def test_memory_efficient_profile_creation() -> None:
    """
    Test creating a memory efficient optimization profile.

    Validates that a MemoryEfficientProfile is created with the expected default settings
    optimized for memory conservation, including enabled gradient checkpointing,
    disk offloading, and reduced memory ratios.
    """
    profile = MemoryEfficientProfile(name="test_memory_efficient", description="Test memory efficient profile")

    assert_equal(profile.name, "test_memory_efficient")
    assert_true(profile.use_flash_attention_2)
    assert_true(profile.gradient_checkpointing)  # Enabled for memory savings
    assert_equal(profile.max_memory_ratio, 0.6)
    assert_true(profile.enable_disk_offloading)


def test_balanced_profile_creation() -> None:
    """
    Test creating a balanced optimization profile.

    Validates that a BalancedProfile is created with the expected default settings
    that balance performance and memory efficiency, including adaptive batching
    and moderate memory constraints.
    """
    profile = BalancedProfile(name="test_balanced", description="Test balanced profile")

    assert_equal(profile.name, "test_balanced")
    assert_true(profile.use_flash_attention_2)
    assert_true(profile.gradient_checkpointing)  # Enabled for balance
    assert_equal(profile.max_memory_ratio, 0.8)
    assert_true(profile.enable_adaptive_batching)


def test_glm47_profile_creation() -> None:
    """
    Test creating a GLM-4.7 specific optimization profile.

    Validates that a GLM47Profile is created with the expected model-specific
    optimizations tailored for the GLM-4.7 architecture, including attention
    patterns and FFN optimizations.
    """
    profile = GLM47Profile(name="test_glm47", description="Test GLM-4.7 profile")

    assert_equal(profile.name, "test_glm47")
    assert_true(profile.use_glm_attention_patterns)
    assert_true(profile.use_glm_ffn_optimization)
    assert_true(profile.use_glm_memory_efficient_kv)


def test_qwen3_4b_profile_creation() -> None:
    """
    Test creating a Qwen3-4B specific optimization profile.

    Validates that a Qwen34BProfile is created with the expected model-specific
    optimizations tailored for the Qwen3-4B architecture, including attention
    and instruction optimizations.
    """
    profile = Qwen34BProfile(name="test_qwen3_4b", description="Test Qwen3-4B profile")

    assert_equal(profile.name, "test_qwen3_4b")
    assert_true(profile.use_qwen3_attention_optimizations)
    assert_true(profile.use_qwen3_kv_cache_optimizations)
    assert_true(profile.use_qwen3_instruction_optimizations)


def test_qwen3_coder_profile_creation() -> None:
    """
    Test creating a Qwen3-Coder specific optimization profile.

    Validates that a Qwen3CoderProfile is created with the expected model-specific
    optimizations tailored for the Qwen3-Coder architecture, particularly for
    code generation tasks with appropriate temperature settings.
    """
    profile = Qwen3CoderProfile(name="test_qwen3_coder", description="Test Qwen3-Coder profile")

    assert_equal(profile.name, "test_qwen3_coder")
    assert_true(profile.use_qwen3_coder_attention_optimizations)
    assert_true(profile.use_qwen3_coder_code_optimizations)
    assert_equal(profile.code_generation_temperature, 0.2)


def test_qwen3_vl_profile_creation() -> None:
    """
    Test creating a Qwen3-VL specific optimization profile.

    Validates that a Qwen3VLProfile is created with the expected model-specific
    optimizations tailored for the Qwen3-VL architecture, particularly for
    multimodal tasks with cross-modal fusion capabilities.
    """
    profile = Qwen3VLProfile(name="test_qwen3_vl", description="Test Qwen3-VL profile")

    assert_equal(profile.name, "test_qwen3_vl")
    assert_true(profile.use_qwen3_vl_attention_optimizations)
    assert_true(profile.use_multimodal_attention)
    assert_true(profile.use_cross_modal_fusion)


def test_profile_manager_registration() -> None:
    """
    Test registering and retrieving optimization profiles via the manager.

    Validates that optimization profiles can be registered with the ProfileManager
    and subsequently retrieved by their assigned names, ensuring proper storage
    and retrieval mechanisms.
    """
    profile_manager = get_profile_manager()

    # Clear any existing profiles to ensure clean test state
    for profile_name in profile_manager.list_profiles():
        try:
            profile_manager.delete_profile(profile_name)
        except:
            pass

    profile = PerformanceProfile(name="perf_test", description="Performance test profile")

    # Register the profile
    result: bool = profile_manager.register_profile("perf_test", profile)
    assert_true(result)

    # Retrieve the profile
    retrieved = profile_manager.get_profile("perf_test")
    assert_is_not_none(retrieved)
    assert_equal(retrieved.name, "perf_test")
    assert_equal(retrieved.description, "Performance test profile")


def test_profile_application_to_config() -> None:
    """
    Test applying an optimization profile to a model configuration.

    Validates that profile settings are correctly applied to model configurations,
    modifying the configuration parameters according to the profile's optimization
    strategies.
    """
    profile_manager = get_profile_manager()

    # Create a profile
    profile = PerformanceProfile(name="apply_test", description="Apply test profile")

    # Register the profile
    profile_manager.register_profile("apply_test", profile)

    # Create a GLM-4.7 config
    config = GLM47DynamicConfig()

    # Apply the profile to the config
    result: bool = profile_manager.apply_profile_to_config("apply_test", config)
    assert_true(result)

    # Check that profile settings were applied
    assert_true(config.use_flash_attention_2)
    assert_false(config.gradient_checkpointing)  # Should be disabled per profile
    assert_true(config.use_quantization)


def test_profile_manager_list_profiles() -> None:
    """
    Test listing all registered optimization profiles.

    Validates that the ProfileManager can return a list of all currently
    registered profile names, allowing for profile discovery and management.
    """
    profile_manager = get_profile_manager()

    # Register a few profiles
    perf_profile = PerformanceProfile(name="perf_list", description="Performance list test")
    mem_profile = MemoryEfficientProfile(name="mem_list", description="Memory list test")

    profile_manager.register_profile("perf_list", perf_profile)
    profile_manager.register_profile("mem_list", mem_profile)

    # List profiles
    profiles: List[str] = profile_manager.list_profiles()

    assert_in("perf_list", profiles)
    assert_in("mem_list", profiles)
    assert_true(len(profiles) >= 2)


def test_profile_manager_delete_profile() -> None:
    """
    Test deleting an optimization profile from the manager.

    Validates that profiles can be removed from the ProfileManager and
    are no longer accessible after deletion.
    """
    profile_manager = get_profile_manager()

    # Register a profile
    profile = BalancedProfile(name="delete_test", description="Delete test profile")
    profile_manager.register_profile("delete_test", profile)

    # Verify it exists
    assert_is_not_none(profile_manager.get_profile("delete_test"))

    # Delete the profile
    result: bool = profile_manager.delete_profile("delete_test")
    assert_true(result)

    # Verify it's gone
    assert_is_none(profile_manager.get_profile("delete_test"))


def test_create_profile_from_template() -> None:
    """
    Test creating an optimization profile from an existing template.

    Validates that new profiles can be instantiated from predefined templates
    with customized parameters, enabling rapid profile creation based on
    standard optimization patterns.
    """
    profile_manager = get_profile_manager()

    # Use an existing template to create a new profile
    success: bool = profile_manager.create_profile_from_template(
        "performance",
        "derived_perf_profile",
        {"max_batch_size": 64, "description": "Derived from template"}
    )

    assert_true(success)
    new_profile = profile_manager.get_profile("derived_perf_profile")
    assert_is_not_none(new_profile)
    assert_equal(new_profile.name, "derived_perf_profile")
    assert_equal(new_profile.description, "Derived from template")
    assert_equal(new_profile.max_batch_size, 64)  # Overridden value
    assert_true(new_profile.use_flash_attention_2)  # Default from template


def test_profile_metadata() -> None:
    """
    Test retrieving metadata for an optimization profile.

    Validates that detailed metadata about profiles can be obtained,
    including name, type, description, version, and tags.
    """
    profile_manager = get_profile_manager()

    profile = GLM47Profile(name="meta_test", description="Metadata test profile", version="2.0", tags=["test", "glm"])
    profile_manager.register_profile("meta_test", profile)

    metadata: Dict[str, Any] = profile_manager.get_profile_metadata("meta_test")

    assert_is_not_none(metadata)
    assert_equal(metadata["name"], "meta_test")
    assert_equal(metadata["type"], "GLM47Profile")
    assert_equal(metadata["description"], "Metadata test profile")
    assert_equal(metadata["version"], "2.0")
    assert_in("test", metadata["tags"])
    assert_in("glm", metadata["tags"])


def test_multiple_model_configs_with_profiles() -> None:
    """
    Test applying optimization profiles to multiple different model configurations.

    Validates that a single optimization profile can be consistently applied
    across different model configurations (GLM-4.7, Qwen3-4B, Qwen3-Coder, Qwen3-VL),
    ensuring uniform optimization settings across various model types.
    """
    profile_manager = get_profile_manager()

    # Create and register a performance profile
    perf_profile = PerformanceProfile(name="multi_model_perf", description="Multi-model performance profile")
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
        assert_true(config.use_quantization)


if __name__ == '__main__':
    run_tests([
        test_performance_profile_creation,
        test_memory_efficient_profile_creation,
        test_balanced_profile_creation,
        test_glm47_profile_creation,
        test_qwen3_4b_profile_creation,
        test_qwen3_coder_profile_creation,
        test_qwen3_vl_profile_creation,
        test_profile_manager_registration,
        test_profile_application_to_config,
        test_profile_manager_list_profiles,
        test_profile_manager_delete_profile,
        test_create_profile_from_template,
        test_profile_metadata,
        test_multiple_model_configs_with_profiles
    ])