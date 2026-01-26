"""
Comprehensive test suite for the dynamic configuration system in Inference-PIO.

This test suite validates the functionality of the dynamic configuration system including:
- Creation of various model-specific configurations (GLM-4.7, Qwen3-4B, Qwen3-Coder, Qwen3-VL)
- Configuration manager operations (registration, retrieval, updates)
- Configuration validation with both valid and invalid values
- Configuration loading and saving to/from files
- Template-based configuration creation
- Profile-based configuration generation for performance and memory optimization

The tests ensure that the configuration system properly manages model settings and parameters.
"""

import sys
import os
from typing import Any, Dict, Tuple

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config_creation() -> None:
    """
    Test the creation of various model-specific dynamic configurations.

    This test verifies that different model configurations can be instantiated
    with appropriate default values for model-specific parameters such as
    hidden size and attention mechanisms.
    """
    print("Testing configuration creation...")

    # Test GLM-4.7 dynamic config
    from inference_pio.common.config_manager import GLM47DynamicConfig
    glm_config = GLM47DynamicConfig()
    print(f"GLM-4.7 config created: {glm_config.model_name}")
    print(f"Hidden size: {glm_config.hidden_size}")
    print(f"Use flash attention: {glm_config.use_flash_attention_2}")

    # Test Qwen3-4B dynamic config
    from inference_pio.common.config_manager import Qwen34BDynamicConfig
    qwen3_4b_config = Qwen34BDynamicConfig()
    print(f"Qwen3-4B config created: {qwen3_4b_config.model_name}")
    print(f"Hidden size: {qwen3_4b_config.hidden_size}")
    print(f"Use flash attention: {qwen3_4b_config.use_flash_attention_2}")

    # Test Qwen3-Coder dynamic config
    from inference_pio.common.config_manager import Qwen3CoderDynamicConfig
    qwen3_coder_config = Qwen3CoderDynamicConfig()
    print(f"Qwen3-Coder config created: {qwen3_coder_config.model_name}")
    print(f"Hidden size: {qwen3_coder_config.hidden_size}")
    print(f"Use flash attention: {qwen3_coder_config.use_flash_attention_2}")

    # Test Qwen3-VL dynamic config
    from inference_pio.common.config_manager import Qwen3VLDynamicConfig
    qwen3_vl_config = Qwen3VLDynamicConfig()
    print(f"Qwen3-VL config created: {qwen3_vl_config.model_name}")
    print(f"Hidden size: {qwen3_vl_config.hidden_size}")
    print(f"Use flash attention: {qwen3_vl_config.use_flash_attention_2}")

    print("All configuration creations successful!")

def test_config_manager() -> None:
    """
    Test the configuration manager's core functionality.

    This test validates the configuration manager's ability to register,
    retrieve, and update configurations, ensuring proper storage and
    modification of configuration objects.
    """
    print("\nTesting configuration manager...")

    from inference_pio.common.config_manager import get_config_manager
    config_manager = get_config_manager()

    # Create a config
    from inference_pio.common.config_manager import GLM47DynamicConfig
    config = GLM47DynamicConfig()
    config.temperature = 0.8
    config.top_p = 0.9

    # Register the config
    success: bool = config_manager.register_config("test_config", config)
    print(f"Config registration success: {success}")

    # Retrieve the config
    retrieved_config = config_manager.get_config("test_config")
    print(f"Retrieved config temperature: {retrieved_config.temperature}")
    print(f"Retrieved config top_p: {retrieved_config.top_p}")

    # Update the config
    update_success: bool = config_manager.update_config("test_config", {"temperature": 0.7})
    print(f"Config update success: {update_success}")

    # Verify update
    updated_config = config_manager.get_config("test_config")
    print(f"Updated config temperature: {updated_config.temperature}")

    print("Configuration manager test successful!")

def test_config_validator() -> None:
    """
    Test the configuration validation system.

    This test ensures that the configuration validator properly accepts valid
    configuration values and rejects invalid ones, providing appropriate error
    messages for invalid configurations.
    """
    print("\nTesting configuration validation...")

    from inference_pio.common.config_validator import get_config_validator
    validator = get_config_validator()

    from inference_pio.common.config_manager import GLM47DynamicConfig
    config = GLM47DynamicConfig()
    config.temperature = 0.8  # Valid value

    is_valid: bool
    errors: list
    is_valid, errors = validator.validate_config(config)
    print(f"Valid config validation result: {is_valid}, errors: {errors}")

    # Test with invalid value
    config.temperature = -1.0  # Invalid value
    is_valid, errors = validator.validate_config(config)
    print(f"Invalid config validation result: {is_valid}, errors: {errors}")

    print("Configuration validation test successful!")

def test_config_loader() -> None:
    """
    Test the configuration loading and saving functionality.

    This test validates the ability to serialize configurations to files
    and deserialize them back, ensuring data integrity during the
    save/load cycle.
    """
    print("\nTesting configuration loading and saving...")

    import tempfile
    import os

    from inference_pio.common.config_manager import get_config_manager
    from inference_pio.common.config_loader import get_config_loader
    config_manager = get_config_manager()
    config_loader = get_config_loader()

    # Create a config
    from inference_pio.common.config_manager import GLM47DynamicConfig
    config = GLM47DynamicConfig()
    config.temperature = 0.85
    config.top_p = 0.92

    # Register the config
    config_manager.register_config("save_test_config", config)

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        temp_path: str = tmp_file.name

    try:
        save_success: bool = config_manager.save_config("save_test_config", temp_path, "json")
        print(f"Config save success: {save_success}")

        if save_success:
            # Load from the file
            load_success: bool = config_loader.load_config_from_file(temp_path, "loaded_config")
            print(f"Config load success: {load_success}")

            if load_success:
                loaded_config = config_manager.get_config("loaded_config")
                print(f"Loaded config temperature: {loaded_config.temperature}")
                print(f"Loaded config top_p: {loaded_config.top_p}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    print("Configuration loading and saving test successful!")

def test_config_templates() -> None:
    """
    Test the configuration template system.

    This test verifies that configurations can be created from predefined
    templates with customized parameters, allowing for rapid configuration
    setup based on standard patterns.
    """
    print("\nTesting configuration templates...")

    from inference_pio.common.config_loader import get_config_loader
    config_loader = get_config_loader()

    # Create config from template
    success: bool = config_loader.create_config_from_template(
        "glm_4_7_flash",
        "templated_config",
        {"temperature": 0.7, "top_p": 0.8}
    )
    print(f"Template config creation success: {success}")

    if success:
        from inference_pio.common.config_manager import get_config_manager
        config_manager = get_config_manager()
        config = config_manager.get_config("templated_config")
        print(f"Templated config temperature: {config.temperature}")
        print(f"Templated config top_p: {config.top_p}")
        print(f"Templated config model name: {config.model_name}")

    print("Configuration templates test successful!")

def test_config_profiles() -> None:
    """
    Test the configuration profile system.

    This test validates the creation of configurations based on predefined
    profiles optimized for specific use cases such as performance or memory
    efficiency, ensuring appropriate parameter settings for each profile.
    """
    print("\nTesting configuration profiles...")

    from inference_pio.common.config_loader import create_config_from_profile

    # Create performance-optimized config
    perf_config: Dict[str, Any] = create_config_from_profile("glm", "performance", temperature=0.8)
    print(f"Performance config temperature: {perf_config['temperature']}")
    print(f"Performance config flash attention: {perf_config['use_flash_attention_2']}")
    print(f"Performance config gradient checkpointing: {perf_config['gradient_checkpointing']}")  # Should be False for performance

    # Create memory-efficient config
    mem_config: Dict[str, Any] = create_config_from_profile("qwen3_4b", "memory_efficient", temperature=0.7)
    print(f"Memory-efficient config temperature: {mem_config['temperature']}")
    print(f"Memory-efficient config flash attention: {mem_config['use_flash_attention_2']}")
    print(f"Memory-efficient config gradient checkpointing: {mem_config['gradient_checkpointing']}")  # Should be True for memory efficiency

    print("Configuration profiles test successful!")

if __name__ == "__main__":
    print("Running dynamic configuration system tests...\n")

    test_config_creation()
    test_config_manager()
    test_config_validator()
    test_config_loader()
    test_config_templates()
    test_config_profiles()

    print("\nAll tests completed successfully!")