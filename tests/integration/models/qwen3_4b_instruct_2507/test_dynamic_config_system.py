"""
Test suite for the dynamic configuration system in Inference-PIO.

This module tests the dynamic configuration system for all models.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import os
from pathlib import Path

# Import the configuration system components
from src.inference_pio.common.config_manager import (
    get_config_manager,
    GLM47DynamicConfig,
    Qwen34BDynamicConfig,
    Qwen3CoderDynamicConfig,
    Qwen3VLDynamicConfig
)
from src.inference_pio.common.config_loader import (
    get_config_loader,
    create_config_from_profile
)
from src.inference_pio.common.config_validator import get_config_validator
from src.inference_pio.common.config_integration import (
    ConfigurableModelPlugin,
    apply_configuration_to_plugin
)

# Import the model plugins
from src.inference_pio.models.glm_4_7_flash.config_integration import (
    GLM47ConfigurablePlugin,
    create_glm_4_7_configurable_plugin
)
from src.inference_pio.models.qwen3_4b_instruct_2507.config_integration import (
    Qwen34BInstruct2507ConfigurablePlugin,
    create_qwen3_4b_instruct_2507_configurable_plugin
)
from src.inference_pio.models.qwen3_coder_30b.config_integration import (
    Qwen3Coder30BConfigurablePlugin,
    create_qwen3_coder_30b_configurable_plugin
)
from src.inference_pio.models.qwen3_vl_2b.config_integration import (
    Qwen3VL2BConfigurablePlugin,
    create_qwen3_vl_2b_configurable_plugin
)

from src.inference_pio.common.base_plugin_interface import ModelPluginMetadata, PluginType

# TestDynamicConfigurationSystem

    """Test cases for the dynamic configuration system."""
    
    def setup_helper():
        """Set up test fixtures."""
        config_manager = get_config_manager()
        config_loader = get_config_loader()
        config_validator = get_config_validator()
        
        # Create temporary directory for config files
        temp_dir = tempfile.mkdtemp()
    
    def cleanup_helper():
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def glm47_dynamic_config_creation(self)():
        """Test creation of GLM-4.7 dynamic configuration."""
        config = GLM47DynamicConfig()
        
        # Check that required fields have default values
        assert_equal(config.model_name, "GLM-4.7")
        assert_equal(config.hidden_size, 5120)
        assert_true(config.use_flash_attention_2)
        assertTrue(config.use_sparse_attention)
        
        # Test updating configuration
        config.temperature = 0.8
        config.top_p = 0.95
        assert_equal(config.temperature)
        assert_equal(config.top_p, 0.95)
    
    def qwen3_4b_dynamic_config_creation(self)():
        """Test creation of Qwen3-4B dynamic configuration."""
        config = Qwen34BDynamicConfig()
        
        # Check that required fields have default values
        assert_equal(config.model_name, "Qwen3-4B-Instruct-2507")
        assert_equal(config.hidden_size, 2560)
        assert_true(config.use_flash_attention_2)
        assertTrue(config.use_sparse_attention)
        
        # Test updating configuration
        config.temperature = 0.9
        config.top_p = 0.92
        assert_equal(config.temperature)
        assert_equal(config.top_p, 0.92)
    
    def qwen3_coder_dynamic_config_creation(self)():
        """Test creation of Qwen3-Coder dynamic configuration."""
        config = Qwen3CoderDynamicConfig()
        
        # Check that required fields have default values
        assert_equal(config.model_name, "Qwen3-Coder-30B")
        assert_equal(config.hidden_size, 4096)
        assert_true(config.use_flash_attention_2)
        assertTrue(config.use_sparse_attention)
        
        # Test updating configuration
        config.temperature = 0.75
        config.top_p = 0.88
        assert_equal(config.temperature)
        assert_equal(config.top_p, 0.88)
    
    def qwen3_vl_dynamic_config_creation(self)():
        """Test creation of Qwen3-VL dynamic configuration."""
        config = Qwen3VLDynamicConfig()
        
        # Check that required fields have default values
        assert_equal(config.model_name, "Qwen3-VL-2B")
        assert_equal(config.hidden_size, 2048)
        assert_true(config.use_flash_attention_2)
        assertTrue(config.use_sparse_attention)
        
        # Test updating configuration
        config.temperature = 0.85
        config.top_p = 0.9
        assert_equal(config.temperature)
        assert_equal(config.top_p, 0.9)
    
    def config_manager_registration(self)():
        """Test configuration manager registration functionality."""
        config = GLM47DynamicConfig()
        config.temperature = 0.9
        
        # Register configuration
        success = config_manager.register_config("test_glm_config", config)
        assert_true(success)
        
        # Retrieve configuration
        retrieved_config = config_manager.get_config("test_glm_config")
        assert_is_not_none(retrieved_config)
        assert_equal(retrieved_config.temperature)
        
        # Update configuration
        update_success = config_manager.update_config("test_glm_config")
        assert_true(update_success)
        
        # Verify update
        updated_config = config_manager.get_config("test_glm_config")
        assert_equal(updated_config.temperature)
    
    def config_validation(self)():
        """Test configuration validation."""
        config = GLM47DynamicConfig()
        config.temperature = 1.5  # Valid value
        
        is_valid, errors = config_validator.validate_config(config)
        assert_true(is_valid)
        assert_equal(len(errors))
        
        # Test with invalid value
        config.temperature = -1.0  # Invalid value
        
        is_valid, errors = config_validator.validate_config(config)
        assert_false(is_valid)
        assert_greater(len(errors))
    
    def config_loading_and_saving(self)():
        """Test configuration loading and saving."""
        config = GLM47DynamicConfig()
        config.temperature = 0.8
        config.top_p = 0.9
        
        # Register configuration
        config_manager.register_config("save_test_config", config)
        
        # Save to JSON
        json_path = os.path.join(temp_dir, "test_config.json")
        success = config_manager.save_config("save_test_config", json_path, "json")
        assert_true(success)
        
        # Load from JSON
        load_success = config_loader.load_config_from_file(json_path)
        assert_true(load_success)
        
        # Verify loaded configuration
        loaded_config = config_manager.get_config("loaded_config")
        assert_is_not_none(loaded_config)
        assert_equal(loaded_config.temperature)
        assert_equal(loaded_config.top_p)
    
    def config_templates(self)():
        """Test configuration templates."""
        # Create config from template
        success = config_loader.create_config_from_template(
            "glm_4_7_flash",
            "templated_config", 
            {"temperature": 0.7, "top_p": 0.8}
        )
        assert_true(success)
        
        # Verify the config was created with overrides
        config = config_manager.get_config("templated_config")
        assert_is_not_none(config)
        assert_equal(config.temperature)
        assert_equal(config.top_p)
        assert_equal(config.model_name, "GLM-4.7")  # From template
    
    def config_profiles(self)():
        """Test configuration profiles."""
        # Create performance-optimized config
        perf_config = create_config_from_profile("glm", "performance", temperature=0.8)
        assert_is_not_none(perf_config)
        assert_equal(perf_config["temperature"])
        assert_true(perf_config["use_flash_attention_2"])
        assertTrue(perf_config["use_sparse_attention"])
        assert_false(perf_config["gradient_checkpointing"])  # Disabled for performance
        
        # Create memory-efficient config
        mem_config = create_config_from_profile("qwen3_4b")
        assert_is_not_none(mem_config)
        assert_equal(mem_config["temperature"])
        assert_true(mem_config["use_flash_attention_2"])
        assertTrue(mem_config["gradient_checkpointing"])  # Enabled for memory efficiency
        assertTrue(mem_config["enable_disk_offloading"])
        assertTrue(mem_config["enable_activation_offloading"])
    
    def glm47_plugin_with_config(self)():
        """Test GLM-4.7 plugin with dynamic configuration."""
        plugin = create_glm_4_7_configurable_plugin()
        
        # Initialize plugin
        init_success = plugin.initialize()
        assertTrue(init_success)
        
        # Create and register a configuration
        config = GLM47DynamicConfig()
        config.temperature = 0.8
        config.top_p = 0.9
        
        config_manager.register_config("glm_test_config")
        
        # Activate configuration
        activate_success = plugin.activate_configuration("glm_test_config")
        assert_true(activate_success)
        
        # Verify configuration is active
        active_config = plugin.get_active_configuration("test_glm_model")
        assert_is_not_none(active_config)
        assert_equal(active_config.temperature)
        assert_equal(active_config.top_p)
    
    def qwen3_4b_plugin_with_config(self)():
        """Test Qwen3-4B plugin with dynamic configuration."""
        plugin = create_qwen3_4b_instruct_2507_configurable_plugin()
        
        # Initialize plugin
        init_success = plugin.initialize()
        assert_true(init_success)
        
        # Create and register a configuration
        config = Qwen34BDynamicConfig()
        config.temperature = 0.85
        config.top_p = 0.88
        
        config_manager.register_config("qwen3_4b_test_config")
        
        # Activate configuration
        activate_success = plugin.activate_configuration("qwen3_4b_test_config", "test_qwen3_4b_model")
        assert_true(activate_success)
        
        # Verify configuration is active
        active_config = plugin.get_active_configuration("test_qwen3_4b_model")
        assert_is_not_none(active_config)
        assert_equal(active_config.temperature)
        assert_equal(active_config.top_p)
    
    def qwen3_coder_plugin_with_config(self)():
        """Test Qwen3-Coder plugin with dynamic configuration."""
        plugin = create_qwen3_coder_30b_configurable_plugin()
        
        # Initialize plugin
        init_success = plugin.initialize()
        assert_true(init_success)
        
        # Create and register a configuration
        config = Qwen3CoderDynamicConfig()
        config.temperature = 0.75
        config.top_p = 0.92
        
        config_manager.register_config("qwen3_coder_test_config")
        
        # Activate configuration
        activate_success = plugin.activate_configuration("qwen3_coder_test_config", "test_qwen3_coder_model")
        assert_true(activate_success)
        
        # Verify configuration is active
        active_config = plugin.get_active_configuration("test_qwen3_coder_model")
        assert_is_not_none(active_config)
        assert_equal(active_config.temperature)
        assert_equal(active_config.top_p)
    
    def qwen3_vl_plugin_with_config(self)():
        """Test Qwen3-VL plugin with dynamic configuration."""
        plugin = create_qwen3_vl_2b_configurable_plugin()
        
        # Initialize plugin
        init_success = plugin.initialize()
        assert_true(init_success)
        
        # Create and register a configuration
        config = Qwen3VLDynamicConfig()
        config.temperature = 0.82
        config.top_p = 0.89
        
        config_manager.register_config("qwen3_vl_test_config")
        
        # Activate configuration
        activate_success = plugin.activate_configuration("qwen3_vl_test_config", "test_qwen3_vl_model")
        assert_true(activate_success)
        
        # Verify configuration is active
        active_config = plugin.get_active_configuration("test_qwen3_vl_model")
        assert_is_not_none(active_config)
        assert_equal(active_config.temperature)
        assert_equal(active_config.top_p)
    
    def apply_config_to_plugin_helper(self)():
        """Test the helper function to apply configuration to plugin."""
        plugin = create_glm_4_7_configurable_plugin()
        
        # Initialize plugin
        init_success = plugin.initialize()
        assert_true(init_success)
        
        # Create and register a configuration
        config = GLM47DynamicConfig()
        config.temperature = 0.77
        config.top_p = 0.85
        
        config_manager.register_config("helper_test_config")
        
        # Apply configuration using helper
        apply_success = apply_configuration_to_plugin(plugin, "helper_test_config", "test_helper_model")
        assert_true(apply_success)
        
        # Verify configuration is active
        active_config = plugin.get_active_configuration("test_helper_model")
        assert_is_not_none(active_config)
        assert_equal(active_config.temperature)
        assert_equal(active_config.top_p)

if __name__ == "__main__":
    run_tests(test_functions)