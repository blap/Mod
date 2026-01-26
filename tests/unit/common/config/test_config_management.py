"""
Tests for the dynamic configuration management system.
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys
import shutil

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'src'))

from src.inference_pio.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false, 
    assert_is_none, assert_is_not_none, assert_in, assert_not_in, 
    assert_greater, assert_less, assert_is_instance, assert_raises, 
    run_tests
)


def test_config_manager_registration():
    """Test that config manager can register and retrieve configurations."""
    from src.inference_pio.config.dynamic_config import DynamicConfigManager, GLM47DynamicConfig
    
    # Create a temporary directory for config storage
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize the config manager
        config_manager = DynamicConfigManager(config_dir=test_dir)
        
        # Create a test config
        config = GLM47DynamicConfig(model_name="test_config")
        
        # Register the config
        result = config_manager.register_config("test_config", config)
        assert_true(result, "Config registration should succeed")
        
        # Retrieve the config
        retrieved = config_manager.get_config("test_config")
        assert_is_not_none(retrieved, "Retrieved config should not be None")
        assert_equal(retrieved.model_name, "test_config", "Retrieved config should have correct name")
        
        # Clean up
        shutil.rmtree(test_dir)
    except Exception as e:
        # Clean up in case of error
        shutil.rmtree(test_dir, ignore_errors=True)
        raise e


def test_config_manager_update():
    """Test that config manager can update configurations."""
    from src.inference_pio.config.dynamic_config import DynamicConfigManager, GLM47DynamicConfig
    
    # Create a temporary directory for config storage
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize the config manager
        config_manager = DynamicConfigManager(config_dir=test_dir)
        
        # Create and register initial config
        initial_config = GLM47DynamicConfig(model_name="update_test", temperature=0.5)
        config_manager.register_config("update_test", initial_config)
        
        # Update the config
        update_result = config_manager.update_config("update_test", {"temperature": 0.8})
        assert_true(update_result, "Config update should succeed")
        
        # Retrieve and verify the updated config
        updated_config = config_manager.get_config("update_test")
        assert_equal(updated_config.temperature, 0.8, "Updated config should have new temperature")
        
        # Clean up
        shutil.rmtree(test_dir)
    except Exception as e:
        # Clean up in case of error
        shutil.rmtree(test_dir, ignore_errors=True)
        raise e


def test_config_manager_list_configs():
    """Test that config manager can list configurations."""
    from src.inference_pio.config.dynamic_config import DynamicConfigManager, GLM47DynamicConfig
    
    # Create a temporary directory for config storage
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize the config manager
        config_manager = DynamicConfigManager(config_dir=test_dir)
        
        # Register multiple configs
        config1 = GLM47DynamicConfig(model_name="list_test_1")
        config2 = GLM47DynamicConfig(model_name="list_test_2")
        config_manager.register_config("list_test_1", config1)
        config_manager.register_config("list_test_2", config2)
        
        # List configs
        config_list = config_manager.list_configs()
        assert_greater(len(config_list), 1, "Should have at least 2 configs")
        assert_in("list_test_1", config_list, "Should contain first config")
        assert_in("list_test_2", config_list, "Should contain second config")
        
        # Clean up
        shutil.rmtree(test_dir)
    except Exception as e:
        # Clean up in case of error
        shutil.rmtree(test_dir, ignore_errors=True)
        raise e


def test_config_manager_delete():
    """Test that config manager can delete configurations."""
    from src.inference_pio.config.dynamic_config import DynamicConfigManager, GLM47DynamicConfig
    
    # Create a temporary directory for config storage
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize the config manager
        config_manager = DynamicConfigManager(config_dir=test_dir)
        
        # Register a config
        config = GLM47DynamicConfig(model_name="delete_test")
        config_manager.register_config("delete_test", config)
        
        # Verify it exists
        retrieved = config_manager.get_config("delete_test")
        assert_is_not_none(retrieved, "Config should exist before deletion")
        
        # Delete the config
        delete_result = config_manager.delete_config("delete_test")
        assert_true(delete_result, "Config deletion should succeed")
        
        # Verify it no longer exists
        deleted_retrieval = config_manager.get_config("delete_test")
        assert_is_none(deleted_retrieval, "Config should not exist after deletion")
        
        # Clean up
        shutil.rmtree(test_dir)
    except Exception as e:
        # Clean up in case of error
        shutil.rmtree(test_dir, ignore_errors=True)
        raise e


def test_config_manager_save_load():
    """Test that config manager can save and load configurations."""
    from src.inference_pio.config.dynamic_config import DynamicConfigManager, GLM47DynamicConfig
    
    # Create a temporary directory for config storage
    test_dir = tempfile.mkdtemp()
    
    try:
        # Initialize the config manager
        config_manager = DynamicConfigManager(config_dir=test_dir)
        
        # Create and register a config
        config = GLM47DynamicConfig(model_name="save_load_test", temperature=0.7)
        config_manager.register_config("save_load_test", config)
        
        # Save the config to a file
        temp_file = os.path.join(test_dir, "temp_config.json")
        save_result = config_manager.save_config("save_load_test", temp_file, "json")
        assert_true(save_result, "Config save should succeed")
        
        # Load the config from the file
        loaded_config = config_manager.load_config_from_file(temp_file, "json")
        assert_is_not_none(loaded_config, "Loaded config should not be None")
        assert_equal(loaded_config.model_name, "save_load_test", "Loaded config should have correct name")
        assert_equal(loaded_config.temperature, 0.7, "Loaded config should have correct temperature")
        
        # Clean up
        shutil.rmtree(test_dir)
    except Exception as e:
        # Clean up in case of error
        shutil.rmtree(test_dir, ignore_errors=True)
        raise e


def run_tests():
    """Run all configuration management tests."""
    print("Running dynamic config management tests...")
    
    test_functions = [
        test_config_manager_registration,
        test_config_manager_update,
        test_config_manager_list_configs,
        test_config_manager_delete,
        test_config_manager_save_load
    ]
    
    all_passed = True
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__} passed")
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {str(e)}")
            all_passed = False
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✓ All dynamic config management tests passed!")
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)