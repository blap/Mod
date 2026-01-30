"""
Comprehensive test for the disk-based inference pipeline system with model plugins.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin


def test_glm_4_7_pipeline_integration():
    """Test GLM-4.7 plugin integration with pipeline functionality."""
    print("Testing GLM-4.7 plugin pipeline integration...")
    
    # Create GLM-4.7 plugin
    plugin = GLM_4_7_Plugin()
    
    # Verify that the plugin has all required pipeline methods
    methods_to_check = [
        'setup_pipeline',
        'execute_pipeline', 
        'create_pipeline_stages',
        'get_pipeline_stats',
        'get_pipeline_manager'
    ]
    
    for method_name in methods_to_check:
        assert hasattr(plugin, method_name), f"Plugin missing {method_name} method"
    
    print("  ✓ All pipeline methods present")
    
    # Test pipeline stats before setup
    stats = plugin.get_pipeline_stats()
    assert 'pipeline_enabled' in stats
    assert 'num_stages' in stats
    assert 'checkpoint_directory' in stats
    print("  ✓ Pipeline stats structure correct")
    
    # Test that pipeline is not enabled by default
    assert stats['pipeline_enabled'] == False
    print("  ✓ Pipeline disabled by default")


def test_qwen3_4b_instruct_pipeline_integration():
    """Test Qwen3-4B-Instruct plugin integration with pipeline functionality."""
    print("Testing Qwen3-4B-Instruct plugin pipeline integration...")
    
    # Create Qwen3-4B-Instruct plugin
    plugin = Qwen3_4B_Instruct_2507_Plugin()
    
    # Verify that the plugin has all required pipeline methods
    methods_to_check = [
        'setup_pipeline',
        'execute_pipeline', 
        'create_pipeline_stages',
        'get_pipeline_stats',
        'get_pipeline_manager'
    ]
    
    for method_name in methods_to_check:
        assert hasattr(plugin, method_name), f"Plugin missing {method_name} method"
    
    print("  ✓ All pipeline methods present")
    
    # Test pipeline stats before setup
    stats = plugin.get_pipeline_stats()
    assert 'pipeline_enabled' in stats
    assert 'num_stages' in stats
    assert 'checkpoint_directory' in stats
    print("  ✓ Pipeline stats structure correct")
    
    # Test that pipeline is not enabled by default
    assert stats['pipeline_enabled'] == False
    print("  ✓ Pipeline disabled by default")


def test_base_plugin_interface_pipeline_methods():
    """Test that the base plugin interface includes pipeline methods."""
    print("Testing base plugin interface pipeline methods...")
    
    from inference_pio.common.base_plugin_interface import ModelPluginInterface, TextModelPluginInterface
    
    # Check that base classes have pipeline methods
    base_methods = [
        'setup_pipeline',
        'execute_pipeline',
        'create_pipeline_stages',
        'get_pipeline_manager',
        'get_pipeline_stats'
    ]
    
    for method_name in base_methods:
        assert hasattr(ModelPluginInterface, method_name), f"ModelPluginInterface missing {method_name}"
        assert hasattr(TextModelPluginInterface, method_name), f"TextModelPluginInterface missing {method_name}"
    
    print("  ✓ All base plugin interfaces have pipeline methods")


def test_pipeline_config_updates():
    """Test that pipeline configuration is properly updated in plugin configs."""
    print("Testing pipeline configuration updates...")
    
    import json
    
    # Check GLM-4.7 config
    glm_config_path = Path(__file__).parent / "plugin_configs" / "glm_4_7_config.json"
    with open(glm_config_path, 'r') as f:
        glm_config = json.load(f)
    
    assert 'enable_pipeline' in glm_config['parameters'], "GLM-4.7 config missing enable_pipeline"
    assert 'pipeline_checkpoint_dir' in glm_config['parameters'], "GLM-4.7 config missing pipeline_checkpoint_dir"
    assert 'pipeline_stages' in glm_config['parameters'], "GLM-4.7 config missing pipeline_stages"
    print("  ✓ GLM-4.7 config has pipeline settings")
    
    # Check Qwen3-4B-Instruct config
    qwen_config_path = Path(__file__).parent / "plugin_configs" / "qwen3_4b_instruct_2507_config.json"
    with open(qwen_config_path, 'r') as f:
        qwen_config = json.load(f)
    
    assert 'enable_pipeline' in qwen_config['parameters'], "Qwen3-4B-Instruct config missing enable_pipeline"
    assert 'pipeline_checkpoint_dir' in qwen_config['parameters'], "Qwen3-4B-Instruct config missing pipeline_checkpoint_dir"
    assert 'pipeline_stages' in qwen_config['parameters'], "Qwen3-4B-Instruct config missing pipeline_stages"
    print("  ✓ Qwen3-4B-Instruct config has pipeline settings")
    
    # Check other configs too
    qwen_coder_config_path = Path(__file__).parent / "plugin_configs" / "qwen3_coder_30b_a3b_instruct_config.json"
    with open(qwen_coder_config_path, 'r') as f:
        qwen_coder_config = json.load(f)
    
    assert 'enable_pipeline' in qwen_coder_config['parameters'], "Qwen3-Coder config missing enable_pipeline"
    print("  ✓ Qwen3-Coder config has pipeline settings")
    
    qwen_vl_config_path = Path(__file__).parent / "plugin_configs" / "qwen3_vl_2b_instruct_config.json"
    with open(qwen_vl_config_path, 'r') as f:
        qwen_vl_config = json.load(f)
    
    assert 'enable_pipeline' in qwen_vl_config['parameters'], "Qwen3-VL config missing enable_pipeline"
    print("  ✓ Qwen3-VL config has pipeline settings")


def main():
    """Run all integration tests."""
    print("Running comprehensive disk-based inference pipeline integration tests...\n")
    
    test_base_plugin_interface_pipeline_methods()
    print()
    
    test_glm_4_7_pipeline_integration()
    print()
    
    test_qwen3_4b_instruct_pipeline_integration()
    print()
    
    test_pipeline_config_updates()
    print()
    
    print("ALL INTEGRATION TESTS PASSED! Disk-based inference pipeline system is fully integrated.")


if __name__ == "__main__":
    main()