"""
Verification script for Model Surgery implementation.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_plugin_import_and_surgery_methods(plugin_module_path, plugin_class_name):
    """Test importing a plugin and checking for surgery methods."""
    try:
        # Import the plugin module
        import importlib
        module = importlib.import_module(plugin_module_path)
        plugin_class = getattr(module, plugin_class_name)
        
        # Create an instance
        plugin = plugin_class()
        
        # Check for surgery methods
        has_surgery_methods = all([
            hasattr(plugin, 'setup_model_surgery'),
            hasattr(plugin, 'perform_model_surgery'),
            hasattr(plugin, 'restore_model_from_surgery'),
            hasattr(plugin, 'analyze_model_for_surgery'),
            hasattr(plugin, 'get_surgery_stats'),
            hasattr(plugin, 'enable_model_surgery')
        ])
        
        print(f"{plugin_class_name}: {'PASS' if has_surgery_methods else 'FAIL'} Surgery methods available")
        return has_surgery_methods
    except Exception as e:
        print(f"{plugin_class_name}: Error - {e}")
        return False

def main():
    print("Verifying Model Surgery Implementation Across Plugins")
    print("=" * 60)
    
    plugins_to_test = [
        ("src.inference_pio.models.glm_4_7.plugin", "GLM_4_7_Plugin"),
        ("src.inference_pio.models.qwen3_4b_instruct_2507.plugin", "Qwen3_4B_Instruct_2507_Plugin"),
        ("src.inference_pio.models.qwen3_coder_30b.plugin", "Qwen3_Coder_30B_Plugin"),
        ("src.inference_pio.models.qwen3_vl_2b.plugin", "Qwen3_VL_2B_Instruct_Plugin"),
    ]
    
    all_passed = True
    
    for module_path, class_name in plugins_to_test:
        result = test_plugin_import_and_surgery_methods(module_path, class_name)
        all_passed = all_passed and result
    
    print("=" * 60)
    print(f"All plugins have surgery methods: {'YES' if all_passed else 'NO'}")
    
    # Test the core model surgery system
    try:
        from src.inference_pio.common.model_surgery import ModelSurgerySystem
        surgery_system = ModelSurgerySystem()
        print(f"ModelSurgerySystem: PASS Available")
    except Exception as e:
        print(f"ModelSurgerySystem: FAIL Error - {e}")
        all_passed = False

    # Test that base plugin interface has surgery methods
    try:
        from src.inference_pio.common.base_plugin_interface import ModelPluginInterface
        has_surgery = all([
            hasattr(ModelPluginInterface, 'setup_model_surgery'),
            hasattr(ModelPluginInterface, 'perform_model_surgery'),
            hasattr(ModelPluginInterface, 'restore_model_from_surgery'),
            hasattr(ModelPluginInterface, 'analyze_model_for_surgery'),
            hasattr(ModelPluginInterface, 'get_surgery_stats'),
            hasattr(ModelPluginInterface, 'enable_model_surgery')
        ])
        print(f"Base Plugin Interface: {'PASS' if has_surgery else 'FAIL'} Surgery methods available")
    except Exception as e:
        print(f"Base Plugin Interface: FAIL Error - {e}")
        all_passed = False

    print("=" * 60)
    print(f"Overall verification: {'SUCCESS' if all_passed else 'FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)