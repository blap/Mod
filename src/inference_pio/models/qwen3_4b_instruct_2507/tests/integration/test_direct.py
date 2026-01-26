"""
Direct test to verify that all plugins implement the standardized interface correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Add the src directory to the path to avoid import conflicts
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime

def test_plugins_directly():
    """Test that all plugins can be imported and implement the required interface."""
    
    print("Testing plugin imports and interface compliance...")
    
    # Test GLM-4-7 plugin
    try:
        import importlib.util
        # Directly load the plugin module
        spec = importlib.util.spec_from_file_location(
            "glm_4_7_plugin", 
            "C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/models/glm_4_7/plugin.py"
        )
        glm_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(glm_module)
        
        plugin = glm_module.GLM_4_7_Plugin()
        print("✓ GLM-4-7 plugin imported successfully")
        
        # Check that required methods exist
        required_methods = ['initialize', 'load_model', 'infer', 'cleanup', 'supports_config']
        for method in required_methods:
            if hasattr(plugin, method):
                print(f"  ✓ Method {method} exists")
            else:
                print(f"  X Method {method} missing")
                
        # Check that text-specific methods exist
        text_methods = ['tokenize', 'detokenize', 'generate_text']
        for method in text_methods:
            if hasattr(plugin, method):
                print(f"  ✓ Text method {method} exists")
            else:
                print(f"  X Text method {method} missing")

    except Exception as e:
        print(f"X Failed to import GLM-4-7 plugin: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Qwen3-4B-Instruct-2507 plugin
    try:
        import importlib.util
        # Directly load the plugin module
        spec = importlib.util.spec_from_file_location(
            "qwen3_4b_instruct_plugin", 
            "C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/models/qwen3_4b_instruct_2507/plugin.py"
        )
        qwen3_4b_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qwen3_4b_module)
        
        plugin = qwen3_4b_module.Qwen3_4B_Instruct_2507_Plugin()
        print("✓ Qwen3-4B-Instruct-2507 plugin imported successfully")
        
        # Check that required methods exist
        required_methods = ['initialize', 'load_model', 'infer', 'cleanup', 'supports_config']
        for method in required_methods:
            if hasattr(plugin, method):
                print(f"  ✓ Method {method} exists")
            else:
                print(f"  X Method {method} missing")
                
        # Check that text-specific methods exist
        text_methods = ['tokenize', 'detokenize', 'generate_text']
        for method in text_methods:
            if hasattr(plugin, method):
                print(f"  ✓ Text method {method} exists")
            else:
                print(f"  X Text method {method} missing")

    except Exception as e:
        print(f"X Failed to import Qwen3-4B-Instruct-2507 plugin: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Qwen3-Coder-30B plugin
    try:
        import importlib.util
        # Directly load the plugin module
        spec = importlib.util.spec_from_file_location(
            "qwen3_coder_plugin", 
            "C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/models/qwen3_coder_30b/plugin.py"
        )
        qwen3_coder_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qwen3_coder_module)
        
        plugin = qwen3_coder_module.Qwen3_Coder_30B_Plugin()
        print("✓ Qwen3-Coder-30B plugin imported successfully")
        
        # Check that required methods exist
        required_methods = ['initialize', 'load_model', 'infer', 'cleanup', 'supports_config']
        for method in required_methods:
            if hasattr(plugin, method):
                print(f"  ✓ Method {method} exists")
            else:
                print(f"  X Method {method} missing")
                
        # Check that text-specific methods exist
        text_methods = ['tokenize', 'detokenize', 'generate_text']
        for method in text_methods:
            if hasattr(plugin, method):
                print(f"  ✓ Text method {method} exists")
            else:
                print(f"  X Text method {method} missing")

    except Exception as e:
        print(f"X Failed to import Qwen3-Coder-30B plugin: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Qwen3-VL-2B plugin
    try:
        import importlib.util
        # Directly load the plugin module
        spec = importlib.util.spec_from_file_location(
            "qwen3_vl_plugin", 
            "C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/models/qwen3_vl_2b/plugin.py"
        )
        qwen3_vl_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(qwen3_vl_module)
        
        plugin = qwen3_vl_module.Qwen3_VL_2B_Instruct_Plugin()
        print("✓ Qwen3-VL-2B plugin imported successfully")
        
        # Check that required methods exist
        required_methods = ['initialize', 'load_model', 'infer', 'cleanup', 'supports_config']
        for method in required_methods:
            if hasattr(plugin, method):
                print(f"  ✓ Method {method} exists")
            else:
                print(f"  X Method {method} missing")
                
        # Check that text-specific methods exist
        text_methods = ['tokenize', 'detokenize', 'generate_text']
        for method in text_methods:
            if hasattr(plugin, method):
                print(f"  ✓ Text method {method} exists")
            else:
                print(f"  X Text method {method} missing")

    except Exception as e:
        print(f"X Failed to import Qwen3-VL-2B plugin: {e}")
        import traceback
        traceback.print_exc()

    print("\nInterface compliance check completed!")


if __name__ == "__main__":
    test_plugins_directly()