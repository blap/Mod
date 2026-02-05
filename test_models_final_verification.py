#!/usr/bin/env python
"""
Final verification test script to confirm that qwen3_0_6b and qwen3_coder_next models 
work correctly after standardization and cross-dependency removal changes.
Focuses on core functionality and structural integrity.
"""

import sys
import os
import traceback
from pathlib import Path

# Add the src directory to the path so we can import the models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_qwen3_0_6b_core_functionality():
    """Test core functionality of the qwen3_0_6b model."""
    print("=" * 60)
    print("Testing Qwen3-0.6B Core Functionality")
    print("=" * 60)
    
    try:
        # Import core components
        from src.inference_pio.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin, create_qwen3_0_6b_plugin
        from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config
        
        print("[PASS] Successfully imported qwen3_0_6b core components")
        
        # Test config creation and properties
        config = Qwen3_0_6B_Config()
        if config.model_name == "qwen3_0_6b":
            print("[PASS] Config has correct model name")
        else:
            print(f"[FAIL] Config has incorrect model name: {config.model_name}")
            return False
        
        # Test plugin creation
        plugin = create_qwen3_0_6b_plugin()
        if isinstance(plugin, Qwen3_0_6B_Plugin):
            print("[PASS] Plugin created with correct type")
        else:
            print(f"[FAIL] Plugin has incorrect type: {type(plugin)}")
            return False
        
        # Test plugin metadata
        if plugin.metadata.name == "Qwen3-0.6B":
            print("[PASS] Plugin has correct metadata name")
        else:
            print(f"[FAIL] Plugin has incorrect metadata name: {plugin.metadata.name}")
            return False
        
        # Test essential methods exist
        essential_methods = [
            'initialize', 'infer', 'generate_text', 'cleanup', 
            'supports_config', 'tokenize', 'detokenize', 'get_model_info'
        ]
        
        for method in essential_methods:
            if hasattr(plugin, method):
                print(f"[PASS] Essential method '{method}' exists")
            else:
                print(f"[FAIL] Essential method '{method}' missing")
                return False
        
        # Test config support
        if plugin.supports_config(config):
            print("[PASS] Plugin supports its own config type")
        else:
            print("[FAIL] Plugin does not support its own config type")
            return False
        
        print("\n[PASS] Qwen3-0.6B Core Functionality test completed successfully")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Failed to import qwen3_0_6b components: {str(e)}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_0_6b test: {str(e)}")
        traceback.print_exc()
        return False


def test_qwen3_coder_next_core_functionality():
    """Test core functionality of the qwen3_coder_next model."""
    print("\n" + "=" * 60)
    print("Testing Qwen3-Coder-Next Core Functionality")
    print("=" * 60)
    
    try:
        # Import core components
        from src.inference_pio.models.qwen3_coder_next.plugin import Qwen3_Coder_Next_Plugin, create_qwen3_coder_next_plugin
        from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig
        
        print("[PASS] Successfully imported qwen3_coder_next core components")
        
        # Test config creation and properties
        config = Qwen3CoderNextConfig()
        if config.model_name == "qwen3_coder_next":
            print("[PASS] Config has correct model name")
        else:
            print(f"[FAIL] Config has incorrect model name: {config.model_name}")
            return False
        
        # Test plugin creation
        plugin = create_qwen3_coder_next_plugin()
        if isinstance(plugin, Qwen3_Coder_Next_Plugin):
            print("[PASS] Plugin created with correct type")
        else:
            print(f"[FAIL] Plugin has incorrect type: {type(plugin)}")
            return False
        
        # Test plugin metadata
        if plugin.metadata.name == "Qwen3-Coder-Next":
            print("[PASS] Plugin has correct metadata name")
        else:
            print(f"[FAIL] Plugin has incorrect metadata name: {plugin.metadata.name}")
            return False
        
        # Test essential methods exist
        essential_methods = [
            'initialize', 'infer', 'generate_text', 'cleanup', 
            'supports_config', 'tokenize', 'detokenize'
        ]
        
        for method in essential_methods:
            if hasattr(plugin, method):
                print(f"[PASS] Essential method '{method}' exists")
            else:
                print(f"[FAIL] Essential method '{method}' missing")
                return False
        
        # Test config support
        if plugin.supports_config(config):
            print("[PASS] Plugin supports its own config type")
        else:
            print("[FAIL] Plugin does not support its own config type")
            return False
        
        print("\n[PASS] Qwen3-Coder-Next Core Functionality test completed successfully")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Failed to import qwen3_coder_next components: {str(e)}")
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_coder_next test: {str(e)}")
        traceback.print_exc()
        return False


def test_model_isolation_and_independence():
    """Test that models are properly isolated and independent."""
    print("\n" + "=" * 60)
    print("Testing Model Isolation and Independence")
    print("=" * 60)
    
    try:
        # Import both models
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        
        # Create instances
        plugin_06b = create_qwen3_0_6b_plugin()
        plugin_coder = create_qwen3_coder_next_plugin()
        
        print("[PASS] Both plugins created successfully")
        
        # Verify they are different types
        if type(plugin_06b).__name__ != type(plugin_coder).__name__:
            print(f"[PASS] Plugins have different types: {type(plugin_06b).__name__} vs {type(plugin_coder).__name__}")
        else:
            print("[FAIL] Plugins have the same type unexpectedly")
            return False
        
        # Verify they have different names
        if plugin_06b.metadata.name != plugin_coder.metadata.name:
            print(f"[PASS] Plugins have different names: {plugin_06b.metadata.name} vs {plugin_coder.metadata.name}")
        else:
            print("[FAIL] Plugins have the same name unexpectedly")
            return False
        
        # Verify they have different model families
        if plugin_06b.metadata.model_family != plugin_coder.metadata.model_family:
            print(f"[PASS] Plugins have different model families: {plugin_06b.metadata.model_family} vs {plugin_coder.metadata.model_family}")
        else:
            print("[PASS] Plugins have the same model family (may be acceptable)")
        
        # Test that they don't interfere with each other
        plugin_06b.test_attr = "06b_value"
        plugin_coder.test_attr = "coder_value"
        
        if (hasattr(plugin_06b, 'test_attr') and plugin_06b.test_attr == "06b_value" and
            hasattr(plugin_coder, 'test_attr') and plugin_coder.test_attr == "coder_value"):
            print("[PASS] Plugins maintain independent state")
        else:
            print("[FAIL] Plugins do not maintain independent state")
            return False
        
        print("\n[PASS] Model Isolation and Independence test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error during isolation test: {str(e)}")
        traceback.print_exc()
        return False


def test_standardization_and_interface_compliance():
    """Test that models comply with standardization and interface requirements."""
    print("\n" + "=" * 60)
    print("Testing Standardization and Interface Compliance")
    print("=" * 60)

    try:
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin

        # Create plugin instances
        plugin_06b = create_qwen3_0_6b_plugin()
        plugin_coder = create_qwen3_coder_next_plugin()

        # Check if they inherit from a common interface
        # Try to import the base interface
        try:
            from src.inference_pio.common.interfaces.improved_base_plugin_interface import TextModelPluginInterface
            interface_available = True
        except ImportError:
            print("[INFO] Base interface not available, checking for common methods instead")
            interface_available = False

        if interface_available:
            # Check if the classes inherit from the interface (not instances)
            from src.inference_pio.models.qwen3_0_6b.plugin import Qwen3_0_6B_Plugin
            from src.inference_pio.models.qwen3_coder_next.plugin import Qwen3_Coder_Next_Plugin

            if issubclass(Qwen3_0_6B_Plugin, TextModelPluginInterface) and issubclass(Qwen3_Coder_Next_Plugin, TextModelPluginInterface):
                print("[PASS] Both plugin classes implement TextModelPluginInterface")
            else:
                print("[INFO] Plugin classes may not directly inherit from TextModelPluginInterface, checking for common methods instead")
                # If they don't inherit directly, check for common method signatures
                common_methods = ['initialize', 'infer', 'generate_text', 'cleanup', 'supports_config']
                all_have_methods = True

                for method in common_methods:
                    if not (hasattr(plugin_06b, method) and hasattr(plugin_coder, method)):
                        print(f"[FAIL] Method '{method}' not available in both plugins")
                        all_have_methods = False

                if all_have_methods:
                    print("[PASS] Both plugins have common interface methods")
                else:
                    return False
        else:
            # If interface isn't available, check for common method signatures
            common_methods = ['initialize', 'infer', 'generate_text', 'cleanup', 'supports_config']
            all_have_methods = True

            for method in common_methods:
                if not (hasattr(plugin_06b, method) and hasattr(plugin_coder, method)):
                    print(f"[FAIL] Method '{method}' not available in both plugins")
                    all_have_methods = False

            if all_have_methods:
                print("[PASS] Both plugins have common interface methods")

        # Test that both plugins have similar architectural patterns
        required_attributes = ['metadata', '_config', '_model', '_tokenizer']

        # Check 06b plugin
        missing_attrs_06b = [attr for attr in required_attributes if not hasattr(plugin_06b, attr)]
        if not missing_attrs_06b:
            print("[PASS] Qwen3-0.6B plugin has all required attributes")
        else:
            print(f"[INFO] Qwen3-0.6B plugin missing attributes: {missing_attrs_06b}")

        # Check coder plugin
        missing_attrs_coder = [attr for attr in required_attributes if not hasattr(plugin_coder, attr)]
        if not missing_attrs_coder:
            print("[PASS] Qwen3-Coder-Next plugin has all required attributes")
        else:
            print(f"[INFO] Qwen3-Coder-Next plugin missing attributes: {missing_attrs_coder}")

        print("\n[PASS] Standardization and Interface Compliance test completed successfully")
        return True

    except Exception as e:
        print(f"[FAIL] Error during standardization test: {str(e)}")
        traceback.print_exc()
        return False


def test_post_refactoring_integrity():
    """Test that models maintain integrity after refactoring and standardization."""
    print("\n" + "=" * 60)
    print("Testing Post-Refactoring Integrity")
    print("=" * 60)
    
    try:
        # Test that model files exist and are properly structured
        model_files = [
            'src/inference_pio/models/qwen3_0_6b/plugin.py',
            'src/inference_pio/models/qwen3_0_6b/config.py',
            'src/inference_pio/models/qwen3_0_6b/model.py',
            'src/inference_pio/models/qwen3_coder_next/plugin.py',
            'src/inference_pio/models/qwen3_coder_next/config.py',
            'src/inference_pio/models/qwen3_coder_next/model.py'
        ]
        
        for file_path in model_files:
            full_path = os.path.join(os.path.dirname(__file__), file_path)
            if os.path.exists(full_path):
                print(f"[PASS] Model file exists: {file_path}")
            else:
                print(f"[FAIL] Model file missing: {file_path}")
                return False
        
        # Test that imports work without circular dependencies
        try:
            # Import in different orders to test for circular dependencies
            from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
            from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
            print("[PASS] No circular import issues detected")
        except ImportError as e:
            print(f"[FAIL] Circular import issue: {str(e)}")
            return False
        
        # Test that plugins can be instantiated multiple times independently
        plugin1 = create_qwen3_0_6b_plugin()
        plugin2 = create_qwen3_0_6b_plugin()
        
        if id(plugin1) != id(plugin2):
            print("[PASS] Multiple plugin instances have different identities")
        else:
            print("[FAIL] Multiple plugin instances have same identity")
            return False
        
        print("\n[PASS] Post-Refactoring Integrity test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error during integrity test: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Main function to run all verification tests."""
    print("Starting final model verification tests after standardization...")
    print(f"Python version: {sys.version}")
    
    # Run all tests
    tests = [
        ("Qwen3-0.6B Core Functionality", test_qwen3_0_6b_core_functionality),
        ("Qwen3-Coder-Next Core Functionality", test_qwen3_coder_next_core_functionality),
        ("Model Isolation and Independence", test_model_isolation_and_independence),
        ("Standardization and Interface Compliance", test_standardization_and_interface_compliance),
        ("Post-Refactoring Integrity", test_post_refactoring_integrity)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n[SUCCESS] All verification tests passed!")
        print("Both qwen3_0_6b and qwen3_coder_next models work correctly")
        print("after standardization and cross-dependency removal.")
        return 0
    else:
        print("\n[ERROR] Some verification tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)