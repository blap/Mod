#!/usr/bin/env python
"""
Quick test script to verify that qwen3_0_6b and qwen3_coder_next models 
can be imported and basic plugin structures work after standardization
and cross-dependency removal changes.
"""

import sys
import os
import traceback
from pathlib import Path

# Add the src directory to the path so we can import the models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_qwen3_0_6b_imports():
    """Test that qwen3_0_6b model can be imported without loading the full model."""
    print("=" * 60)
    print("Testing Qwen3-0.6B Model Imports")
    print("=" * 60)
    
    try:
        # Import the plugin creation function
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        print("[PASS] Successfully imported qwen3_0_6b plugin")
        
        # Import the model class
        from src.inference_pio.models.qwen3_0_6b.model import Qwen3_0_6B_Model
        print("[PASS] Successfully imported qwen3_0_6b model class")
        
        # Import the config class
        from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config
        print("[PASS] Successfully imported qwen3_0_6b config class")
        
        # Test creating a plugin instance (without initializing the full model)
        plugin = create_qwen3_0_6b_plugin()
        print("[PASS] Successfully created qwen3_0_6b plugin instance")
        
        # Check if plugin has required methods
        required_methods = [
            'initialize', 'infer', 'generate_text', 'cleanup', 
            'supports_config', 'tokenize', 'detokenize', 'get_model_info'
        ]
        
        for method in required_methods:
            if hasattr(plugin, method):
                print(f"[PASS] Method '{method}' exists")
            else:
                print(f"[FAIL] Method '{method}' missing")
        
        # Test creating a config instance
        config = Qwen3_0_6B_Config()
        print(f"[PASS] Created config instance with model_name: {config.model_name}")
        
        print("\n[PASS] Qwen3-0.6B Model import tests completed successfully")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Failed to import qwen3_0_6b components: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_0_6b import test: {str(e)}")
        traceback.print_exc()
        return False


def test_qwen3_coder_next_imports():
    """Test that qwen3_coder_next model can be imported without loading the full model."""
    print("\n" + "=" * 60)
    print("Testing Qwen3-Coder-Next Model Imports")
    print("=" * 60)
    
    try:
        # Import the plugin creation function
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        print("[PASS] Successfully imported qwen3_coder_next plugin")
        
        # Import the model class
        from src.inference_pio.models.qwen3_coder_next.model import Qwen3CoderNextModel
        print("[PASS] Successfully imported qwen3_coder_next model class")
        
        # Import the config class
        from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig
        print("[PASS] Successfully imported qwen3_coder_next config class")
        
        # Test creating a plugin instance (without initializing the full model)
        plugin = create_qwen3_coder_next_plugin()
        print("[PASS] Successfully created qwen3_coder_next plugin instance")
        
        # Check if plugin has required methods
        required_methods = [
            'initialize', 'infer', 'generate_text', 'cleanup', 
            'supports_config', 'tokenize', 'detokenize'
        ]
        
        for method in required_methods:
            if hasattr(plugin, method):
                print(f"[PASS] Method '{method}' exists")
            else:
                print(f"[FAIL] Method '{method}' missing")
        
        # Test creating a config instance
        config = Qwen3CoderNextConfig()
        print(f"[PASS] Created config instance with model_name: {config.model_name}")
        
        print("\n[PASS] Qwen3-Coder-Next Model import tests completed successfully")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Failed to import qwen3_coder_next components: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_coder_next import test: {str(e)}")
        traceback.print_exc()
        return False


def test_model_independence():
    """Test that models can work independently after cross-dependency removal."""
    print("\n" + "=" * 60)
    print("Testing Model Independence")
    print("=" * 60)
    
    try:
        # Test that each model can be imported separately
        print("Testing individual imports...")
        
        # Import qwen3_0_6b only
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        plugin_06b = create_qwen3_0_6b_plugin()
        print("[PASS] qwen3_0_6b can be imported and instantiated independently")
        
        # Import qwen3_coder_next only
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        plugin_coder = create_qwen3_coder_next_plugin()
        print("[PASS] qwen3_coder_next can be imported and instantiated independently")
        
        # Verify they are different plugins
        if type(plugin_06b).__name__ != type(plugin_coder).__name__:
            print("[PASS] Plugins are different types as expected")
        else:
            print("[FAIL] Plugins appear to be the same type unexpectedly")
        
        print("\n[PASS] Model independence tests completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error during independence test: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Main function to run all tests."""
    print("Starting quick model verification tests...")
    print(f"Python version: {sys.version}")
    
    # Run tests for both models
    qwen3_0_6b_success = test_qwen3_0_6b_imports()
    qwen3_coder_next_success = test_qwen3_coder_next_imports()
    independence_success = test_model_independence()
    
    # Summary
    print("\n" + "=" * 60)
    print("QUICK TEST SUMMARY")
    print("=" * 60)
    
    print(f"Qwen3-0.6B Model Imports: {'[PASS]' if qwen3_0_6b_success else '[FAIL]'}")
    print(f"Qwen3-Coder-Next Model Imports: {'[PASS]' if qwen3_coder_next_success else '[FAIL]'}")
    print(f"Model Independence: {'[PASS]' if independence_success else '[FAIL]'}")
    
    all_passed = qwen3_0_6b_success and qwen3_coder_next_success and independence_success
    
    if all_passed:
        print("\n[SUCCESS] All models passed import and independence tests!")
        print("Models can be imported and work independently after standardization.")
        return 0
    else:
        print("\n[ERROR] Some tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)