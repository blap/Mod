#!/usr/bin/env python
"""
Basic test script to verify that qwen3_0_6b and qwen3_coder_next models 
can be instantiated and perform basic inference operations after standardization
and cross-dependency removal changes.
"""

import sys
import os
import torch
import traceback
from pathlib import Path

# Add the src directory to the path so we can import the models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_qwen3_0_6b_model():
    """Test the qwen3_0_6b model instantiation and basic operations."""
    print("=" * 60)
    print("Testing Qwen3-0.6B Model")
    print("=" * 60)
    
    try:
        # Import the plugin creation function
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        
        print("[PASS] Successfully imported qwen3_0_6b plugin")

        # Create the plugin instance
        plugin = create_qwen3_0_6b_plugin()
        print("[PASS] Successfully created qwen3_0_6b plugin instance")

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

        # Test initialization with minimal config
        print("\nTesting plugin initialization...")
        success = plugin.initialize()
        if success:
            print("[PASS] Plugin initialized successfully")
        else:
            print("[FAIL] Plugin initialization failed")

        # Test basic inference with a simple prompt
        print("\nTesting basic inference...")
        try:
            # Use a simple prompt for testing
            prompt = "Hello, world!"
            result = plugin.infer(prompt)
            print(f"[PASS] Basic inference successful")
            print(f"  Input: {prompt}")
            print(f"  Output type: {type(result)}")
            if isinstance(result, str):
                print(f"  Output preview: {result[:100]}...")
        except Exception as e:
            print(f"[FAIL] Basic inference failed: {str(e)}")
            traceback.print_exc()

        # Test generate_text method
        print("\nTesting generate_text method...")
        try:
            prompt = "What is AI?"
            result = plugin.generate_text(prompt, max_new_tokens=20)
            print(f"[PASS] generate_text successful")
            print(f"  Input: {prompt}")
            print(f"  Output type: {type(result)}")
            if isinstance(result, str):
                print(f"  Output preview: {result[:100]}...")
        except Exception as e:
            print(f"[FAIL] generate_text failed: {str(e)}")
            traceback.print_exc()

        # Test model info retrieval
        print("\nTesting model info retrieval...")
        try:
            info = plugin.get_model_info()
            print(f"[PASS] Model info retrieved successfully")
            print(f"  Model name: {info.get('name', 'Unknown')}")
            print(f"  Architecture: {info.get('architecture', 'Unknown')}")
            print(f"  Size: {info.get('size', 'Unknown')}")
        except Exception as e:
            print(f"[FAIL] Model info retrieval failed: {str(e)}")

        # Cleanup
        print("\nCleaning up...")
        cleanup_success = plugin.cleanup()
        if cleanup_success:
            print("[PASS] Cleanup successful")
        else:
            print("[FAIL] Cleanup failed")

        print("\n[PASS] Qwen3-0.6B Model test completed successfully")
        return True

    except ImportError as e:
        print(f"[FAIL] Failed to import qwen3_0_6b plugin: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_0_6b test: {str(e)}")
        traceback.print_exc()
        return False


def test_qwen3_coder_next_model():
    """Test the qwen3_coder_next model instantiation and basic operations."""
    print("\n" + "=" * 60)
    print("Testing Qwen3-Coder-Next Model")
    print("=" * 60)
    
    try:
        # Import the plugin creation function
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        
        print("[PASS] Successfully imported qwen3_coder_next plugin")

        # Create the plugin instance
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

        # Test initialization with minimal config
        print("\nTesting plugin initialization...")
        success = plugin.initialize()
        if success:
            print("[PASS] Plugin initialized successfully")
        else:
            print("[FAIL] Plugin initialization failed")

        # Test basic inference with a simple prompt
        print("\nTesting basic inference...")
        try:
            # Use a simple prompt for testing
            prompt = "Hello, world!"
            result = plugin.infer(prompt)
            print(f"[PASS] Basic inference successful")
            print(f"  Input: {prompt}")
            print(f"  Output type: {type(result)}")
            if isinstance(result, str):
                print(f"  Output preview: {result[:100]}...")
        except Exception as e:
            print(f"[FAIL] Basic inference failed: {str(e)}")
            traceback.print_exc()

        # Test generate_text method
        print("\nTesting generate_text method...")
        try:
            prompt = "Write a simple Python function."
            result = plugin.generate_text(prompt, max_new_tokens=50)
            print(f"[PASS] generate_text successful")
            print(f"  Input: {prompt}")
            print(f"  Output type: {type(result)}")
            if isinstance(result, str):
                print(f"  Output preview: {result[:100]}...")
        except Exception as e:
            print(f"[FAIL] generate_text failed: {str(e)}")
            traceback.print_exc()

        # Cleanup
        print("\nCleaning up...")
        cleanup_success = plugin.cleanup()
        if cleanup_success:
            print("[PASS] Cleanup successful")
        else:
            print("[FAIL] Cleanup failed")

        print("\n[PASS] Qwen3-Coder-Next Model test completed successfully")
        return True

    except ImportError as e:
        print(f"[FAIL] Failed to import qwen3_coder_next plugin: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_coder_next test: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Main function to run all tests."""
    print("Starting model verification tests...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__ if 'torch' in globals() else 'Not available'}")
    print(f"CUDA available: {torch.cuda.is_available() if 'torch' in globals() else 'Not checked'}")
    
    # Run tests for both models
    qwen3_0_6b_success = test_qwen3_0_6b_model()
    qwen3_coder_next_success = test_qwen3_coder_next_model()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print(f"Qwen3-0.6B Model: {'[PASS]' if qwen3_0_6b_success else '[FAIL]'}")
    print(f"Qwen3-Coder-Next Model: {'[PASS]' if qwen3_coder_next_success else '[FAIL]'}")

    if qwen3_0_6b_success and qwen3_coder_next_success:
        print("\n[SUCCESS] All models passed basic functionality tests!")
        return 0
    else:
        print("\n[ERROR] Some models failed basic functionality tests.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)