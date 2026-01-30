"""
Test script to verify the optimization impact benchmark infrastructure
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that we can import the required modules."""
    print("Testing imports...")
    
    try:
        from inference_pio.models.glm_4_7.plugin import create_glm_4_7_plugin
        print("[OK] Successfully imported GLM-4-7 plugin")
    except ImportError as e:
        print(f"[ERROR] Failed to import GLM-4-7 plugin: {e}")
        return False

    try:
        from inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
        print("[OK] Successfully imported Qwen3-4B-Instruct-2507 plugin")
    except ImportError as e:
        print(f"[ERROR] Failed to import Qwen3-4B-Instruct-2507 plugin: {e}")
        return False

    try:
        from inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
        print("[OK] Successfully imported Qwen3-Coder-30B plugin")
    except ImportError as e:
        print(f"[ERROR] Failed to import Qwen3-Coder-30B plugin: {e}")
        return False

    try:
        from inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
        print("[OK] Successfully imported Qwen3-VL-2B plugin")
    except ImportError as e:
        print(f"[ERROR] Failed to import Qwen3-VL-2B plugin: {e}")
        return False

    return True

def test_plugin_creation():
    """Test that we can create plugin instances."""
    print("\nTesting plugin creation...")

    try:
        from inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
        plugin = create_qwen3_vl_2b_instruct_plugin()
        print("[OK] Successfully created Qwen3-VL-2B plugin instance")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create plugin instance: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without running full benchmarks."""
    print("\nTesting basic functionality...")

    try:
        import torch
        import psutil
        print("[OK] PyTorch and psutil are available")
    except ImportError as e:
        print(f"[ERROR] Missing required packages: {e}")
        return False

    # Test basic torch functionality
    try:
        x = torch.tensor([1, 2, 3])
        y = x * 2
        print("[OK] PyTorch tensor operations work")
    except Exception as e:
        print(f"[ERROR] PyTorch operations failed: {e}")
        return False

    # Test basic psutil functionality
    try:
        memory = psutil.virtual_memory()
        print(f"[OK] psutil memory monitoring works: {memory.percent}% used")
    except Exception as e:
        print(f"[ERROR] psutil operations failed: {e}")
        return False

    return True

def main():
    """Run all tests."""
    print("Testing optimization impact benchmark infrastructure...\n")
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test plugin creation
    if not test_plugin_creation():
        all_tests_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_tests_passed = False
    
    print(f"\nInfrastructure test {'PASSED' if all_tests_passed else 'FAILED'}")
    return all_tests_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)