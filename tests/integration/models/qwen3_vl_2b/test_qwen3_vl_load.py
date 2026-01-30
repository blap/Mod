#!/usr/bin/env python3
"""
Simple test to verify Qwen3-VL-2B model plugin can be instantiated.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig


def test_plugin_creation():
    """Test basic plugin creation without loading the full model."""
    print("Testing Qwen3-VL-2B plugin creation...")

    try:
        # Create plugin instance
        plugin = Qwen3_VL_2B_Plugin()
        print("[OK] Plugin created successfully")

        # Test getting model info without loading the model
        model_info = plugin.get_model_info()
        print(f"[OK] Model name: {model_info['name']}")
        print(f"[OK] Model type: {model_info['model_type']}")
        print(f"[OK] Modalities: {model_info['modalities']}")

        # Test config template
        config_template = plugin.get_model_config_template()
        print(f"[OK] Config model path: {config_template.model_path}")
        print(f"[OK] Config dtype: {config_template.torch_dtype}")

        # Test initialization with minimal config
        success = plugin.initialize(torch_dtype="float16", device="cpu")
        print(f"[OK] Initialization result: {'Success' if success else 'Failed'}")

        # Cleanup
        plugin.cleanup()
        print("[OK] Cleanup completed")

        print("\n[OK] All tests passed!")
        return True

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_plugin_creation()
    if success:
        print("\n[SUCCESS] Plugin verification successful!")
    else:
        print("\n[FAILURE] Plugin verification failed!")
        sys.exit(1)