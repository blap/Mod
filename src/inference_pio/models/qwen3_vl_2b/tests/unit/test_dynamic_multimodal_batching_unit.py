#!/usr/bin/env python3
"""
Unit test for Dynamic Multimodal Batching in Qwen3-VL-2B model - focusing on core functionality.
"""

import sys
import os
import inspect

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin


def test_model_has_required_methods():
    """Test that the model has the required methods for dynamic multimodal batching."""
    print("Testing Model Required Methods...")
    
    # Check that the model class has the required methods
    model_methods = [method for method in dir(Qwen3VL2BModel) if not method.startswith('_')]
    
    required_methods = [
        '_initialize_dynamic_multimodal_batching',
        'generate_with_adaptive_batching',
        '_generate_with_regular_adaptive_batching',
        '_generate_with_dynamic_multimodal_batching',
        '_generate_multimodal_chunk'
    ]
    
    missing_methods = []
    for method in required_methods:
        if method not in model_methods:
            missing_methods.append(method)
    
    print(f"Required methods: {required_methods}")
    print(f"Missing methods: {missing_methods}")
    
    if missing_methods:
        print(f"ERROR: Missing methods: {missing_methods}")
        return False
    
    print("[PASS] All required methods exist in the model")
    return True


def test_plugin_has_required_methods():
    """Test that the plugin has the required methods for dynamic multimodal batching."""
    print("\nTesting Plugin Required Methods...")
    
    # Check that the plugin class has the required methods
    plugin_methods = [method for method in dir(Qwen3_VL_2B_Plugin) if not method.startswith('_')]
    
    required_methods = [
        'setup_dynamic_multimodal_batching',
        'get_optimal_multimodal_batch_size'
    ]
    
    missing_methods = []
    for method in required_methods:
        if method not in plugin_methods:
            missing_methods.append(method)
    
    print(f"Required methods: {required_methods}")
    print(f"Missing methods: {missing_methods}")
    
    if missing_methods:
        print(f"ERROR: Missing methods: {missing_methods}")
        return False
    
    print("[PASS] All required methods exist in the plugin")
    return True


def test_config_parameter_integration():
    """Test that the config supports the required parameters."""
    print("\nTesting Config Parameter Integration...")
    
    config = Qwen3VL2BConfig()
    
    # Check that the config has the required attributes
    required_attrs = [
        'enable_dynamic_multimodal_batching',
        'initial_batch_size',
        'min_batch_size',
        'max_batch_size',
        'text_weight',
        'image_weight',
        'complexity_threshold_low',
        'complexity_threshold_high'
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(config, attr):
            missing_attrs.append(attr)
    
    print(f"Required attributes: {required_attrs}")
    print(f"Missing attributes: {missing_attrs}")
    
    if missing_attrs:
        print(f"ERROR: Missing attributes: {missing_attrs}")
        return False
    
    print("[PASS] All required config attributes exist")
    return True


def test_method_signatures():
    """Test that the method signatures are correct."""
    print("\nTesting Method Signatures...")
    
    # Check signature of generate_with_adaptive_batching
    sig = inspect.signature(Qwen3VL2BModel.generate_with_adaptive_batching)
    params = list(sig.parameters.keys())
    
    print(f"generate_with_adaptive_batching parameters: {params}")
    
    # Should have inputs and **kwargs
    expected_params = ['self', 'inputs']
    for expected in expected_params:
        if expected not in params:
            print(f"ERROR: Missing parameter {expected} in generate_with_adaptive_batching")
            return False
    
    # Check signature of setup_dynamic_multimodal_batching in plugin
    sig = inspect.signature(Qwen3_VL_2B_Plugin.setup_dynamic_multimodal_batching)
    params = list(sig.parameters.keys())
    
    print(f"setup_dynamic_multimodal_batching parameters: {params}")
    
    # Should have self and **kwargs
    expected_params = ['self']
    for expected in expected_params:
        if expected not in params:
            print(f"ERROR: Missing parameter {expected} in setup_dynamic_multimodal_batching")
            return False
    
    print("[PASS] Method signatures are correct")
    return True


def test_import_statements():
    """Test that the required imports are in place."""
    print("\nTesting Import Statements...")
    
    # Read the model file to check for imports
    model_file_path = os.path.join(os.path.dirname(__file__), 'model.py')
    with open(model_file_path, 'r') as f:
        model_content = f.read()
    
    # Check for required import
    required_import = "from ...common.dynamic_multimodal_batching import get_dynamic_multimodal_batch_manager, DynamicMultimodalBatchManager"
    if required_import not in model_content:
        print(f"ERROR: Missing import in model.py: {required_import}")
        return False
    
    # Read the plugin file to check for imports
    plugin_file_path = os.path.join(os.path.dirname(__file__), 'plugin.py')
    with open(plugin_file_path, 'r') as f:
        plugin_content = f.read()
    
    # Check for the setup method
    if "def setup_dynamic_multimodal_batching" not in plugin_content:
        print("ERROR: setup_dynamic_multimodal_batching method not found in plugin.py")
        return False
    
    print("[PASS] Import statements and method definitions are correct")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("Unit Test for Dynamic Multimodal Batching Implementation")
    print("=" * 70)

    all_passed = True
    
    all_passed &= test_model_has_required_methods()
    all_passed &= test_plugin_has_required_methods()
    all_passed &= test_config_parameter_integration()
    all_passed &= test_method_signatures()
    all_passed &= test_import_statements()

    if all_passed:
        print("\n" + "=" * 70)
        print("All unit tests passed! [PASS]")
        print("=" * 70)
        return 0
    else:
        print("\n" + "=" * 70)
        print("Some tests failed! [FAIL]")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())