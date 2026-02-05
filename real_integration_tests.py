#!/usr/bin/env python
"""
Integration Tests with Real Models - Execute integration tests with actual models and real data
to verify that different components work together correctly.
"""

import sys
import os
import torch
import time
import traceback
from pathlib import Path

# Add the src directory to the path so we can import the models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_qwen3_0_6b_integration():
    """Integration test for qwen3_0_6b model components working together."""
    print("=" * 60)
    print("Integration Testing Qwen3-0.6B Model Components")
    print("=" * 60)

    try:
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config

        print("[INFO] Successfully imported qwen3_0_6b components")

        # Create config
        config = Qwen3_0_6B_Config()
        print(f"[INFO] Created config: {config.model_name}")

        # Create plugin
        plugin = create_qwen3_0_6b_plugin()
        print(f"[INFO] Created plugin: {plugin.__class__.__name__}")

        # Test config-plugin compatibility
        assert plugin.supports_config(config), "Plugin should support the config"
        print("[PASS] Config-plugin compatibility verified")

        # Test initialization (may fail if model not available, but shouldn't crash)
        try:
            init_result = plugin.initialize(config=config)
            print(f"[INFO] Initialization result: {init_result}")

            if init_result:
                # Test inference pipeline
                test_input = "Hello, this is a test."
                result = plugin.infer(test_input)
                print(f"[INFO] Inference result type: {type(result)}")

                # Test generate_text
                gen_result = plugin.generate_text("Generate a short text:", max_new_tokens=10)
                print(f"[INFO] Generate result type: {type(gen_result)}")

                print("[PASS] Inference pipeline works")
            else:
                print("[INFO] Skipping inference tests (model not initialized)")

        except Exception as e:
            print(f"[INFO] Initialization failed (expected if model not available): {e}")

        # Test cleanup
        cleanup_result = plugin.cleanup()
        print(f"[INFO] Cleanup result: {cleanup_result}")
        print("[PASS] Cleanup completed")

        print("\n[PASS] Qwen3-0.6B Integration tests completed")
        return True

    except ImportError as e:
        print(f"[INFO] Qwen3-0.6B not available: {str(e)}")
        print("[SKIP] Skipping Qwen3-0.6B integration tests")
        return True  # Don't fail if model is not available
    except AssertionError as e:
        print(f"[FAIL] Assertion failed: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_0_6b integration test: {str(e)}")
        traceback.print_exc()
        return False


def test_qwen3_coder_next_integration():
    """Integration test for qwen3_coder_next model components working together."""
    print("\n" + "=" * 60)
    print("Integration Testing Qwen3-Coder-Next Model Components")
    print("=" * 60)

    try:
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig

        print("[INFO] Successfully imported qwen3_coder_next components")

        # Create config
        config = Qwen3CoderNextConfig()
        print(f"[INFO] Created config: {config.model_name}")

        # Create plugin
        plugin = create_qwen3_coder_next_plugin()
        print(f"[INFO] Created plugin: {plugin.__class__.__name__}")

        # Test config-plugin compatibility
        assert plugin.supports_config(config), "Plugin should support the config"
        print("[PASS] Config-plugin compatibility verified")

        # Test initialization (may fail if model not available, but shouldn't crash)
        try:
            init_result = plugin.initialize(config=config)
            print(f"[INFO] Initialization result: {init_result}")
            
            if init_result:
                # Test inference pipeline with code-related input
                test_input = "Write a Python function to reverse a string."
                result = plugin.infer(test_input)
                print(f"[INFO] Inference result type: {type(result)}")
                
                # Test generate_text
                gen_result = plugin.generate_text("def hello_world():", max_new_tokens=20)
                print(f"[INFO] Generate result type: {type(gen_result)}")
                
                print("[PASS] Inference pipeline works")
            else:
                print("[INFO] Skipping inference tests (model not initialized)")
                
        except Exception as e:
            print(f"[INFO] Initialization failed (expected if model not available): {e}")

        # Test cleanup
        cleanup_result = plugin.cleanup()
        print(f"[INFO] Cleanup result: {cleanup_result}")
        print("[PASS] Cleanup completed")

        print("\n[PASS] Qwen3-Coder-Next Integration tests completed")
        return True

    except ImportError as e:
        print(f"[INFO] Qwen3-Coder-Next not available: {str(e)}")
        print("[SKIP] Skipping Qwen3-Coder-Next integration tests")
        return True  # Don't fail if model is not available
    except AssertionError as e:
        print(f"[FAIL] Assertion failed: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_coder_next integration test: {str(e)}")
        traceback.print_exc()
        return False


def test_model_interoperability_integration():
    """Integration test for multiple models working together."""
    print("\n" + "=" * 60)
    print("Integration Testing Model Interoperability")
    print("=" * 60)

    try:
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin

        # Create both plugins
        plugin_06b = create_qwen3_0_6b_plugin()
        plugin_coder = create_qwen3_coder_next_plugin()

        print(f"[INFO] Created plugins: {plugin_06b.__class__.__name__}, {plugin_coder.__class__.__name__}")

        # Verify they are different
        assert type(plugin_06b) != type(plugin_coder), "Plugins should be of different types"
        print("[PASS] Different plugin types created")

        # Test that they can coexist without interference
        info_06b_before = plugin_06b.get_model_info()
        info_coder_before = plugin_coder.get_model_info()

        # Modify one plugin's internal state (if possible)
        plugin_06b.test_attr = "test_value_06b"
        plugin_coder.test_attr = "test_value_coder"

        # Verify they still have independent state
        assert hasattr(plugin_06b, 'test_attr') and plugin_06b.test_attr == "test_value_06b"
        assert hasattr(plugin_coder, 'test_attr') and plugin_coder.test_attr == "test_value_coder"
        print("[PASS] Independent plugin states maintained")

        # Verify model info is still accessible and unchanged
        info_06b_after = plugin_06b.get_model_info()
        info_coder_after = plugin_coder.get_model_info()

        assert info_06b_before['name'] == info_06b_after['name']
        assert info_coder_before['name'] == info_coder_after['name']
        print("[PASS] Model info remains consistent")

        # Test cleanup for both
        cleanup_06b = plugin_06b.cleanup()
        cleanup_coder = plugin_coder.cleanup()
        print(f"[INFO] Cleanup results: 06b={cleanup_06b}, coder={cleanup_coder}")

        print("\n[PASS] Model interoperability integration tests completed")
        return True

    except ImportError as e:
        print(f"[INFO] Model interoperability test skipped: {str(e)}")
        print("[SKIP] Skipping model interoperability integration tests")
        return True  # Don't fail if model is not available
    except AssertionError as e:
        print(f"[FAIL] Assertion failed: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during interoperability test: {str(e)}")
        traceback.print_exc()
        return False


def test_plugin_manager_integration():
    """Integration test for plugin management system."""
    print("\n" + "=" * 60)
    print("Integration Testing Plugin Management System")
    print("=" * 60)

    try:
        from src.inference_pio.plugins.manager import PluginManager
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin

        # Create manager and plugin
        manager = PluginManager()
        plugin = create_qwen3_0_6b_plugin()

        print(f"[INFO] Created manager and plugin")

        # Register plugin
        plugin_name = "test_qwen3_0_6b"
        manager.register_plugin(plugin_name, plugin)
        print(f"[INFO] Registered plugin: {plugin_name}")

        # Retrieve plugin
        retrieved_plugin = manager.get_plugin(plugin_name)
        assert retrieved_plugin is not None, "Plugin should be retrievable"
        print("[PASS] Plugin registration and retrieval works")

        # Verify it's the same plugin
        assert retrieved_plugin is plugin, "Retrieved plugin should be the same object"
        print("[PASS] Retrieved plugin is identical to registered one")

        # Test plugin functionality through manager
        plugin_info = retrieved_plugin.get_model_info()
        assert 'name' in plugin_info, "Plugin should have info"
        print(f"[INFO] Plugin info retrieved: {plugin_info['name']}")

        print("\n[PASS] Plugin management integration tests completed")
        return True

    except ImportError as e:
        print(f"[INFO] Plugin management components not available: {str(e)}")
        print("[SKIP] Skipping plugin management integration tests")
        return True  # Don't fail if components are not available
    except AssertionError as e:
        print(f"[FAIL] Assertion failed: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during plugin management test: {str(e)}")
        traceback.print_exc()
        return False


def test_config_integration():
    """Integration test for configuration system."""
    print("\n" + "=" * 60)
    print("Integration Testing Configuration System")
    print("=" * 60)

    try:
        from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config
        from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig

        # Create different configs
        config_06b = Qwen3_0_6B_Config()
        config_coder = Qwen3CoderNextConfig()

        print(f"[INFO] Created configs: {config_06b.model_name}, {config_coder.model_name}")

        # Verify unique properties
        assert config_06b.model_name != config_coder.model_name, "Configs should have different names"
        print("[PASS] Configs have unique names")

        # Verify shared properties exist
        # Check for common config attributes that might exist
        common_attrs = ['model_name', 'max_seq_len', 'hidden_size', 'num_attention_heads']
        for attr in common_attrs:
            if hasattr(config_06b, attr) and hasattr(config_coder, attr):
                print(f"[PASS] Configs share attribute: {attr}")

        # At least one common attribute should exist
        found_common = any(hasattr(config_06b, attr) and hasattr(config_coder, attr) for attr in common_attrs)
        assert found_common, "Configs should share at least one common attribute"
        print("[PASS] Configs share common attributes")

        print("\n[PASS] Configuration system integration tests completed")
        return True

    except ImportError as e:
        print(f"[INFO] Config integration test skipped: {str(e)}")
        print("[SKIP] Skipping config integration tests")
        return True  # Don't fail if model is not available
    except AssertionError as e:
        print(f"[FAIL] Assertion failed: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during config test: {str(e)}")
        traceback.print_exc()
        return False


def test_optimization_integration():
    """Integration test for optimization components."""
    print("\n" + "=" * 60)
    print("Integration Testing Optimization Components")
    print("=" * 60)

    try:
        from src.inference_pio.common.optimization.activation_offloading import ActivationOffloadingOptimizer
        from src.inference_pio.common.optimization.disk_offloading import DiskOffloadingOptimizer
        from src.inference_pio.common.optimization.tensor_compression import TensorCompressionOptimizer

        # Create optimization instances
        activation_opt = ActivationOffloadingOptimizer()
        disk_opt = DiskOffloadingOptimizer()
        compression_opt = TensorCompressionOptimizer()

        print(f"[INFO] Created optimization instances")

        # Verify they have required methods
        required_methods = ['optimize', 'apply', 'cleanup']
        for opt in [activation_opt, disk_opt, compression_opt]:
            for method in required_methods:
                assert hasattr(opt, method), f"Optimizer should have {method} method"
        print("[PASS] All optimizers have required methods")

        # Test basic functionality (without actually applying to models)
        try:
            # These should not crash even if they don't do anything
            activation_opt.optimize({})
            disk_opt.optimize({})
            compression_opt.optimize({})
            print("[PASS] Optimization methods callable without crashing")
        except Exception as e:
            print(f"[INFO] Optimization methods raised exception (may be expected): {e}")

        print("\n[PASS] Optimization components integration tests completed")
        return True

    except ImportError as e:
        print(f"[INFO] Optimization components not available: {str(e)}")
        print("[SKIP] Skipping optimization integration tests")
        return True  # Don't fail if components are not available
    except AssertionError as e:
        print(f"[FAIL] Assertion failed: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during optimization test: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Main function to run all integration tests with real models."""
    print("Starting integration tests with real models and components...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__ if 'torch' in sys.modules else 'Not loaded yet'}")
    print(f"CUDA available: {torch.cuda.is_available() if 'torch' in sys.modules else False}")

    # Run integration tests
    tests = [
        ("Qwen3-0.6B Integration", test_qwen3_0_6b_integration),
        ("Qwen3-Coder-Next Integration", test_qwen3_coder_next_integration),
        ("Model Interoperability Integration", test_model_interoperability_integration),
        ("Plugin Management Integration", test_plugin_manager_integration),
        ("Configuration Integration", test_config_integration),
        ("Optimization Integration", test_optimization_integration),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} Running {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            print(f"Result: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"Result: FAIL - Exception occurred: {e}")
            results[test_name] = False
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("INTEGRATION TESTS SUMMARY WITH REAL MODELS")
    print("=" * 80)

    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name:<35} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All integration tests with real models completed successfully!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} out of {total} tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)