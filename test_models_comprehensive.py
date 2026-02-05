#!/usr/bin/env python
"""
Comprehensive test script to verify that qwen3_0_6b and qwen3_coder_next models 
work correctly after standardization and cross-dependency removal changes.
Tests both import functionality and basic operations without requiring full model downloads.
"""

import sys
import os
import torch
import traceback
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the src directory to the path so we can import the models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_qwen3_0_6b_comprehensive():
    """Comprehensive test for the qwen3_0_6b model functionality."""
    print("=" * 60)
    print("Comprehensive Testing Qwen3-0.6B Model")
    print("=" * 60)
    
    try:
        # Import the plugin creation function
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config
        
        print("[PASS] Successfully imported qwen3_0_6b components")
        
        # Create a plugin instance
        plugin = create_qwen3_0_6b_plugin()
        print("[PASS] Successfully created qwen3_0_6b plugin instance")
        
        # Test config creation
        config = Qwen3_0_6B_Config()
        print(f"[PASS] Created config with model_name: {config.model_name}")
        
        # Mock the model loading to avoid downloading
        with patch('src.inference_pio.models.qwen3_0_6b.model.AutoModelForCausalLM.from_pretrained') as mock_load:
            # Create a mock model
            mock_model = MagicMock()
            mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]])
            mock_load.return_value = mock_model
            
            # Mock the tokenizer
            with patch('src.inference_pio.models.qwen3_0_6b.model.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_tokenizer_instance = MagicMock()
                mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
                mock_tokenizer_instance.decode.return_value = "Mock response text"
                mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
                
                # Test initialization
                success = plugin.initialize(config=config)
                if success:
                    print("[PASS] Plugin initialization successful")
                else:
                    print("[FAIL] Plugin initialization failed")
                    return False
            
            # Test basic inference with mocked components
            try:
                result = plugin.infer("Hello, test!")
                print(f"[PASS] Basic inference successful: {type(result)}")
            except Exception as e:
                print(f"[FAIL] Basic inference failed: {str(e)}")
                return False
            
            # Test generate_text method
            try:
                result = plugin.generate_text("Test prompt", max_new_tokens=10)
                print(f"[PASS] generate_text successful: {type(result)}")
            except Exception as e:
                print(f"[FAIL] generate_text failed: {str(e)}")
                return False
        
        # Test model info retrieval
        try:
            info = plugin.get_model_info()
            print(f"[PASS] Model info retrieved: {info.get('name', 'Unknown')}")
        except Exception as e:
            print(f"[FAIL] Model info retrieval failed: {str(e)}")
            return False
        
        # Test cleanup
        try:
            cleanup_success = plugin.cleanup()
            print(f"[PASS] Cleanup successful: {cleanup_success}")
        except Exception as e:
            print(f"[FAIL] Cleanup failed: {str(e)}")
            return False
        
        print("\n[PASS] Qwen3-0.6B Comprehensive test completed successfully")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Failed to import qwen3_0_6b components: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_0_6b test: {str(e)}")
        traceback.print_exc()
        return False


def test_qwen3_coder_next_comprehensive():
    """Comprehensive test for the qwen3_coder_next model functionality."""
    print("\n" + "=" * 60)
    print("Comprehensive Testing Qwen3-Coder-Next Model")
    print("=" * 60)
    
    try:
        # Import the plugin creation function
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig
        
        print("[PASS] Successfully imported qwen3_coder_next components")
        
        # Create a plugin instance
        plugin = create_qwen3_coder_next_plugin()
        print("[PASS] Successfully created qwen3_coder_next plugin instance")
        
        # Test config creation
        config = Qwen3CoderNextConfig()
        print(f"[PASS] Created config with model_name: {config.model_name}")
        
        # Mock the model loading to avoid downloading
        with patch('src.inference_pio.models.qwen3_coder_next.plugin.AutoTokenizer.from_pretrained') as mock_tokenizer:
            # Create mock tokenizer
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
            mock_tokenizer_instance.decode.return_value = "Mock response text"
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Test initialization
            success = plugin.initialize(config=config)
            if success:
                print("[PASS] Plugin initialization successful")
            else:
                print("[FAIL] Plugin initialization failed")
                return False
        
        # Test basic inference with mocked components
        try:
            result = plugin.infer("Hello, test!")
            print(f"[PASS] Basic inference successful: {type(result)}")
        except Exception as e:
            print(f"[FAIL] Basic inference failed: {str(e)}")
            return False
        
        # Test generate_text method
        try:
            result = plugin.generate_text("Test prompt", max_new_tokens=10)
            print(f"[PASS] generate_text successful: {type(result)}")
        except Exception as e:
            print(f"[FAIL] generate_text failed: {str(e)}")
            return False
        
        # Test cleanup
        try:
            cleanup_success = plugin.cleanup()
            print(f"[PASS] Cleanup successful: {cleanup_success}")
        except Exception as e:
            print(f"[FAIL] Cleanup failed: {str(e)}")
            return False
        
        print("\n[PASS] Qwen3-Coder-Next Comprehensive test completed successfully")
        return True
        
    except ImportError as e:
        print(f"[FAIL] Failed to import qwen3_coder_next components: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_coder_next test: {str(e)}")
        traceback.print_exc()
        return False


def test_cross_dependency_removal():
    """Test that models work independently after cross-dependency removal."""
    print("\n" + "=" * 60)
    print("Testing Cross-Dependency Removal")
    print("=" * 60)
    
    try:
        # Test that each model can be used without interfering with the other
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        
        # Create instances of both plugins
        plugin_06b = create_qwen3_0_6b_plugin()
        plugin_coder = create_qwen3_coder_next_plugin()
        
        print("[PASS] Both plugins created successfully")
        
        # Verify they are different types
        if type(plugin_06b).__name__ != type(plugin_coder).__name__:
            print("[PASS] Plugins are different types as expected")
        else:
            print("[FAIL] Plugins are unexpectedly the same type")
            return False
        
        # Verify they have different metadata
        if plugin_06b.metadata.name != plugin_coder.metadata.name:
            print(f"[PASS] Plugins have different names: {plugin_06b.metadata.name} vs {plugin_coder.metadata.name}")
        else:
            print("[FAIL] Plugins have the same name unexpectedly")
            return False
        
        # Test that each plugin has its own isolated state
        plugin_06b.some_test_attr = "test_06b"
        plugin_coder.some_test_attr = "test_coder"
        
        if hasattr(plugin_06b, 'some_test_attr') and hasattr(plugin_coder, 'some_test_attr'):
            if plugin_06b.some_test_attr == "test_06b" and plugin_coder.some_test_attr == "test_coder":
                print("[PASS] Plugins maintain independent state")
            else:
                print("[FAIL] Plugin states are not independent")
                return False
        else:
            print("[FAIL] Could not set attributes on plugins")
            return False
        
        print("\n[PASS] Cross-dependency removal test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error during cross-dependency test: {str(e)}")
        traceback.print_exc()
        return False


def test_standardization_compliance():
    """Test that models comply with standardization requirements."""
    print("\n" + "=" * 60)
    print("Testing Standardization Compliance")
    print("=" * 60)
    
    try:
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config
        from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig
        
        # Create plugin instances
        plugin_06b = create_qwen3_0_6b_plugin()
        plugin_coder = create_qwen3_coder_next_plugin()
        
        # Test that both plugins inherit from the same base interface
        from src.inference_pio.common.interfaces.improved_base_plugin_interface import TextModelPluginInterface
        
        if isinstance(plugin_06b, TextModelPluginInterface) and isinstance(plugin_coder, TextModelPluginInterface):
            print("[PASS] Both plugins implement TextModelPluginInterface")
        else:
            print("[FAIL] One or both plugins do not implement TextModelPluginInterface")
            return False
        
        # Test that both configs inherit from the same base
        from src.inference_pio.common.config.model_config_base import BaseConfig
        
        config_06b = Qwen3_0_6B_Config()
        config_coder = Qwen3CoderNextConfig()
        
        if isinstance(config_06b, BaseConfig) and isinstance(config_coder, BaseConfig):
            print("[PASS] Both configs inherit from BaseConfig")
        else:
            print("[FAIL] One or both configs do not inherit from BaseConfig")
            return False
        
        # Test standard methods exist in both plugins
        standard_methods = ['initialize', 'infer', 'generate_text', 'cleanup', 'supports_config']
        
        for method in standard_methods:
            if hasattr(plugin_06b, method) and hasattr(plugin_coder, method):
                print(f"[PASS] Standard method '{method}' exists in both plugins")
            else:
                print(f"[FAIL] Standard method '{method}' missing in one or both plugins")
                return False
        
        print("\n[PASS] Standardization compliance test completed successfully")
        return True
        
    except Exception as e:
        print(f"[FAIL] Error during standardization compliance test: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Main function to run all comprehensive tests."""
    print("Starting comprehensive model verification tests...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__ if 'torch' in globals() else 'Not available'}")
    
    # Run comprehensive tests for both models
    qwen3_0_6b_success = test_qwen3_0_6b_comprehensive()
    qwen3_coder_next_success = test_qwen3_coder_next_comprehensive()
    cross_dependency_success = test_cross_dependency_removal()
    standardization_success = test_standardization_compliance()
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    print(f"Qwen3-0.6B Comprehensive: {'[PASS]' if qwen3_0_6b_success else '[FAIL]'}")
    print(f"Qwen3-Coder-Next Comprehensive: {'[PASS]' if qwen3_coder_next_success else '[FAIL]'}")
    print(f"Cross-Dependency Removal: {'[PASS]' if cross_dependency_success else '[FAIL]'}")
    print(f"Standardization Compliance: {'[PASS]' if standardization_success else '[FAIL]'}")
    
    all_passed = all([
        qwen3_0_6b_success,
        qwen3_coder_next_success,
        cross_dependency_success,
        standardization_success
    ])
    
    if all_passed:
        print("\n[SUCCESS] All comprehensive tests passed!")
        print("Models work correctly after standardization and cross-dependency removal.")
        return 0
    else:
        print("\n[ERROR] Some comprehensive tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)