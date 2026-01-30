"""
Simple test for multimodal attention optimization in Qwen3-VL-2B model.
This script verifies that the multimodal attention optimization system is properly implemented.
"""

import sys
import os

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

def test_multimodal_attention_optimization_imports():
    """Test that multimodal attention optimization modules can be imported successfully."""
    print("Testing multimodal attention optimization imports...")
    
    try:
        from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
        print("✓ Qwen3VL2BConfig imported successfully")
    except ImportError as e:
        print(f"X Failed to import Qwen3VL2BConfig: {e}")
        return False
    
    try:
        from src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization import (
            Qwen3VL2BMultimodalAttentionOptimizer,
            Qwen3VL2BAttentionManager,
            create_qwen3_vl_multimodal_attention_optimizer,
            apply_qwen3_vl_multimodal_attention_optimizations_to_model,
            get_qwen3_vl_multimodal_attention_optimization_report
        )
        print("✓ Multimodal attention optimization modules imported successfully")
    except ImportError as e:
        print(f"X Failed to import multimodal attention optimization modules: {e}")
        return False
    
    return True


def test_config_has_multimodal_attention_optimization_params():
    """Test that the config has multimodal attention optimization parameters."""
    print("\nTesting config parameters for multimodal attention optimization...")
    
    try:
        from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
        
        config = Qwen3VL2BConfig()
        
        # Check that the config has the required parameters
        required_params = [
            'use_multimodal_attention_optimization',
            'multimodal_attention_sparsity_ratio',
            'multimodal_attention_temperature',
            'multimodal_attention_lambda',
            'multimodal_attention_window_size',
            'multimodal_attention_use_flash',
            'multimodal_attention_use_sparse',
            'multimodal_attention_use_sliding_window',
            'multimodal_attention_use_mqa_gqa',
            'multimodal_attention_use_paged',
            'multimodal_attention_cross_modal_fusion_method',
            'multimodal_attention_cross_modal_alignment_method',
            'multimodal_attention_enable_dynamic_fusion',
            'multimodal_attention_enable_adaptive_compression',
            'multimodal_attention_compression_ratio',
            'multimodal_attention_enable_tensor_fusion',
            'multimodal_attention_tensor_fusion_method',
            'multimodal_attention_enable_quantization',
            'multimodal_attention_quantization_bits',
            'multimodal_attention_enable_lora',
            'multimodal_attention_lora_rank',
            'multimodal_attention_lora_alpha'
        ]
        
        missing_params = []
        for param in required_params:
            if not hasattr(config, param):
                missing_params.append(param)
        
        if missing_params:
            print(f"X Missing config parameters: {missing_params}")
            return False
        else:
            print("OK All required multimodal attention optimization parameters found in config")
            print(f"  - use_multimodal_attention_optimization: {config.use_multimodal_attention_optimization}")
            print(f"  - multimodal_attention_sparsity_ratio: {config.multimodal_attention_sparsity_ratio}")
            print(f"  - multimodal_attention_temperature: {config.multimodal_attention_temperature}")
            print(f"  - multimodal_attention_lambda: {config.multimodal_attention_lambda}")
            print(f"  - multimodal_attention_window_size: {config.multimodal_attention_window_size}")
            print(f"  - multimodal_attention_use_flash: {config.multimodal_attention_use_flash}")
            print(f"  - multimodal_attention_cross_modal_fusion_method: {config.multimodal_attention_cross_modal_fusion_method}")
            return True
            
    except Exception as e:
        print(f"X Error testing config parameters: {e}")
        return False


def test_multimodal_attention_optimizer_creation():
    """Test that multimodal attention optimizer can be created successfully."""
    print("\nTesting multimodal attention optimizer creation...")

    try:
        from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
        from src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization import (
            Qwen3VL2BMultimodalAttentionOptimizer,
            create_qwen3_vl_multimodal_attention_optimizer
        )

        config = Qwen3VL2BConfig()
        config.hidden_size = 1024
        config.num_attention_heads = 8
        config.num_key_value_heads = 2

        # Test direct class instantiation
        optimizer = Qwen3VL2BMultimodalAttentionOptimizer(
            config=config,
            layer_idx=0
        )

        print("OK Qwen3VL2BMultimodalAttentionOptimizer created successfully")

        # Test factory function
        optimizer_factory = create_qwen3_vl_multimodal_attention_optimizer(
            config=config,
            layer_idx=1
        )

        print("OK create_qwen3_vl_multimodal_attention_optimizer works")

        # Check that both are instances of the correct class
        assert isinstance(optimizer, Qwen3VL2BMultimodalAttentionOptimizer)
        assert isinstance(optimizer_factory, Qwen3VL2BMultimodalAttentionOptimizer)

        print("  - Both optimizers are correct type")
        print(f"  - Hidden size: {optimizer.hidden_size}")
        print(f"  - Num attention heads: {optimizer.num_attention_heads}")
        print(f"  - Num KV heads: {optimizer.num_key_value_heads}")

        return True

    except Exception as e:
        print(f"ERROR creating multimodal attention optimizer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_manager_creation():
    """Test that attention manager can be created successfully."""
    print("\nTesting attention manager creation...")
    
    try:
        from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
        from src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization import (
            Qwen3VL2BAttentionManager
        )
        
        config = Qwen3VL2BConfig()
        config.num_hidden_layers = 4  # Use smaller number for test
        
        manager = Qwen3VL2BAttentionManager(config)
        
        print("✓ Qwen3VL2BAttentionManager created successfully")
        print(f"  - Number of optimizers created: {len(manager.attention_optimizers)}")
        
        # Check that each layer has an optimizer
        for layer_idx in range(config.num_hidden_layers):
            optimizer_key = f'layer_{layer_idx}'
            if optimizer_key in manager.attention_optimizers:
                print(f"  - Layer {layer_idx}: ✓ Optimizer exists")
            else:
                print(f"  - Layer {layer_idx}: ✗ Optimizer missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating attention manager: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_integration():
    """Test integration with the Qwen3-VL-2B model."""
    print("\nTesting model integration...")
    
    try:
        from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
        from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
        from src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization import (
            apply_qwen3_vl_multimodal_attention_optimizations_to_model
        )
        
        config = Qwen3VL2BConfig()
        config.hidden_size = 512  # Smaller for test
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.use_multimodal_attention_optimization = True
        
        # Create a simple mock model for testing the integration
        class MockModel:
            def __init__(self):
                self.config = config
                self.transformer = type('Transformer', (), {})()
                self.transformer.layers = []
                for i in range(2):  # Create 2 mock layers
                    layer = type('Layer', (), {})()
                    layer.self_attn = type('Attention', (), {})()
                    self.transformer.layers.append(layer)
        
        mock_model = MockModel()
        
        # Apply multimodal attention optimizations
        optimized_model = apply_qwen3_vl_multimodal_attention_optimizations_to_model(mock_model, config)
        
        print("✓ apply_qwen3_vl_multimodal_attention_optimizations_to_model executed successfully")
        
        # Check that the model was returned
        assert optimized_model is not None
        print("  - Model returned successfully after optimization application")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in model integration test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests for multimodal attention optimization implementation."""
    print("Testing Qwen3-VL-2B Multimodal Attention Optimization Implementation")
    print("=" * 65)
    
    all_tests_passed = True
    
    # Run all tests
    tests = [
        test_multimodal_attention_optimization_imports,
        test_config_has_multimodal_attention_optimization_params,
        test_multimodal_attention_optimizer_creation,
        test_attention_manager_creation,
        test_model_integration
    ]
    
    for test_func in tests:
        if not test_func():
            all_tests_passed = False
    
    print("\n" + "=" * 65)
    if all_tests_passed:
        print("✓ ALL TESTS PASSED! Qwen3-VL-2B multimodal attention optimization is properly implemented.")
    else:
        print("✗ SOME TESTS FAILED! Please check the implementation.")
    
    return all_tests_passed


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)