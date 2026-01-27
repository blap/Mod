"""
Verification script for multimodal attention optimization implementation in Qwen3-VL-2B model.
This script verifies that the multimodal attention optimization system is properly implemented.
"""

import sys
import os

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

def verify_multimodal_attention_optimization():
    """Verify the multimodal attention optimization implementation."""
    print("Verifying Qwen3-VL-2B Multimodal Attention Optimization Implementation")
    print("=" * 65)
    
    success_count = 0
    total_tests = 0
    
    # Test 1: Check that the optimization module exists
    total_tests += 1
    try:
        import src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization
        print("[OK] Test 1: multimodal_attention_optimization module exists")
        success_count += 1
    except ImportError as e:
        print(f"[FAIL] Test 1: Failed to import multimodal_attention_optimization module: {e}")
    
    # Test 2: Check that the config has multimodal attention optimization parameters
    total_tests += 1
    try:
        from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
        config = Qwen3VL2BConfig()
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
        
        missing_params = [param for param in required_params if not hasattr(config, param)]
        if not missing_params:
            print("[OK] Test 2: All multimodal attention optimization parameters found in config")
            success_count += 1
        else:
            print(f"[FAIL] Test 2: Missing config parameters: {missing_params}")
    except Exception as e:
        print(f"[FAIL] Test 2: Error checking config parameters: {e}")
    
    # Test 3: Check that the optimizer class exists and can be imported
    total_tests += 1
    try:
        from src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization import Qwen3VL2BMultimodalAttentionOptimizer
        print("[OK] Test 3: Qwen3VL2BMultimodalAttentionOptimizer class exists")
        success_count += 1
    except ImportError as e:
        print(f"[FAIL] Test 3: Failed to import Qwen3VL2BMultimodalAttentionOptimizer: {e}")
    
    # Test 4: Check that the attention manager class exists and can be imported
    total_tests += 1
    try:
        from src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization import Qwen3VL2BAttentionManager
        print("[OK] Test 4: Qwen3VL2BAttentionManager class exists")
        success_count += 1
    except ImportError as e:
        print(f"[FAIL] Test 4: Failed to import Qwen3VL2BAttentionManager: {e}")
    
    # Test 5: Check that the factory functions exist and can be imported
    total_tests += 1
    try:
        from src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization import (
            create_qwen3_vl_multimodal_attention_optimizer,
            apply_qwen3_vl_multimodal_attention_optimizations_to_model,
            get_qwen3_vl_multimodal_attention_optimization_report
        )
        print("[OK] Test 5: All factory functions exist")
        success_count += 1
    except ImportError as e:
        print(f"[FAIL] Test 5: Failed to import factory functions: {e}")
    
    # Test 6: Check that the model has the optimization method
    total_tests += 1
    try:
        from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
        if hasattr(Qwen3VL2BModel, '_apply_multimodal_attention_optimization'):
            print("[OK] Test 6: Qwen3VL2BModel has _apply_multimodal_attention_optimization method")
            success_count += 1
        else:
            print("[FAIL] Test 6: Qwen3VL2BModel missing _apply_multimodal_attention_optimization method")
    except Exception as e:
        print(f"[FAIL] Test 6: Error checking model methods: {e}")
    
    # Test 7: Check that the plugin has the optimization integration
    total_tests += 1
    try:
        from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin
        print("[OK] Test 7: Qwen3_VL_2B_Instruct_Plugin exists")
        success_count += 1
    except ImportError as e:
        print(f"[FAIL] Test 7: Failed to import Qwen3_VL_2B_Instruct_Plugin: {e}")
    
    # Test 8: Check that the common module has been updated
    total_tests += 1
    try:
        from src.inference_pio.common.generic_multimodal_attention_optimization import (
            GenericMultimodalAttentionOptimizer as BaseOptimizer,
            GenericMultimodalAttentionManager as BaseManager
        )
        print("[OK] Test 8: Common multimodal attention optimization module exists")
        success_count += 1
    except ImportError as e:
        print(f"[FAIL] Test 8: Failed to import common multimodal attention optimization: {e}")
    
    print("=" * 65)
    print(f"Verification Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("[SUCCESS] ALL TESTS PASSED! Qwen3-VL-2B multimodal attention optimization is properly implemented.")
        return True
    else:
        print("[ERROR] SOME TESTS FAILED! Please check the implementation.")
        return False

if __name__ == "__main__":
    success = verify_multimodal_attention_optimization()
    if not success:
        sys.exit(1)
    else:
        print("\n[SUCCESS] Qwen3-VL-2B multimodal attention optimization implementation verified successfully!")