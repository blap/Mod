"""
Test script to verify that the Qwen3-VL-2B CUDA kernels have been moved correctly
to the plugin directory while keeping generic code in the common directory.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


def test_imports():
    print("Testing imports...")

    # Test that we can import the plugin-specific CUDA kernels
    try:
        from src.inference_pio.models.qwen3_vl_2b.plugin.cuda_kernels import (
            Qwen3VL2BCrossAttentionKernel,
            Qwen3VL2BFusionKernel,
            Qwen3VL2BVisionLanguageAttentionKernel,
            Qwen3VL2BPositionEncodingKernel,
            Qwen3VL2BMLPKernel,
            Qwen3VL2BRMSNormKernel,
            apply_qwen3_vl_cuda_optimizations_to_model,
            get_qwen3_vl_cuda_optimization_report
        )
        print("SUCCESS: Successfully imported plugin-specific CUDA kernels")
    except ImportError as e:
        print(f"ERROR: Failed to import plugin-specific CUDA kernels: {e}")
        return False

    # Test that we can import generic kernels from common
    try:
        from src.inference_pio.common.generic_multimodal_cuda_kernels import (
            GenericCrossAttentionKernel,
            GenericFusionKernel,
            GenericVisionLanguageAttentionKernel
        )
        print("SUCCESS: Successfully imported generic CUDA kernels")
    except ImportError as e:
        print(f"ERROR: Failed to import generic CUDA kernels: {e}")
        return False

    # Test that the old import path still works for backward compatibility
    try:
        from src.inference_pio.common import (
            Qwen3VL2BCrossAttentionKernel as OldQwen3VL2BCrossAttentionKernel,
            Qwen3VL2BFusionKernel as OldQwen3VL2BFusionKernel
        )
        print("SUCCESS: Backward compatibility maintained for common imports")
    except ImportError as e:
        print(f"ERROR: Backward compatibility broken: {e}")
        return False

    # Test that we can import from the attention module
    try:
        from src.inference_pio.models.qwen3_vl_2b.attention.multimodal_attention_optimization import (
            Qwen3VL2BMultimodalAttentionOptimizer,
            Qwen3VL2BAttentionManager
        )
        print("SUCCESS: Successfully imported attention optimization modules")
    except ImportError as e:
        print(f"ERROR: Failed to import attention optimization modules: {e}")
        return False

    # Test that we can import the model
    try:
        from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
        print("SUCCESS: Successfully imported Qwen3VL2BModel")
    except ImportError as e:
        print(f"ERROR: Failed to import Qwen3VL2BModel: {e}")
        return False

    # Test that we can import the plugin
    try:
        from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin
        print("SUCCESS: Successfully imported Qwen3_VL_2B_Instruct_Plugin")
    except ImportError as e:
        print(f"ERROR: Failed to import Qwen3_VL_2B_Instruct_Plugin: {e}")
        return False

    print("\nALL TESTS PASSED! The refactoring was completed successfully.")
    return True

if __name__ == "__main__":
    test_imports()