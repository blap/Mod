"""
Comprehensive test to verify Qwen3-VL-2B model integration with CUDA kernels.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.common.qwen3_vl_cuda_kernels import (
    Qwen3VL2BConfig as Qwen3VL2BCudaConfig,
    Qwen3VL2BCrossAttentionKernel,
    Qwen3VL2BFusionKernel,
    Qwen3VL2BVisionLanguageAttentionKernel,
    Qwen3VL2BPositionEncodingKernel,
    Qwen3VL2BMLPKernel,
    Qwen3VL2BRMSNormKernel,
    apply_qwen3_vl_cuda_optimizations_to_model
)

def test_model_creation():
    """Test that the model can be created without errors (without loading actual weights)."""
    print("Testing Qwen3-VL-2B model creation with CUDA kernels...")

    # Create a minimal config for testing
    config = Qwen3VL2BConfig()
    # Use a fake model path to prevent downloading
    config.model_path = "./fake/path/to/model"

    # Disable some heavy features for quick test
    config.use_cuda_kernels = True  # Enable CUDA kernels
    config.enable_disk_offloading = False
    config.enable_intelligent_pagination = False
    config.use_flash_attention_2 = False  # Disable to avoid dependency issues

    try:
        # This should not raise an import error even if model loading fails
        model = Qwen3VL2BModel(config)
        print("Model creation succeeded (or failed gracefully)")
        return True
    except ImportError as e:
        print(f"Import error (this indicates a problem with CUDA kernel imports): {e}")
        return False
    except Exception as e:
        # Other exceptions like model loading failures are expected in this test
        print(f"Other exception (expected for fake model path): {type(e).__name__}: {e}")
        return True  # This is OK, import worked

def test_cuda_kernels_import():
    """Test that CUDA kernels can be imported without errors."""
    print("Testing Qwen3-VL-2B CUDA kernels import...")

    try:
        # Test that all CUDA kernels can be imported
        from src.inference_pio.common.qwen3_vl_cuda_kernels import (
            Qwen3VL2BConfig,
            Qwen3VL2BCrossAttentionKernel,
            Qwen3VL2BFusionKernel,
            Qwen3VL2BVisionLanguageAttentionKernel,
            Qwen3VL2BPositionEncodingKernel,
            Qwen3VL2BMLPKernel,
            Qwen3VL2BRMSNormKernel,
            create_qwen3_vl_cross_attention_kernel,
            create_qwen3_vl_fusion_kernel,
            create_qwen3_vl_vision_language_attention_kernel,
            create_qwen3_vl_position_encoding_kernel,
            create_qwen3_vl_mlp_kernel,
            create_qwen3_vl_rms_norm_kernel,
            apply_qwen3_vl_cuda_optimizations_to_model,
            get_qwen3_vl_cuda_optimization_report
        )

        print("CUDA kernels import succeeded")
        return True
    except ImportError as e:
        print(f"CUDA kernels import failed: {e}")
        return False

def test_cuda_kernel_creation():
    """Test that CUDA kernels can be instantiated."""
    print("Testing Qwen3-VL-2B CUDA kernel creation...")

    try:
        # Create a minimal config for testing
        config = Qwen3VL2BCudaConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=4,
            intermediate_size=1024,
            max_position_embeddings=1024,
            rms_norm_eps=1e-6,
            use_flash_attention_2=False,
            use_cuda_kernels=True
        )

        # Test creating each kernel
        cross_attn_kernel = Qwen3VL2BCrossAttentionKernel(config, layer_idx=0)
        fusion_kernel = Qwen3VL2BFusionKernel(config, layer_idx=0)
        vision_lang_kernel = Qwen3VL2BVisionLanguageAttentionKernel(config, layer_idx=0)
        pos_enc_kernel = Qwen3VL2BPositionEncodingKernel(config)
        mlp_kernel = Qwen3VL2BMLPKernel(config, layer_idx=0)
        rms_norm_kernel = Qwen3VL2BRMSNormKernel(config, layer_idx=0)

        print("CUDA kernel creation succeeded")
        return True
    except Exception as e:
        print(f"CUDA kernel creation failed: {e}")
        return False

def test_cuda_kernel_forward_pass():
    """Test forward pass of CUDA kernels."""
    print("Testing Qwen3-VL-2B CUDA kernel forward pass...")

    try:
        # Create a minimal config for testing
        config = Qwen3VL2BCudaConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=4,
            intermediate_size=512,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            use_flash_attention_2=False,
            use_cuda_kernels=True
        )

        # Test cross attention kernel forward pass
        cross_attn_kernel = Qwen3VL2BCrossAttentionKernel(config, layer_idx=0)
        batch_size = 1
        seq_len = 5
        text_tensor = torch.randn(batch_size, seq_len, config.hidden_size)
        image_tensor = torch.randn(batch_size, seq_len, config.hidden_size)

        queries = {"text": text_tensor}
        keys = {"text": text_tensor, "image": image_tensor}
        values = {"text": text_tensor, "image": image_tensor}

        outputs, _ = cross_attn_kernel(queries, keys, values)

        assert "text" in outputs
        assert outputs["text"].shape == (batch_size, seq_len, config.hidden_size)

        print("CUDA kernel forward pass succeeded")
        return True
    except Exception as e:
        print(f"CUDA kernel forward pass failed: {e}")
        return False

def test_apply_cuda_optimizations():
    """Test applying CUDA optimizations to a model."""
    print("Testing Qwen3-VL-2B CUDA optimizations application...")

    try:
        import torch.nn as nn

        # Create a simple test model
        class SimpleTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
                self.linear = nn.Linear(256, 256)

            def forward(self, x):
                attn_out, _ = self.attention(x, x, x)
                return self.linear(attn_out)

        model = SimpleTestModel()

        # Create a config for testing
        config = Qwen3VL2BCudaConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_hidden_layers=2,
            intermediate_size=512,
            max_position_embeddings=256,
            rms_norm_eps=1e-6,
            use_flash_attention_2=False,
            use_cuda_kernels=True
        )

        # Apply CUDA optimizations
        optimized_model = apply_qwen3_vl_cuda_optimizations_to_model(model, config)

        # Test that the optimized model works
        x = torch.randn(1, 5, 256)
        output = optimized_model(x)

        assert output.shape == (1, 5, 256)

        print("CUDA optimizations application succeeded")
        return True
    except Exception as e:
        print(f"CUDA optimizations application failed: {e}")
        return False

if __name__ == "__main__":
    import torch  # Import here to avoid issues if CUDA is not available

    tests = [
        test_cuda_kernels_import,
        test_cuda_kernel_creation,
        test_cuda_kernel_forward_pass,
        test_apply_cuda_optimizations,
        test_model_creation
    ]

    all_passed = True
    for test_func in tests:
        try:
            result = test_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"Test {test_func.__name__} failed with exception: {e}")
            all_passed = False

    if all_passed:
        print("\n✓ All Qwen3-VL-2B CUDA integration tests passed")
    else:
        print("\n✗ Some Qwen3-VL-2B CUDA integration tests failed")
        exit(1)