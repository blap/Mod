"""
Simple Test for Vision Encoder Optimization in Qwen3-VL-2B Model

This module provides a simple test to verify that the vision encoder optimization
system is properly integrated with the Qwen3-VL-2B model.
"""

import torch
import torch.nn as nn

from src.common.vision_transformer_kernels import (
    Qwen3VL2BVisionEncoderKernel,
    VisionTransformerConfig,
)
from src.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.models.qwen3_vl_2b.vision_transformer import (
    OptimizedVisionMLPKernel,
    OptimizedVisionPatchEmbeddingKernel,
    OptimizedVisionSelfAttentionKernel,
    VisionEncoderOptimizationConfig,
    VisionEncoderOptimizer,
)


def test_vision_encoder_optimization():
    """Simple test to verify vision encoder optimization functionality."""
    print("Testing Vision Encoder Optimization System...")

    # Create vision transformer config
    vision_config = VisionTransformerConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_hidden_layers=2,
        patch_size=14,
        image_size=224,
        intermediate_size=1024,
        layer_norm_eps=1e-6,
        use_flash_attention=True,
        use_cuda_kernels=True,
    )

    # Create a basic vision encoder
    print("Creating basic vision encoder...")
    basic_vision_encoder = Qwen3VL2BVisionEncoderKernel(vision_config)

    # Create optimization config
    print("Creating optimization config...")
    opt_config = VisionEncoderOptimizationConfig(
        enable_patch_embedding_optimization=True,
        enable_attention_optimization=True,
        enable_mlp_optimization=True,
        enable_block_optimization=True,
        use_flash_attention=True,
        use_convolution_fusion=True,
        enable_gradient_checkpointing=False,
        enable_memory_efficient_attention=True,
        enable_tensor_fusion=True,
        enable_quantization=False,
        enable_lora_adaptation=False,
    )

    # Create optimizer
    print("Creating vision encoder optimizer...")
    optimizer = VisionEncoderOptimizer(opt_config)

    # Create model config
    model_config = Qwen3VL2BConfig()
    model_config.hidden_size = 256
    model_config.num_attention_heads = 8
    model_config.num_hidden_layers = 2
    model_config.vision_patch_size = 14
    model_config.vision_image_size = 224
    model_config.vision_intermediate_size = 1024
    model_config.vision_layer_norm_eps = 1e-6

    # Optimize the vision encoder
    print("Optimizing vision encoder...")
    optimized_vision_encoder = optimizer.optimize_vision_encoder(
        basic_vision_encoder, model_config
    )

    # Test forward pass with sample data
    print("Testing forward pass...")
    sample_images = torch.randn(1, 3, 224, 224)  # Batch of 1 RGB image 224x224

    # Test basic encoder
    basic_output, _ = basic_vision_encoder(sample_images, output_hidden_states=False)
    print(f"Basic encoder output shape: {basic_output.shape}")

    # Test optimized encoder
    optimized_output, _ = optimized_vision_encoder(
        sample_images, output_hidden_states=False
    )
    print(f"Optimized encoder output shape: {optimized_output.shape}")

    # Verify shapes are the same
    assert (
        basic_output.shape == optimized_output.shape
    ), "Output shapes should be the same"
    print("[SUCCESS] Output shapes match")

    # Verify outputs are reasonable (not all zeros)
    assert not torch.allclose(
        basic_output, torch.zeros_like(basic_output)
    ), "Basic output should not be all zeros"
    assert not torch.allclose(
        optimized_output, torch.zeros_like(optimized_output)
    ), "Optimized output should not be all zeros"
    print("[SUCCESS] Outputs are not all zeros")

    # Check that patch embedding was optimized if enabled
    if opt_config.enable_patch_embedding_optimization:
        assert isinstance(
            optimized_vision_encoder.patch_embedding,
            OptimizedVisionPatchEmbeddingKernel,
        ), "Patch embedding should be optimized"
        print("[SUCCESS] Patch embedding optimization applied")

    print("\n[SUCCESS] Vision Encoder Optimization Test Passed!")


def test_vision_encoder_optimization_config():
    """Test the vision encoder optimization configuration."""
    print("\nTesting Vision Encoder Optimization Configuration...")

    # Test default config
    default_config = VisionEncoderOptimizationConfig()
    print(
        f"Default patch embedding optimization: {default_config.enable_patch_embedding_optimization}"
    )
    print(
        f"Default attention optimization: {default_config.enable_attention_optimization}"
    )
    print(f"Default MLP optimization: {default_config.enable_mlp_optimization}")
    print(f"Default quantization: {default_config.enable_quantization}")

    # Test custom config
    custom_config = VisionEncoderOptimizationConfig(
        enable_patch_embedding_optimization=False,
        enable_quantization=True,
        quantization_bits=4,
        enable_lora_adaptation=True,
        lora_rank=32,
    )
    print(
        f"Custom patch embedding optimization: {custom_config.enable_patch_embedding_optimization}"
    )
    print(f"Custom quantization: {custom_config.enable_quantization}")
    print(f"Custom quantization bits: {custom_config.quantization_bits}")
    print(f"Custom LoRA adaptation: {custom_config.enable_lora_adaptation}")
    print(f"Custom LoRA rank: {custom_config.lora_rank}")

    assert not custom_config.enable_patch_embedding_optimization
    assert custom_config.enable_quantization
    assert custom_config.quantization_bits == 4
    assert custom_config.enable_lora_adaptation
    assert custom_config.lora_rank == 32
    print("[SUCCESS] Configuration test passed")


def test_optimized_components():
    """Test individual optimized components."""
    print("\nTesting Individual Optimized Components...")

    vision_config = VisionTransformerConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=1,
        patch_size=14,
        image_size=224,
        intermediate_size=512,
        layer_norm_eps=1e-6,
        use_flash_attention=False,  # Disable for CPU testing
        use_cuda_kernels=False,
    )

    # Test optimized patch embedding
    patch_embed = OptimizedVisionPatchEmbeddingKernel(vision_config)
    sample_images = torch.randn(2, 3, 224, 224)
    patch_output = patch_embed(sample_images)
    print(f"Patch embedding output shape: {patch_output.shape}")
    assert patch_output.shape == (2, (224 // 14) ** 2, 128)
    print("[SUCCESS] Patch embedding test passed")

    # Test optimized attention
    attention = OptimizedVisionSelfAttentionKernel(vision_config)
    hidden_states = torch.randn(2, (224 // 14) ** 2, 128)
    attention_output = attention(hidden_states)
    print(f"Attention output shape: {attention_output.shape}")
    assert attention_output.shape == (2, (224 // 14) ** 2, 128)
    print("[SUCCESS] Attention test passed")

    # Test optimized MLP
    mlp = OptimizedVisionMLPKernel(vision_config)
    mlp_output = mlp(hidden_states)
    print(f"MLP output shape: {mlp_output.shape}")
    assert mlp_output.shape == (2, (224 // 14) ** 2, 128)
    print("[SUCCESS] MLP test passed")


if __name__ == "__main__":
    print("Running Simple Vision Encoder Optimization Tests...\n")

    test_vision_encoder_optimization_config()
    test_optimized_components()
    test_vision_encoder_optimization()

    print(
        "\n[ALL TESTS PASSED] All Simple Vision Encoder Optimization Tests Completed Successfully!"
    )
