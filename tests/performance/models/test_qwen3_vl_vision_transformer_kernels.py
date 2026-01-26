"""
Test suite for Vision Transformer Kernels for Qwen3-VL-2B Model.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from src.inference_pio.common.vision_transformer_kernels import (
    VisionTransformerConfig,
    VisionPatchEmbeddingKernel,
    VisionSelfAttentionKernel,
    VisionMLPKernel,
    VisionTransformerBlockKernel,
    VisionConvolutionKernel,
    Qwen3VL2BVisionEncoderKernel,
    create_vision_patch_embedding_kernel,
    create_vision_self_attention_kernel,
    create_vision_mlp_kernel,
    create_vision_transformer_block_kernel,
    create_qwen3_vl_2b_vision_encoder_kernel,
    apply_vision_cuda_optimizations_to_model
)

# TestVisionTransformerKernels

    """Test suite for Vision Transformer kernels."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = VisionTransformerConfig(
            hidden_size=768,
            num_attention_heads=12,
            num_hidden_layers=4,
            patch_size=14,
            image_size=224,
            intermediate_size=2048,
            layer_norm_eps=1e-6,
            use_flash_attention=True,
            use_cuda_kernels=True
        )

    def vision_transformer_config_creation(self)():
        """Test VisionTransformerConfig creation."""
        config = VisionTransformerConfig()
        assert_is_instance(config, VisionTransformerConfig)
        assert_equal(config.hidden_size, 1024)
        assert_equal(config.patch_size, 14)

    def vision_patch_embedding_kernel_creation(self)():
        """Test VisionPatchEmbeddingKernel creation."""
        kernel = VisionPatchEmbeddingKernel(config)
        assert_is_instance(kernel, VisionPatchEmbeddingKernel)
        assert_equal(kernel.hidden_size, config.hidden_size)
        assert_equal(kernel.patch_size, config.patch_size)

    def vision_patch_embedding_kernel_forward(self)():
        """Test VisionPatchEmbeddingKernel forward pass."""
        kernel = VisionPatchEmbeddingKernel(config)

        # Create sample image input
        batch_size = 2
        channels = 3  # RGB
        height = config.image_size
        width = config.image_size
        
        pixel_values = torch.randn(batch_size, channels, height, width)

        output = kernel(pixel_values)

        # Calculate expected number of patches
        expected_num_patches = (height // config.patch_size) * (width // config.patch_size)
        assert_equal(output.shape, (batch_size))

    def vision_self_attention_kernel_creation(self)():
        """Test VisionSelfAttentionKernel creation."""
        kernel = VisionSelfAttentionKernel(config)
        assert_is_instance(kernel, VisionSelfAttentionKernel)
        assert_equal(kernel.hidden_size, config.hidden_size)
        assert_equal(kernel.num_attention_heads, config.num_attention_heads)

    def vision_self_attention_kernel_forward(self)():
        """Test VisionSelfAttentionKernel forward pass."""
        kernel = VisionSelfAttentionKernel(config)

        # Create sample input
        batch_size = 2
        seq_len = 25  # 25 patches from 224x224 image with patch size 14 (224/14 = 16, 16*16 = 256, but using 25 for test)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = kernel(hidden_states)

        assert_equal(output.shape, (batch_size))

    def vision_mlp_kernel_creation(self)():
        """Test VisionMLPKernel creation."""
        kernel = VisionMLPKernel(config)
        assert_is_instance(kernel, VisionMLPKernel)
        assert_equal(kernel.fc1.out_features, config.intermediate_size)
        assert_equal(kernel.fc2.in_features, config.intermediate_size)

    def vision_mlp_kernel_forward(self)():
        """Test VisionMLPKernel forward pass."""
        kernel = VisionMLPKernel(config)

        # Create sample input
        batch_size = 2
        seq_len = 25
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = kernel(hidden_states)

        assert_equal(output.shape, (batch_size))

    def vision_transformer_block_kernel_creation(self)():
        """Test VisionTransformerBlockKernel creation."""
        kernel = VisionTransformerBlockKernel(config, layer_idx=0)
        assert_is_instance(kernel, VisionTransformerBlockKernel)
        assert_equal(kernel.layer_idx, 0)

    def vision_transformer_block_kernel_forward(self)():
        """Test VisionTransformerBlockKernel forward pass."""
        kernel = VisionTransformerBlockKernel(config, layer_idx=0)

        # Create sample input
        batch_size = 2
        seq_len = 25
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = kernel(hidden_states)

        assert_equal(output.shape, (batch_size))

    def vision_convolution_kernel_creation(self)():
        """Test VisionConvolutionKernel creation."""
        kernel = VisionConvolutionKernel(config)
        assert_is_instance(kernel, VisionConvolutionKernel)

    def vision_convolution_kernel_forward(self)():
        """Test VisionConvolutionKernel forward pass."""
        kernel = VisionConvolutionKernel(config)

        # Create sample input - simulate reshaped patches
        batch_size = 2
        seq_len = 25  # Assuming 5x5 grid
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output = kernel(hidden_states)

        assert_equal(output.shape, (batch_size))

    def qwen3_vl_2b_vision_encoder_kernel_creation(self)():
        """Test Qwen3VL2BVisionEncoderKernel creation."""
        kernel = Qwen3VL2BVisionEncoderKernel(config)
        assert_is_instance(kernel, Qwen3VL2BVisionEncoderKernel)
        assert_equal(len(kernel.blocks), config.num_hidden_layers)

    def qwen3_vl_2b_vision_encoder_kernel_forward(self)():
        """Test Qwen3VL2BVisionEncoderKernel forward pass."""
        kernel = Qwen3VL2BVisionEncoderKernel(config)

        # Create sample image input
        batch_size = 2
        channels = 3  # RGB
        height = config.image_size
        width = config.image_size
        
        pixel_values = torch.randn(batch_size, channels, height, width)

        output, hidden_states = kernel(pixel_values, output_hidden_states=True)

        # Calculate expected number of patches
        expected_num_patches = (height // config.patch_size) * (width // config.patch_size)
        assert_equal(output.shape, (batch_size))
        
        # Check that hidden states are returned when requested
        assert_is_not_none(hidden_states)
        assert_equal(len(hidden_states))

    def create_functions(self)():
        """Test all factory functions."""
        # Test patch embedding kernel creation
        kernel1 = create_vision_patch_embedding_kernel(config)
        assert_is_instance(kernel1, VisionPatchEmbeddingKernel)

        # Test self attention kernel creation
        kernel2 = create_vision_self_attention_kernel(config)
        assert_is_instance(kernel2, VisionSelfAttentionKernel)

        # Test MLP kernel creation
        kernel3 = create_vision_mlp_kernel(config)
        assert_is_instance(kernel3, VisionMLPKernel)

        # Test transformer block kernel creation
        kernel4 = create_vision_transformer_block_kernel(config, 0)
        assert_is_instance(kernel4, VisionTransformerBlockKernel)

        # Test vision encoder kernel creation
        kernel5 = create_qwen3_vl_2b_vision_encoder_kernel(config)
        assert_is_instance(kernel5, Qwen3VL2BVisionEncoderKernel)

    def apply_vision_cuda_optimizations_to_model(self)():
        """Test applying vision CUDA optimizations to a simple model."""
        # Create a simple test model with vision components
        class SimpleVisionTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                patch_embed = nn.Conv2d(3, 768, kernel_size=14, stride=14)
                attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)
                linear = nn.Linear(768, 768)

            def forward(self, x):
                patches = patch_embed(x)
                patches = patches.flatten(2).transpose(1, 2)
                attn_out, _ = attention(patches, patches, patches)
                return linear(attn_out)

        model = SimpleVisionTestModel()

        # Apply vision optimizations
        optimized_model = apply_vision_cuda_optimizations_to_model(model, config)

        # Check that the model is still functional
        x = torch.randn(2, 3, 224, 224)  # Sample image input
        output, _ = optimized_model.patch_embed(x).flatten(2).transpose(1, 2), None
        # Note: Since we only optimize certain layers, the model structure might remain the same
        # in this simple test, but the optimization function should run without errors

    def cuda_availability_handling(self)():
        """Test that vision kernels handle CUDA availability properly."""
        # This test ensures that the vision kernels don't crash when CUDA is not available
        # or when running on CPU

        kernel = VisionPatchEmbeddingKernel(config)

        # Create CPU tensors
        batch_size = 1
        channels = 3
        height = config.image_size
        width = config.image_size
        
        pixel_values = torch.randn(batch_size, channels, height, width)

        output = kernel(pixel_values)

        assert_equal(output.device.type, 'cpu')
        # Calculate expected number of patches
        expected_num_patches = (height // config.patch_size) * (width // config.patch_size)
        assert_equal(output.shape, (batch_size))

if __name__ == '__main__':
    run_tests(test_functions)