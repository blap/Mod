"""
Test suite for Qwen3-VL-2B specific CUDA kernels.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from src.inference_pio.common.qwen3_vl_cuda_kernels import (
    Qwen3VL2BConfig,
    Qwen3VL2BCrossAttentionKernel,
    Qwen3VL2BFusionKernel,
    Qwen3VL2BVisionLanguageAttentionKernel,
    Qwen3VL2BPositionEncodingKernel,
    Qwen3VL2BMLPKernel,
    Qwen3VL2BRMSNormKernel,
    Qwen3VL2BVisionProcessingKernel,
    create_qwen3_vl_cross_attention_kernel,
    create_qwen3_vl_fusion_kernel,
    create_qwen3_vl_vision_language_attention_kernel,
    create_qwen3_vl_position_encoding_kernel,
    create_qwen3_vl_mlp_kernel,
    create_qwen3_vl_rms_norm_kernel,
    create_qwen3_vl_vision_processing_kernel,
    apply_qwen3_vl_cuda_optimizations_to_model,
    get_qwen3_vl_cuda_optimization_report
)

# TestQwen3VL2BCudaKernels

    """Test suite for Qwen3-VL-2B specific CUDA kernels."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = Qwen3VL2BConfig(
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=12,
            intermediate_size=2048,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            use_flash_attention_2=True,
            use_cuda_kernels=True
        )

    def qwen3_vl_2b_config_creation(self)():
        """Test Qwen3VL2BConfig creation."""
        config = Qwen3VL2BConfig()
        assert_is_instance(config, Qwen3VL2BConfig)
        assert_equal(config.hidden_size, 2048)
        assert_equal(config.num_attention_heads, 16)

    def qwen3_vl_cross_attention_kernel_creation(self)():
        """Test Qwen3VL2BCrossAttentionKernel creation."""
        kernel = Qwen3VL2BCrossAttentionKernel(config, layer_idx=0)
        assert_is_instance(kernel, Qwen3VL2BCrossAttentionKernel)
        assert_equal(kernel.d_model, config.hidden_size)
        assert_equal(kernel.nhead, config.num_attention_heads)

    def qwen3_vl_cross_attention_kernel_forward(self)():
        """Test Qwen3VL2BCrossAttentionKernel forward pass."""
        kernel = Qwen3VL2BCrossAttentionKernel(config, layer_idx=0)
        
        # Create sample inputs
        batch_size = 2
        seq_len = 10
        text_tensor = torch.randn(batch_size, seq_len, config.hidden_size)
        image_tensor = torch.randn(batch_size, seq_len, config.hidden_size)
        
        queries = {"text": text_tensor}
        keys = {"text": text_tensor, "image": image_tensor}
        values = {"text": text_tensor, "image": image_tensor}
        
        outputs, attention_weights = kernel(queries, keys, values)
        
        assert_in("text", outputs)
        assert_equal(outputs["text"].shape, (batch_size))
        if attention_weights is not None:
            assert_in("text", attention_weights)

    def qwen3_vl_fusion_kernel_creation(self)():
        """Test Qwen3VL2BFusionKernel creation."""
        kernel = Qwen3VL2BFusionKernel(config, layer_idx=0)
        assert_is_instance(kernel, Qwen3VL2BFusionKernel)
        assert_equal(kernel.d_model, config.hidden_size)
        assert_equal(kernel.nhead, config.num_attention_heads)

    def qwen3_vl_fusion_kernel_forward(self)():
        """Test Qwen3VL2BFusionKernel forward pass."""
        kernel = Qwen3VL2BFusionKernel(config, layer_idx=0)
        
        # Create sample inputs
        batch_size = 2
        seq_len = 10
        text_tensor = torch.randn(batch_size, seq_len, config.hidden_size)
        image_tensor = torch.randn(batch_size, seq_len, config.hidden_size)
        
        inputs = {"text": text_tensor, "image": image_tensor}
        
        outputs = kernel(inputs)
        
        assert_in("text", outputs)
        assert_in("image", outputs)
        assert_equal(outputs["text"].shape, (batch_size))
        assert_equal(outputs["image"].shape, (batch_size))

    def qwen3_vl_vision_language_attention_kernel_creation(self)():
        """Test Qwen3VL2BVisionLanguageAttentionKernel creation."""
        kernel = Qwen3VL2BVisionLanguageAttentionKernel(config, layer_idx=0)
        assert_is_instance(kernel, Qwen3VL2BVisionLanguageAttentionKernel)
        assert_equal(kernel.d_model, config.hidden_size)
        assert_equal(kernel.nhead, config.num_attention_heads)

    def qwen3_vl_vision_language_attention_kernel_forward(self)():
        """Test Qwen3VL2BVisionLanguageAttentionKernel forward pass."""
        kernel = Qwen3VL2BVisionLanguageAttentionKernel(config, layer_idx=0)
        
        # Create sample inputs
        batch_size = 2
        num_patches = 10
        seq_len = 15
        vision_features = torch.randn(batch_size, num_patches, config.hidden_size)
        language_features = torch.randn(batch_size, seq_len, config.hidden_size)
        
        fused_output, vision_output, language_output, attention_weights = kernel(
            vision_features, language_features
        )
        
        expected_fused_shape = (batch_size, num_patches + seq_len, config.hidden_size)
        assert_equal(fused_output.shape, expected_fused_shape)
        assert_equal(vision_output.shape, vision_features.shape)
        assert_equal(language_output.shape, language_features.shape)

    def qwen3_vl_position_encoding_kernel_creation(self)():
        """Test Qwen3VL2BPositionEncodingKernel creation."""
        kernel = Qwen3VL2BPositionEncodingKernel(config)
        assert_is_instance(kernel, Qwen3VL2BPositionEncodingKernel)

    def qwen3_vl_position_encoding_kernel_forward(self)():
        """Test Qwen3VL2BPositionEncodingKernel forward pass."""
        kernel = Qwen3VL2BPositionEncodingKernel(config)
        
        # Create sample inputs
        batch_size = 2
        seq_len = 10
        text_tensor = torch.randn(batch_size, seq_len, config.hidden_size)
        image_tensor = torch.randn(batch_size, seq_len, config.hidden_size)
        
        features = {"text": text_tensor, "image": image_tensor}
        
        encoded_features = kernel(features)
        
        assert_in("text", encoded_features)
        assert_in("image", encoded_features)
        assert_equal(encoded_features["text"].shape, (batch_size))
        assert_equal(encoded_features["image"].shape, (batch_size))

    def qwen3_vl_mlp_kernel_creation(self)():
        """Test Qwen3VL2BMLPKernel creation."""
        kernel = Qwen3VL2BMLPKernel(config, layer_idx=0)
        assert_is_instance(kernel, Qwen3VL2BMLPKernel)
        assert_equal(kernel.gate_proj.out_features, config.intermediate_size)

    def qwen3_vl_mlp_kernel_forward(self)():
        """Test Qwen3VL2BMLPKernel forward pass."""
        kernel = Qwen3VL2BMLPKernel(config, layer_idx=0)
        
        # Create sample input
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = kernel(x)
        
        assert_equal(output.shape, (batch_size))

    def qwen3_vl_rms_norm_kernel_creation(self)():
        """Test Qwen3VL2BRMSNormKernel creation."""
        kernel = Qwen3VL2BRMSNormKernel(config, layer_idx=0)
        assert_is_instance(kernel, Qwen3VL2BRMSNormKernel)
        assert_equal(kernel.weight.shape[0], config.hidden_size)

    def qwen3_vl_rms_norm_kernel_forward(self)():
        """Test Qwen3VL2BRMSNormKernel forward pass."""
        kernel = Qwen3VL2BRMSNormKernel(config, layer_idx=0)
        
        # Create sample input
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = kernel(x)
        
        assert_equal(output.shape, x.shape)

    def create_functions(self)():
        """Test all factory functions."""
        # Test cross attention kernel creation
        kernel1 = create_qwen3_vl_cross_attention_kernel(config, 0)
        assert_is_instance(kernel1, Qwen3VL2BCrossAttentionKernel)
        
        # Test fusion kernel creation
        kernel2 = create_qwen3_vl_fusion_kernel(config, 0)
        assert_is_instance(kernel2, Qwen3VL2BFusionKernel)
        
        # Test vision-language attention kernel creation
        kernel3 = create_qwen3_vl_vision_language_attention_kernel(config, 0)
        assert_is_instance(kernel3, Qwen3VL2BVisionLanguageAttentionKernel)
        
        # Test position encoding kernel creation
        kernel4 = create_qwen3_vl_position_encoding_kernel(config)
        assert_is_instance(kernel4, Qwen3VL2BPositionEncodingKernel)
        
        # Test MLP kernel creation
        kernel5 = create_qwen3_vl_mlp_kernel(config, 0)
        assert_is_instance(kernel5, Qwen3VL2BMLPKernel)
        
        # Test RMSNorm kernel creation
        kernel6 = create_qwen3_vl_rms_norm_kernel(config, 0)
        assert_is_instance(kernel6, Qwen3VL2BRMSNormKernel)

    def apply_qwen3_vl_cuda_optimizations_to_model(self)():
        """Test applying CUDA optimizations to a simple model."""
        # Create a simple test model
        class SimpleTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                attention = nn.MultiheadAttention(embed_dim=1024, num_heads=16)
                linear = nn.Linear(1024, 1024)
                
            def forward(self, x):
                attn_out, _ = attention(x, x, x)
                return linear(attn_out)
        
        model = SimpleTestModel()
        
        # Create a config with smaller dimensions for testing
        test_config = Qwen3VL2BConfig(
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=2,
            intermediate_size=2048,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            use_flash_attention_2=False,  # Disable for CPU testing
            use_cuda_kernels=True
        )
        
        # Apply optimizations
        optimized_model = apply_qwen3_vl_cuda_optimizations_to_model(model, test_config)
        
        # Check that the model is still functional
        x = torch.randn(2, 10, 1024)
        output = optimized_model(x)
        assert_equal(output.shape, (2))

    def get_qwen3_vl_cuda_optimization_report(self)():
        """Test getting optimization report."""
        # Create a simple test model
        class SimpleTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                linear = nn.Linear(1024, 1024)
                
            def forward(self, x):
                return linear(x)
        
        model = SimpleTestModel()
        
        report = get_qwen3_vl_cuda_optimization_report(model, config)
        
        assert_in("model_type", report)
        assert_in("optimizations_applied", report)
        assert_in("config", report)
        assert_equal(report["model_type"], "Qwen3-VL-2B")
        assert_true(report["optimizations_applied"]["qwen3_vl_cross_attention"])

    def cuda_availability_handling(self)():
        """Test that kernels handle CUDA availability properly."""
        # This test ensures that the kernels don't crash when CUDA is not available
        # or when running on CPU

        kernel = Qwen3VL2BCrossAttentionKernel(config)

        # Create CPU tensors
        batch_size = 1
        seq_len = 5
        text_tensor = torch.randn(batch_size, seq_len, config.hidden_size)
        image_tensor = torch.randn(batch_size, seq_len, config.hidden_size)

        queries = {"text": text_tensor}
        keys = {"text": text_tensor, "image": image_tensor}
        values = {"text": text_tensor, "image": image_tensor}

        outputs, _ = kernel(queries, keys, values)

        assert_in("text", outputs)
        assert_equal(outputs["text"].device.type, 'cpu')

    def qwen3_vl_vision_processing_kernel_creation(self)():
        """Test Qwen3VL2BVisionProcessingKernel creation."""
        kernel = Qwen3VL2BVisionProcessingKernel(config)
        assert_is_instance(kernel, Qwen3VL2BVisionProcessingKernel)

    def qwen3_vl_vision_processing_kernel_forward(self)():
        """Test Qwen3VL2BVisionProcessingKernel forward pass."""
        kernel = Qwen3VL2BVisionProcessingKernel(config)

        # Create sample image input
        batch_size = 2
        channels = 3  # RGB
        height = config.vision_image_size
        width = config.vision_image_size

        pixel_values = torch.randn(batch_size, channels, height, width)

        output, hidden_states = kernel(pixel_values, output_hidden_states=True)

        # Check output shape
        expected_seq_len = (height // config.vision_patch_size) * (width // config.vision_patch_size)
        assert_equal(output.shape, (batch_size))

        # Check that hidden states are returned when requested
        assert_is_not_none(hidden_states)
        assert_equal(len(hidden_states))

    def create_vision_processing_kernel_function(self)():
        """Test create_qwen3_vl_vision_processing_kernel function."""
        kernel = create_qwen3_vl_vision_processing_kernel(config)
        assert_is_instance(kernel, Qwen3VL2BVisionProcessingKernel)

if __name__ == '__main__':
    run_tests(test_functions)