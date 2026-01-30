"""
Test suite for Qwen3-VL-2B specific CUDA kernels.
"""
import unittest
import torch
import torch.nn as nn
from inference_pio.models.qwen3_vl_2b.cuda_kernels.qwen3_vl_kernels import (
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

class TestQwen3VL2BCudaKernels(unittest.TestCase):
    """Test suite for Qwen3-VL-2B specific CUDA kernels."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = Qwen3VL2BConfig(
            hidden_size=1024,
            num_attention_heads=16,
            num_hidden_layers=12,
            intermediate_size=2048,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            use_flash_attention_2=False, # Disable for testing environment stability
            use_cuda_kernels=True
        )
        # Mock vision attributes if needed (since I modified config usage)
        if not hasattr(self.config, 'vision_hidden_size'):
             self.config.vision_hidden_size = 1024
             self.config.vision_num_attention_heads = 16
             self.config.vision_num_hidden_layers = 12
             self.config.vision_intermediate_size = 2048
             self.config.vision_image_size = 224
             self.config.vision_patch_size = 14
             self.config.vision_num_channels = 3

    def test_qwen3_vl_2b_config_creation(self):
        """Test Qwen3VL2BConfig creation."""
        config = self.config
        # Note: isinstance check might fail if Qwen3VL2BConfig is imported from ..config vs local definition
        # But I updated kernels file to import from ..config, so it should be consistent.
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.num_attention_heads, 16)

    def test_qwen3_vl_cross_attention_kernel_creation(self):
        """Test Qwen3VL2BCrossAttentionKernel creation."""
        kernel = Qwen3VL2BCrossAttentionKernel(self.config, layer_idx=0)
        self.assertIsInstance(kernel, Qwen3VL2BCrossAttentionKernel)
        self.assertEqual(kernel.d_model, self.config.hidden_size)
        self.assertEqual(kernel.nhead, self.config.num_attention_heads)

    def test_qwen3_vl_cross_attention_kernel_forward(self):
        """Test Qwen3VL2BCrossAttentionKernel forward pass."""
        kernel = Qwen3VL2BCrossAttentionKernel(self.config, layer_idx=0)
        
        # Create sample inputs
        batch_size = 2
        seq_len = 10
        text_tensor = torch.randn(batch_size, seq_len, self.config.hidden_size)
        image_tensor = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        queries = {"text": text_tensor}
        keys = {"text": text_tensor, "image": image_tensor}
        values = {"text": text_tensor, "image": image_tensor}
        
        outputs, attention_weights = kernel(queries, keys, values)
        
        self.assertIn("text", outputs)
        self.assertEqual(outputs["text"].shape, (batch_size, seq_len, self.config.hidden_size))
        # attention_weights might be None as per implementation

    def test_qwen3_vl_fusion_kernel_creation(self):
        """Test Qwen3VL2BFusionKernel creation."""
        kernel = Qwen3VL2BFusionKernel(self.config, layer_idx=0)
        self.assertIsInstance(kernel, Qwen3VL2BFusionKernel)
        self.assertEqual(kernel.d_model, self.config.hidden_size)
        self.assertEqual(kernel.nhead, self.config.num_attention_heads)

    def test_qwen3_vl_fusion_kernel_forward(self):
        """Test Qwen3VL2BFusionKernel forward pass."""
        kernel = Qwen3VL2BFusionKernel(self.config, layer_idx=0)
        
        # Create sample inputs
        batch_size = 2
        seq_len = 10
        text_tensor = torch.randn(batch_size, seq_len, self.config.hidden_size)
        image_tensor = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        inputs = {"text": text_tensor, "image": image_tensor}
        
        outputs = kernel(inputs)
        
        self.assertIn("text", outputs)
        self.assertIn("image", outputs)
        self.assertEqual(outputs["text"].shape, (batch_size, seq_len, self.config.hidden_size))
        self.assertEqual(outputs["image"].shape, (batch_size, seq_len, self.config.hidden_size))

    def test_qwen3_vl_vision_language_attention_kernel_creation(self):
        """Test Qwen3VL2BVisionLanguageAttentionKernel creation."""
        kernel = Qwen3VL2BVisionLanguageAttentionKernel(self.config, layer_idx=0)
        self.assertIsInstance(kernel, Qwen3VL2BVisionLanguageAttentionKernel)
        self.assertEqual(kernel.d_model, self.config.hidden_size)
        self.assertEqual(kernel.nhead, self.config.num_attention_heads)

    def test_qwen3_vl_vision_language_attention_kernel_forward(self):
        """Test Qwen3VL2BVisionLanguageAttentionKernel forward pass."""
        kernel = Qwen3VL2BVisionLanguageAttentionKernel(self.config, layer_idx=0)
        
        # Create sample inputs
        batch_size = 2
        num_patches = 10
        seq_len = 15
        vision_features = torch.randn(batch_size, num_patches, self.config.hidden_size)
        language_features = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        fused_output, vision_output, language_output, attention_weights = kernel(
            vision_features, language_features
        )
        
        expected_fused_shape = (batch_size, num_patches + seq_len, self.config.hidden_size)
        self.assertEqual(fused_output.shape, expected_fused_shape)
        self.assertEqual(vision_output.shape, vision_features.shape)
        self.assertEqual(language_output.shape, language_features.shape)

    def test_qwen3_vl_position_encoding_kernel_creation(self):
        """Test Qwen3VL2BPositionEncodingKernel creation."""
        kernel = Qwen3VL2BPositionEncodingKernel(self.config)
        self.assertIsInstance(kernel, Qwen3VL2BPositionEncodingKernel)

    def test_qwen3_vl_position_encoding_kernel_forward(self):
        """Test Qwen3VL2BPositionEncodingKernel forward pass."""
        kernel = Qwen3VL2BPositionEncodingKernel(self.config)
        
        # Create sample inputs
        batch_size = 2
        seq_len = 10
        text_tensor = torch.randn(batch_size, seq_len, self.config.hidden_size)
        image_tensor = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        features = {"text": text_tensor, "image": image_tensor}
        
        encoded_features = kernel(features)
        
        self.assertIn("text", encoded_features)
        self.assertIn("image", encoded_features)
        self.assertEqual(encoded_features["text"].shape, (batch_size, seq_len, self.config.hidden_size))
        self.assertEqual(encoded_features["image"].shape, (batch_size, seq_len, self.config.hidden_size))

    def test_qwen3_vl_mlp_kernel_creation(self):
        """Test Qwen3VL2BMLPKernel creation."""
        kernel = Qwen3VL2BMLPKernel(self.config, layer_idx=0)
        self.assertIsInstance(kernel, Qwen3VL2BMLPKernel)
        self.assertEqual(kernel.gate_proj.out_features, self.config.intermediate_size)

    def test_qwen3_vl_mlp_kernel_forward(self):
        """Test Qwen3VL2BMLPKernel forward pass."""
        kernel = Qwen3VL2BMLPKernel(self.config, layer_idx=0)
        
        # Create sample input
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        output = kernel(x)
        
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_qwen3_vl_rms_norm_kernel_creation(self):
        """Test Qwen3VL2BRMSNormKernel creation."""
        kernel = Qwen3VL2BRMSNormKernel(self.config, layer_idx=0)
        self.assertIsInstance(kernel, Qwen3VL2BRMSNormKernel)
        self.assertEqual(kernel.weight.shape[0], self.config.hidden_size)

    def test_qwen3_vl_rms_norm_kernel_forward(self):
        """Test Qwen3VL2BRMSNormKernel forward pass."""
        kernel = Qwen3VL2BRMSNormKernel(self.config, layer_idx=0)
        
        # Create sample input
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        output = kernel(x)
        
        self.assertEqual(output.shape, x.shape)

    def test_apply_qwen3_vl_cuda_optimizations_to_model(self):
        """Test applying CUDA optimizations to a simple model."""
        # Create a simple test model
        class SimpleTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=16)
                self.linear = nn.Linear(1024, 1024)
                
            def forward(self, x):
                attn_out, _ = self.attention(x, x, x)
                return self.linear(attn_out)
        
        model = SimpleTestModel()
        
        # Apply optimizations
        optimized_model = apply_qwen3_vl_cuda_optimizations_to_model(model, self.config)
        
        # Check that the model is still functional
        x = torch.randn(10, 2, 1024) # Seq, Batch, Dim for MultiheadAttention default
        output = optimized_model(x)
        self.assertEqual(output.shape, (10, 2, 1024))

    def test_get_qwen3_vl_cuda_optimization_report(self):
        """Test getting optimization report."""
        # Create a simple test model
        class SimpleTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1024, 1024)
                
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleTestModel()
        
        report = get_qwen3_vl_cuda_optimization_report(model, self.config)
        
        self.assertIn("model_type", report)
        self.assertIn("optimizations_applied", report)
        self.assertIn("config", report)
        self.assertEqual(report["model_type"], "Qwen3-VL-2B")
        self.assertTrue(report["optimizations_applied"]["qwen3_vl_cross_attention"])

    def test_qwen3_vl_vision_processing_kernel_creation(self):
        """Test Qwen3VL2BVisionProcessingKernel creation."""
        kernel = Qwen3VL2BVisionProcessingKernel(self.config)
        self.assertIsInstance(kernel, Qwen3VL2BVisionProcessingKernel)

    def test_qwen3_vl_vision_processing_kernel_forward(self):
        """Test Qwen3VL2BVisionProcessingKernel forward pass."""
        kernel = Qwen3VL2BVisionProcessingKernel(self.config)

        # Create sample image input
        batch_size = 2
        channels = 3  # RGB
        height = self.config.vision_image_size
        width = self.config.vision_image_size

        pixel_values = torch.randn(batch_size, channels, height, width)

        output, hidden_states = kernel(pixel_values, output_hidden_states=True)

        # Check output shape (batch, num_patches + cls, hidden_size)
        # num_patches = (224 // 14) * (224 // 14) = 16 * 16 = 256
        # +1 for cls token
        num_patches = (self.config.vision_image_size // self.config.vision_patch_size) ** 2

        self.assertEqual(output.shape, (batch_size, num_patches + 1, self.config.vision_hidden_size))

        # Check that hidden states are returned when requested
        self.assertIsNotNone(hidden_states)
        self.assertEqual(len(hidden_states), self.config.vision_num_hidden_layers)

if __name__ == '__main__':
    unittest.main()
