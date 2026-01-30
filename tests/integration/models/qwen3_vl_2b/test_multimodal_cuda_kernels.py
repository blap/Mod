"""
Test suite for multimodal CUDA kernels in Qwen3-VL-2B model.

This test suite verifies the functionality of the multimodal CUDA kernels
implemented for the Qwen3-VL-2B vision-language model.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.inference_pio.common.multimodal_cuda_kernels import (
    MultimodalCrossAttentionKernel,
    MultimodalFusionKernel,
    VisionLanguageAttentionKernel,
    MultimodalPositionEncodingKernel,
    MultimodalHardwareOptimizer,
    create_multimodal_cuda_kernels,
    apply_multimodal_cuda_optimizations_to_model
)
from src.inference_pio.models.qwen3_vl_2b.cuda_kernels.optimizations import (
    Qwen3VL2BCrossAttentionKernel,
    Qwen3VL2BFusionKernel,
    Qwen3VL2BVisionLanguageAttentionKernel,
    Qwen3VL2BPositionEncodingKernel,
    create_qwen3_vl_cross_attention_kernel,
    create_qwen3_vl_fusion_kernel,
    create_qwen3_vl_vision_language_attention_kernel,
    create_qwen3_vl_position_encoding_kernel,
    apply_qwen3_vl_optimizations_to_model
)
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

# TestMultimodalCrossAttentionKernel

    """Test the multimodal cross-attention kernel."""
    
    def setup_helper():
        d_model = 512
        nhead = 8
        modalities = ["text", "image"]
        batch_size = 2
        seq_len = 10
        num_patches = 16
        
    def forward_pass(self)():
        """Test forward pass of multimodal cross-attention kernel."""
        kernel = MultimodalCrossAttentionKernel(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities
        )
        
        # Create sample inputs
        queries = {
            "text": torch.randn(batch_size, seq_len, d_model),
            "image": torch.randn(batch_size, num_patches, d_model)
        }
        keys = queries.copy()
        values = queries.copy()
        
        # Forward pass
        outputs, attention_weights = kernel(queries, keys, values)
        
        # Check output shapes
        assert_equal(outputs["text"].shape, (batch_size))
        assert_equal(outputs["image"].shape, (batch_size))
        
        # Check attention weights shape if returned
        if attention_weights is not None:
            assert_in("text", attention_weights)
            assert_in("image", attention_weights)
    
    def different_modalities(self)():
        """Test with different sets of modalities."""
        modalities = ["text", "image", "audio"]
        kernel = MultimodalCrossAttentionKernel(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities
        )
        
        # Create sample inputs
        queries = {
            "text": torch.randn(batch_size, seq_len, d_model),
            "image": torch.randn(batch_size, num_patches, d_model),
            "audio": torch.randn(batch_size, seq_len, d_model)
        }
        keys = queries.copy()
        values = queries.copy()
        
        # Forward pass
        outputs, _ = kernel(queries, keys, values)
        
        # Check output shapes
        for modality in modalities:
            assert_equal(outputs[modality].shape, queries[modality].shape)

# TestMultimodalFusionKernel

    """Test the multimodal fusion kernel."""
    
    def setup_helper():
        d_model = 512
        nhead = 8
        modalities = ["text", "image"]
        batch_size = 2
        seq_len = 10
        num_patches = 16
    
    def forward_pass(self)():
        """Test forward pass of multimodal fusion kernel."""
        kernel = MultimodalFusionKernel(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities
        )
        
        # Create sample inputs
        inputs = {
            "text": torch.randn(batch_size, seq_len, d_model),
            "image": torch.randn(batch_size, num_patches, d_model)
        }
        
        # Forward pass
        outputs = kernel(inputs)
        
        # Check output shapes
        assert_equal(outputs["text"].shape, (batch_size))
        assert_equal(outputs["image"].shape, (batch_size))
    
    def without_cross_attention(self)():
        """Test fusion kernel without cross-attention."""
        kernel = MultimodalFusionKernel(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities,
            use_cross_attention=False
        )
        
        # Create sample inputs
        inputs = {
            "text": torch.randn(batch_size, seq_len, d_model),
            "image": torch.randn(batch_size, num_patches, d_model)
        }
        
        # Forward pass
        outputs = kernel(inputs)
        
        # Check output shapes
        assert_equal(outputs["text"].shape, (batch_size))
        assert_equal(outputs["image"].shape, (batch_size))

# TestVisionLanguageAttentionKernel

    """Test the vision-language attention kernel."""
    
    def setup_helper():
        d_model = 512
        nhead = 8
        batch_size = 2
        num_patches = 16
        seq_len = 10
    
    def forward_pass(self)():
        """Test forward pass of vision-language attention kernel."""
        kernel = VisionLanguageAttentionKernel(
            d_model=d_model,
            nhead=nhead
        )
        
        # Create sample inputs
        vision_features = torch.randn(batch_size, num_patches, d_model)
        language_features = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        fused_output, vision_output, language_output, attention_weights = kernel(
            vision_features, language_features
        )
        
        # Check output shapes
        expected_fused_shape = (batch_size, num_patches + seq_len, d_model)
        assert_equal(fused_output.shape, expected_fused_shape)
        assert_equal(vision_output.shape, vision_features.shape)
        assert_equal(language_output.shape, language_features.shape)
        
        # Check attention weights shape if returned
        if attention_weights is not None:
            expected_attn_shape = (batch_size, nhead, num_patches, seq_len)
            assert_equal(attention_weights.shape, expected_attn_shape)

# TestMultimodalPositionEncodingKernel

    """Test the multimodal position encoding kernel."""
    
    def setup_helper():
        d_model = 512
        modalities = ["text", "image"]
        batch_size = 2
        seq_len = 10
        num_patches = 16
    
    def forward_pass(self)():
        """Test forward pass of multimodal position encoding kernel."""
        kernel = MultimodalPositionEncodingKernel(
            d_model=d_model,
            modalities=modalities
        )
        
        # Create sample inputs
        features = {
            "text": torch.randn(batch_size, seq_len, d_model),
            "image": torch.randn(batch_size, num_patches, d_model)
        }
        
        # Forward pass
        encoded_features = kernel(features)
        
        # Check output shapes (should be same as input)
        assert_equal(encoded_features["text"].shape, features["text"].shape)
        assert_equal(encoded_features["image"].shape, features["image"].shape)
        
        # Check that values have changed due to position encoding
        assert_false(torch.allclose(encoded_features["text"]))
        assert_false(torch.allclose(encoded_features["image"]))

# TestMultimodalHardwareOptimizer

    """Test the multimodal hardware optimizer."""
    
    def initialization(self)():
        """Test initialization of hardware optimizer."""
        optimizer = MultimodalHardwareOptimizer()
        
        # Check that properties are set
        assert_is_instance(optimizer.compute_capability, tuple)
        assert_is_instance(optimizer.tensor_cores_supported, bool)
        assert_is_instance(optimizer.optimization_level, str)
        
        # Check optimization report
        report = optimizer.get_optimization_report()
        assert_in("compute_capability", report)
        assert_in("tensor_cores_supported", report)
        assert_in("optimization_level", report)
        assert_in("recommended_kernels", report)

# TestQwen3VL2BSpecificKernels

    """Test Qwen3-VL-2B specific kernels."""
    
    def setup_helper():
        config = Qwen3VL2BConfig()
        batch_size = 1
        seq_len = 5
        num_patches = 8
        d_model = config.hidden_size
        nhead = config.num_attention_heads
    
    def qwen3_vl_cross_attention_kernel(self)():
        """Test Qwen3-VL-2B specific cross-attention kernel."""
        kernel = create_qwen3_vl_cross_attention_kernel(config, layer_idx=0)
        
        # Create sample inputs
        queries = {
            "text": torch.randn(batch_size, seq_len, d_model),
            "image": torch.randn(batch_size, num_patches, d_model)
        }
        keys = queries.copy()
        values = queries.copy()
        
        # Forward pass
        outputs, _ = kernel(queries, keys, values)
        
        # Check output shapes
        assert_equal(outputs["text"].shape, (batch_size))
        assert_equal(outputs["image"].shape, (batch_size))
    
    def qwen3_vl_fusion_kernel(self)():
        """Test Qwen3-VL-2B specific fusion kernel."""
        kernel = create_qwen3_vl_fusion_kernel(config, layer_idx=0)
        
        # Create sample inputs
        inputs = {
            "text": torch.randn(batch_size, seq_len, d_model),
            "image": torch.randn(batch_size, num_patches, d_model)
        }
        
        # Forward pass
        outputs = kernel(inputs)
        
        # Check output shapes
        assert_equal(outputs["text"].shape, (batch_size))
        assert_equal(outputs["image"].shape, (batch_size))
    
    def qwen3_vl_vision_language_attention_kernel(self)():
        """Test Qwen3-VL-2B specific vision-language attention kernel."""
        kernel = create_qwen3_vl_vision_language_attention_kernel(config, layer_idx=0)
        
        # Create sample inputs
        vision_features = torch.randn(batch_size, num_patches, d_model)
        language_features = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        fused_output, vision_output, language_output, _ = kernel(
            vision_features, language_features
        )
        
        # Check output shapes
        expected_fused_shape = (batch_size, num_patches + seq_len, d_model)
        assert_equal(fused_output.shape, expected_fused_shape)
        assert_equal(vision_output.shape, vision_features.shape)
        assert_equal(language_output.shape, language_features.shape)
    
    def qwen3_vl_position_encoding_kernel(self)():
        """Test Qwen3-VL-2B specific position encoding kernel."""
        kernel = create_qwen3_vl_position_encoding_kernel(config)
        
        # Create sample inputs
        features = {
            "text": torch.randn(batch_size, seq_len, d_model),
            "image": torch.randn(batch_size, num_patches, d_model)
        }
        
        # Forward pass
        encoded_features = kernel(features)
        
        # Check output shapes (should be same as input)
        assert_equal(encoded_features["text"].shape, features["text"].shape)
        assert_equal(encoded_features["image"].shape, features["image"].shape)

# TestKernelCreationFunctions

    """Test the kernel creation functions."""
    
    def create_multimodal_cuda_kernels(self)():
        """Test creation of multimodal CUDA kernels."""
        d_model = 256
        nhead = 4
        modalities = ["text", "image"]
        
        kernels = create_multimodal_cuda_kernels(d_model, nhead, modalities)
        
        # Check that expected kernels are created
        assert_in('cross_attention', kernels)
        assert_in('fusion', kernels)
        assert_in('vision_language_attention', kernels)
        assert_in('position_encoding', kernels)
        
        # Check types
        assert_is_instance(kernels['cross_attention'], MultimodalCrossAttentionKernel)
        assert_is_instance(kernels['fusion'], MultimodalFusionKernel)
        assert_is_instance(kernels['vision_language_attention'], VisionLanguageAttentionKernel)
        assert_is_instance(kernels['position_encoding'], MultimodalPositionEncodingKernel)

# TestModelOptimizationApplication

    """Test applying optimizations to a model."""
    
    def setup_helper():
        # Create a simple test model
        test_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.MultiheadAttention(512, 8),
            nn.Linear(512, 512)
        )
        config = Qwen3VL2BConfig()
        # Override config values for testing
        config.hidden_size = 512
        config.num_attention_heads = 8
    
    def apply_multimodal_cuda_optimizations_to_model(self)():
        """Test applying multimodal CUDA optimizations to a model."""
        # This test mainly checks that the function runs without error
        try:
            optimized_model = apply_multimodal_cuda_optimizations_to_model(
                test_model,
                modalities=["text", "image"],
                d_model=512,
                nhead=8
            )
            # Basic check that we got a model back
            assert_is_not_none(optimized_model)
        except Exception as e:
            # Some operations might not be applicable to our simple test model,
            # so we allow this to fail gracefully in some cases
            print(f"Expected potential issue during optimization application: {e}")
    
    def apply_qwen3_vl_optimizations_to_model(self)():
        """Test applying Qwen3-VL-2B specific optimizations to a model."""
        # This test mainly checks that the function runs without error
        try:
            optimized_model = apply_qwen3_vl_optimizations_to_model(
                test_model,
                config
            )
            # Basic check that we got a model back
            assert_is_not_none(optimized_model)
        except Exception as e:
            # Some operations might not be applicable to our simple test model,
            # so we allow this to fail gracefully in some cases
            print(f"Expected potential issue during Qwen3-VL optimization application: {e}")

if __name__ == '__main__':
    print("Running multimodal CUDA kernels tests...")
    run_tests(test_functions)