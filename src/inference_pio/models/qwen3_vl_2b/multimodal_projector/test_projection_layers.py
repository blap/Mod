"""
Test suite for Qwen3-VL-2B Projection Layers Implementation

This module tests the projection layer optimizations for the Qwen3-VL-2B model.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from src.inference_pio.models.qwen3_vl_2b.multimodal_projector.projection_layers import (
    Qwen3VL2BProjectionLayer,
    Qwen3VL2BMultiModalProjector,
    Qwen3VL2BVisionLanguageProjector,
    create_qwen3_vl_projection_layer,
    create_qwen3_vl_multimodal_projector,
    apply_qwen3_vl_projection_optimizations
)
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

# TestQwen3VL2BProjectionLayer

    """Test cases for Qwen3-VL-2B projection layer implementation."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        vision_dim = 1024
        language_dim = 2048
        batch_size = 2
        seq_len_vision = 10
        seq_len_language = 15

    def projection_layer_initialization(self)():
        """Test that the projection layer initializes correctly."""
        layer = Qwen3VL2BProjectionLayer(
            vision_dim=vision_dim,
            language_dim=language_dim
        )
        
        assert_is_instance(layer, Qwen3VL2BProjectionLayer)
        assert_equal(layer.vision_dim, vision_dim)
        assert_equal(layer.language_dim, language_dim)
        assert_is_instance(layer.vision_projection, nn.Linear)
        assert_is_instance(layer.language_projection, nn.Linear)
        assert_is_instance(layer.multimodal_projection, nn.Linear)
        assert_is_instance(layer.output_projection, nn.Linear)
        assert_is_instance(layer.norm, nn.LayerNorm)
        assert_is_instance(layer.activation, nn.GELU)
        assert_is_instance(layer.dropout, nn.Dropout)

    def projection_layer_forward_pass(self)():
        """Test the forward pass of the projection layer."""
        layer = Qwen3VL2BProjectionLayer(
            vision_dim=vision_dim,
            language_dim=language_dim
        )
        
        # Create sample inputs
        vision_features = torch.randn(batch_size, seq_len_vision, vision_dim)
        language_features = torch.randn(batch_size, seq_len_language, language_dim)
        
        # Run forward pass
        projected_vision, projected_language, fused_features = layer(
            vision_features, language_features
        )
        
        # Check output shapes
        assert_equal(projected_vision.shape, (batch_size))
        assert_equal(projected_language.shape, (batch_size))
        assert_equal(fused_features.shape, (batch_size), language_dim))

    def projection_layer_with_low_rank(self)():
        """Test the projection layer with low-rank optimization."""
        layer = Qwen3VL2BProjectionLayer(
            vision_dim=vision_dim,
            language_dim=language_dim,
            use_low_rank=True,
            low_rank_dim=256
        )
        
        # Check that vision_projection is a Sequential with two Linear layers
        assert_is_instance(layer.vision_projection, nn.Sequential)
        assert_equal(len(layer.vision_projection), 2)
        assert_is_instance(layer.vision_projection[0], nn.Linear)
        assert_is_instance(layer.vision_projection[1], nn.Linear)
        
        # Check that language_projection is a Sequential with two Linear layers
        assert_is_instance(layer.language_projection, nn.Sequential)
        assert_equal(len(layer.language_projection), 2)
        assert_is_instance(layer.language_projection[0], nn.Linear)
        assert_is_instance(layer.language_projection[1], nn.Linear)

    def projection_layer_with_group_norm(self)():
        """Test the projection layer with group normalization."""
        layer = Qwen3VL2BProjectionLayer(
            vision_dim=vision_dim,
            language_dim=language_dim,
            use_group_norm=True,
            num_groups=16
        )
        
        assert_is_instance(layer.norm, nn.GroupNorm)

    def projection_layer_different_activations(self)():
        """Test the projection layer with different activation functions."""
        for activation in ["relu", "swish", "linear"]:
            layer = Qwen3VL2BProjectionLayer(
                vision_dim=vision_dim,
                language_dim=language_dim,
                activation=activation
            )
            
            if activation == "relu":
                assert_is_instance(layer.activation, nn.ReLU)
            elif activation == "swish":
                assert_is_instance(layer.activation, nn.SiLU)
            elif activation == "linear":
                assert_is_instance(layer.activation, nn.Identity)

# TestQwen3VL2BMultiModalProjector

    """Test cases for Qwen3-VL-2B multimodal projector implementation."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        vision_dim = 1024
        language_dim = 2048
        batch_size = 2
        seq_len_vision = 10
        seq_len_language = 15

    def multimodal_projector_initialization(self)():
        """Test that the multimodal projector initializes correctly."""
        projector = Qwen3VL2BMultiModalProjector(
            vision_dim=vision_dim,
            language_dim=language_dim,
            num_layers=2
        )
        
        assert_is_instance(projector, Qwen3VL2BMultiModalProjector)
        assert_equal(len(projector.projection_layers), 2)
        assert_is_instance(projector.final_projection, nn.Linear)
        for layer in projector.projection_layers:
            assert_is_instance(layer, Qwen3VL2BProjectionLayer)

    def multimodal_projector_forward_pass(self)():
        """Test the forward pass of the multimodal projector."""
        projector = Qwen3VL2BMultiModalProjector(
            vision_dim=vision_dim,
            language_dim=language_dim,
            num_layers=2
        )
        
        # Create sample inputs
        vision_features = torch.randn(batch_size, seq_len_vision, vision_dim)
        language_features = torch.randn(batch_size, seq_len_language, language_dim)
        
        # Run forward pass
        output = projector(vision_features, language_features)
        
        # Check output shape
        assert_equal(output.shape, (batch_size))

    def multimodal_projector_with_cross_attention(self)():
        """Test the multimodal projector with cross-attention enabled."""
        projector = Qwen3VL2BMultiModalProjector(
            vision_dim=vision_dim,
            language_dim=language_dim,
            num_layers=1,
            use_cross_attention=True,
            num_attention_heads=8
        )
        
        assert_true(projector.use_cross_attention)
        assert_is_not_none(projector.vision_to_language_attn)
        assertIsNotNone(projector.language_to_vision_attn)

# TestQwen3VL2BVisionLanguageProjector

    """Test cases for Qwen3-VL-2B vision-language projector implementation."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = Qwen3VL2BConfig()
        batch_size = 2
        seq_len_vision = 10
        seq_len_language = 15
        vision_dim = 1024
        language_dim = 2048

    def vision_language_projector_initialization(self)():
        """Test that the vision-language projector initializes correctly."""
        projector = Qwen3VL2BVisionLanguageProjector(
            config=config
        )
        
        assert_is_instance(projector)
        assert_equal(projector.config)
        assertIsInstance(projector.vision_projection, nn.Linear)
        assert_is_instance(projector.language_projection, nn.Linear)
        assert_is_instance(projector.layer_norm, nn.LayerNorm)

    def vision_language_projector_with_conv_projection(self)():
        """Test the vision-language projector with convolutional projection."""
        projector = Qwen3VL2BVisionLanguageProjector(
            config=config,
            use_conv_projection=True,
            conv_kernel_size=3
        )
        
        assert_is_instance(projector.vision_projection, nn.Conv2d)

    def vision_language_projector_with_mlp_fusion(self)():
        """Test the vision-language projector with MLP fusion."""
        projector = Qwen3VL2BVisionLanguageProjector(
            config=config,
            use_mlp_fusion=True,
            mlp_expansion_ratio=2.0
        )
        
        assert_true(projector.use_mlp_fusion)
        assert_is_not_none(projector.mlp_fusion)
        assert_is_instance(projector.mlp_fusion)

# TestFactoryFunctions

    """Test cases for factory functions."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = Qwen3VL2BConfig()

    def create_qwen3_vl_projection_layer(self)():
        """Test creating a Qwen3-VL-2B projection layer."""
        layer = create_qwen3_vl_projection_layer(config)
        
        assertIsInstance(layer)
        assert_equal(layer.vision_dim, config.vision_hidden_size)
        assert_equal(layer.language_dim, config.hidden_size)

    def create_qwen3_vl_multimodal_projector(self)():
        """Test creating a Qwen3-VL-2B multimodal projector."""
        projector = create_qwen3_vl_multimodal_projector(config)
        
        assert_is_instance(projector, Qwen3VL2BMultiModalProjector)
        assert_equal(projector.vision_dim, config.vision_hidden_size)
        assert_equal(projector.language_dim, config.hidden_size)

# TestApplyOptimizations

    """Test cases for applying projection optimizations to model."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = Qwen3VL2BConfig()

    @patch('torch.nn.Linear')
    def apply_qwen3_vl_projection_optimizations(self, mock_linear)():
        """Test applying projection optimizations to a model."""
        # Create a mock model with projection layers
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                vision_projector = nn.Linear(1024, 2048)
                language_projector = nn.Linear(2048, 2048)
                
        model = MockModel()
        
        # Apply optimizations
        optimized_model = apply_qwen3_vl_projection_optimizations(model, config)
        
        # Check that the function ran without error
        assert_is_instance(optimized_model, MockModel)

if __name__ == '__main__':
    print("Running Qwen3-VL-2B Projection Layer Tests...")
    
    # Run all tests
    run_tests(test_functions)