"""
Test suite for Qwen3-VL-2B Projection Layers Implementation

This module tests the projection layer optimizations for the Qwen3-VL-2B model.
"""

from abc import abstractmethod
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from src.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.models.qwen3_vl_2b.multimodal_projector.projection_layers import (
    Qwen3VL2BMultiModalProjector,
    Qwen3VL2BProjectionLayer,
    Qwen3VL2BVisionLanguageProjector,
    apply_qwen3_vl_projection_optimizations,
    create_qwen3_vl_multimodal_projector,
    create_qwen3_vl_projection_layer,
)
from tests.base.unit_test_base import BaseUnitTest


class TestQwen3VL2BProjectionLayer(BaseUnitTest):
    """Test cases for Qwen3-VL-2B projection layer implementation."""

    def test_required_functionality(self):
        """Implementation of abstract method."""
        self.test_projection_layer_initialization()

    def test_projection_layer_initialization(self):
        """Test that the projection layer initializes correctly."""
        vision_dim = 1024
        language_dim = 2048

        layer = Qwen3VL2BProjectionLayer(
            vision_dim=vision_dim, language_dim=language_dim
        )

        self.assertIsInstance(layer, Qwen3VL2BProjectionLayer)
        self.assertEqual(layer.vision_dim, vision_dim)
        self.assertEqual(layer.language_dim, language_dim)
        self.assertIsInstance(layer.vision_projection, nn.Linear)
        self.assertIsInstance(layer.language_projection, nn.Linear)
        self.assertIsInstance(layer.multimodal_projection, nn.Linear)
        self.assertIsInstance(layer.output_projection, nn.Linear)
        self.assertIsInstance(layer.norm, nn.LayerNorm)
        self.assertIsInstance(layer.activation, nn.GELU)
        self.assertIsInstance(layer.dropout, nn.Dropout)

    def test_projection_layer_forward_pass(self):
        """Test the forward pass of the projection layer."""
        vision_dim = 1024
        language_dim = 2048
        batch_size = 2
        seq_len_vision = 10
        seq_len_language = 15

        layer = Qwen3VL2BProjectionLayer(
            vision_dim=vision_dim, language_dim=language_dim
        )

        # Create sample inputs
        vision_features = torch.randn(batch_size, seq_len_vision, vision_dim)
        language_features = torch.randn(batch_size, seq_len_language, language_dim)

        # Run forward pass
        projected_vision, projected_language, fused_features = layer(
            vision_features, language_features
        )

        # Check output shapes
        self.assert_tensor_shape(
            projected_vision, (batch_size, seq_len_vision, language_dim)
        )
        self.assert_tensor_shape(
            projected_language, (batch_size, seq_len_language, language_dim)
        )
        self.assert_tensor_shape(
            fused_features,
            (batch_size, seq_len_vision + seq_len_language, language_dim),
        )

    def test_projection_layer_with_low_rank(self):
        """Test the projection layer with low-rank optimization."""
        vision_dim = 1024
        language_dim = 2048

        layer = Qwen3VL2BProjectionLayer(
            vision_dim=vision_dim,
            language_dim=language_dim,
            use_low_rank=True,
            low_rank_dim=256,
        )

        # Check that vision_projection is a Sequential with two Linear layers
        self.assertIsInstance(layer.vision_projection, nn.Sequential)
        self.assertEqual(len(layer.vision_projection), 2)
        self.assertIsInstance(layer.vision_projection[0], nn.Linear)
        self.assertIsInstance(layer.vision_projection[1], nn.Linear)

        # Check that language_projection is a Sequential with two Linear layers
        self.assertIsInstance(layer.language_projection, nn.Sequential)
        self.assertEqual(len(layer.language_projection), 2)
        self.assertIsInstance(layer.language_projection[0], nn.Linear)
        self.assertIsInstance(layer.language_projection[1], nn.Linear)

    def test_projection_layer_with_group_norm(self):
        """Test the projection layer with group normalization."""
        vision_dim = 1024
        language_dim = 2048

        layer = Qwen3VL2BProjectionLayer(
            vision_dim=vision_dim,
            language_dim=language_dim,
            use_group_norm=True,
            num_groups=16,
        )

        self.assertIsInstance(layer.norm, nn.GroupNorm)

    def test_projection_layer_different_activations(self):
        """Test the projection layer with different activation functions."""
        vision_dim = 1024
        language_dim = 2048

        for activation in ["relu", "swish", "linear"]:
            layer = Qwen3VL2BProjectionLayer(
                vision_dim=vision_dim, language_dim=language_dim, activation=activation
            )

            if activation == "relu":
                self.assertIsInstance(layer.activation, nn.ReLU)
            elif activation == "swish":
                self.assertIsInstance(layer.activation, nn.SiLU)
            elif activation == "linear":
                self.assertIsInstance(layer.activation, nn.Identity)


class TestQwen3VL2BMultiModalProjector(BaseUnitTest):
    """Test cases for Qwen3-VL-2B multimodal projector implementation."""

    def test_required_functionality(self):
        """Implementation of abstract method."""
        self.test_multimodal_projector_initialization()

    def test_multimodal_projector_initialization(self):
        """Test that the multimodal projector initializes correctly."""
        vision_dim = 1024
        language_dim = 2048

        projector = Qwen3VL2BMultiModalProjector(
            vision_dim=vision_dim, language_dim=language_dim, num_layers=2
        )

        self.assertIsInstance(projector, Qwen3VL2BMultiModalProjector)
        self.assertEqual(len(projector.projection_layers), 2)
        self.assertIsInstance(projector.final_projection, nn.Linear)
        for layer in projector.projection_layers:
            self.assertIsInstance(layer, Qwen3VL2BProjectionLayer)

    def test_multimodal_projector_forward_pass(self):
        """Test the forward pass of the multimodal projector."""
        vision_dim = 1024
        language_dim = 2048
        batch_size = 2
        seq_len_vision = 10
        seq_len_language = 15

        projector = Qwen3VL2BMultiModalProjector(
            vision_dim=vision_dim, language_dim=language_dim, num_layers=2
        )

        # Create sample inputs
        vision_features = torch.randn(batch_size, seq_len_vision, vision_dim)
        language_features = torch.randn(batch_size, seq_len_language, language_dim)

        # Run forward pass
        output = projector(vision_features, language_features)

        # Check output shape
        self.assert_tensor_shape(
            output, (batch_size, seq_len_vision + seq_len_language, language_dim)
        )

    def test_multimodal_projector_with_cross_attention(self):
        """Test the multimodal projector with cross-attention enabled."""
        vision_dim = 1024
        language_dim = 2048

        projector = Qwen3VL2BMultiModalProjector(
            vision_dim=vision_dim,
            language_dim=language_dim,
            num_layers=1,
            use_cross_attention=True,
            num_attention_heads=8,
        )

        self.assertTrue(projector.use_cross_attention)
        self.assertIsNotNone(projector.vision_to_language_attn)
        self.assertIsNotNone(projector.language_to_vision_attn)


class TestQwen3VL2BVisionLanguageProjector(BaseUnitTest):
    """Test cases for Qwen3-VL-2B vision-language projector implementation."""

    def test_required_functionality(self):
        """Implementation of abstract method."""
        self.test_vision_language_projector_initialization()

    def test_vision_language_projector_initialization(self):
        """Test that the vision-language projector initializes correctly."""
        config = Qwen3VL2BConfig()
        batch_size = 2
        seq_len_vision = 10
        seq_len_language = 15
        vision_dim = 1024
        language_dim = 2048

        projector = Qwen3VL2BVisionLanguageProjector(config=config)

        self.assertIsInstance(projector, Qwen3VL2BVisionLanguageProjector)
        self.assertEqual(projector.config, config)
        self.assertIsInstance(projector.vision_projection, nn.Linear)
        self.assertIsInstance(projector.language_projection, nn.Linear)
        self.assertIsInstance(projector.layer_norm, nn.LayerNorm)

    def test_vision_language_projector_with_conv_projection(self):
        """Test the vision-language projector with convolutional projection."""
        config = Qwen3VL2BConfig()

        projector = Qwen3VL2BVisionLanguageProjector(
            config=config, use_conv_projection=True, conv_kernel_size=3
        )

        self.assertIsInstance(projector.vision_projection, nn.Conv2d)

    def test_vision_language_projector_with_mlp_fusion(self):
        """Test the vision-language projector with MLP fusion."""
        config = Qwen3VL2BConfig()

        projector = Qwen3VL2BVisionLanguageProjector(
            config=config, use_mlp_fusion=True, mlp_expansion_ratio=2.0
        )

        self.assertTrue(projector.use_mlp_fusion)
        self.assertIsNotNone(projector.mlp_fusion)
        self.assertIsInstance(projector.mlp_fusion, nn.Sequential)


class TestFactoryFunctions(BaseUnitTest):
    """Test cases for factory functions."""

    def test_required_functionality(self):
        """Implementation of abstract method."""
        self.test_create_qwen3_vl_projection_layer()

    def test_create_qwen3_vl_projection_layer(self):
        """Test creating a Qwen3-VL-2B projection layer."""
        config = Qwen3VL2BConfig()

        layer = create_qwen3_vl_projection_layer(config)

        self.assertIsInstance(layer, Qwen3VL2BProjectionLayer)
        self.assertEqual(layer.vision_dim, config.vision_hidden_size)
        self.assertEqual(layer.language_dim, config.hidden_size)

    def test_create_qwen3_vl_multimodal_projector(self):
        """Test creating a Qwen3-VL-2B multimodal projector."""
        config = Qwen3VL2BConfig()

        projector = create_qwen3_vl_multimodal_projector(config)

        self.assertIsInstance(projector, Qwen3VL2BMultiModalProjector)
        self.assertEqual(projector.vision_dim, config.vision_hidden_size)
        self.assertEqual(projector.language_dim, config.hidden_size)


class TestApplyOptimizations(BaseUnitTest):
    """Test cases for applying projection optimizations to model."""

    def test_required_functionality(self):
        """Implementation of abstract method."""
        config = Qwen3VL2BConfig()

        # Create a mock model with projection layers
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_projector = nn.Linear(1024, 2048)
                self.language_projector = nn.Linear(2048, 2048)

        model = MockModel()

        # Apply optimizations
        optimized_model = apply_qwen3_vl_projection_optimizations(model, config)

        # Check that the function ran without error
        self.assertIsInstance(optimized_model, MockModel)

    @patch("torch.nn.Linear")
    def test_apply_qwen3_vl_projection_optimizations(self, mock_linear):
        """Test applying projection optimizations to a model."""
        config = Qwen3VL2BConfig()

        # Create a mock model with projection layers
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_projector = nn.Linear(1024, 2048)
                self.language_projector = nn.Linear(2048, 2048)

        model = MockModel()

        # Apply optimizations
        optimized_model = apply_qwen3_vl_projection_optimizations(model, config)

        # Check that the function ran without error
        self.assertIsInstance(optimized_model, MockModel)


if __name__ == "__main__":
    pytest.main([__file__])
