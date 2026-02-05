"""
Test suite for GLM-4.7-Flash Projection Optimizations Implementation

This module tests the projection layer optimizations for the GLM-4.7-Flash model.
"""

import unittest
import torch
import torch.nn as nn
from src.models.specialized.glm_4_7_flash.projection_optimizations import (
    GLM47FlashProjectionLayer,
    GLM47FlashMultiProjectionLayer,
    create_glm47_flash_projection_layer,
    create_glm47_flash_multi_projection_layer,
    apply_glm47_flash_projection_optimizations
)
from src.models.specialized.glm_4_7_flash.projection_optimizations.config import get_glm47_flash_projection_config


class BaseUnitTest(unittest.TestCase):
    """Base test class for GLM-4.7-Flash projection optimizations."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 2
        self.seq_len = 10
        self.input_dim = 512
        self.output_dim = 256
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)


class TestGLM47FlashProjectionLayer(BaseUnitTest):
    """Test cases for GLM-4.7-Flash projection layer implementation."""
    
    def test_projection_layer_initialization(self):
        """Test that the projection layer initializes correctly."""
        layer = GLM47FlashProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        
        self.assertIsInstance(layer, GLM47FlashProjectionLayer)
        self.assertIsInstance(layer.input_projection, nn.Linear)
        self.assertIsInstance(layer.output_projection, nn.Linear)
        self.assertEqual(layer.input_projection.in_features, self.input_dim)
        self.assertEqual(layer.output_projection.out_features, self.output_dim)
    
    def test_projection_layer_forward_pass(self):
        """Test the forward pass of the projection layer."""
        layer = GLM47FlashProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        
        output = layer(self.test_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_projection_layer_with_low_rank(self):
        """Test the projection layer with low-rank optimization."""
        layer = GLM47FlashProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            use_low_rank=True,
            low_rank_dim=64
        )
        
        # Check that intermediate projection exists
        self.assertTrue(hasattr(layer, 'intermediate_projection'))
        self.assertIsInstance(layer.intermediate_projection, nn.Linear)
        
        output = layer(self.test_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_projection_layer_with_group_norm(self):
        """Test the projection layer with group normalization."""
        layer = GLM47FlashProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            use_group_norm=True,
            group_norm_num_groups=16
        )
        
        self.assertIsNotNone(layer.group_norm)
        self.assertIsInstance(layer.group_norm, nn.GroupNorm)
        
        output = layer(self.test_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_projection_layer_different_activations(self):
        """Test the projection layer with different activation functions."""
        activations = ["silu", "gelu", "relu"]
        
        for activation in activations:
            layer = GLM47FlashProjectionLayer(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                activation=activation
            )
            
            output = layer(self.test_input)
            
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
            self.assertTrue(torch.isfinite(output).all())


class TestGLM47FlashMultiProjectionLayer(BaseUnitTest):
    """Test cases for GLM-4.7-Flash multi-projection layer implementation."""
    
    def test_multi_projection_layer_initialization(self):
        """Test that the multi-projection layer initializes correctly."""
        layer = GLM47FlashMultiProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_layers=2
        )
        
        self.assertIsInstance(layer, GLM47FlashMultiProjectionLayer)
        self.assertEqual(len(layer.projection_layers), 2)
        self.assertIsInstance(layer.final_projection, nn.Linear)
        
        for proj_layer in layer.projection_layers:
            self.assertIsInstance(proj_layer, GLM47FlashProjectionLayer)
    
    def test_multi_projection_layer_forward_pass(self):
        """Test the forward pass of the multi-projection layer."""
        layer = GLM47FlashMultiProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_layers=2
        )
        
        output = layer(self.test_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertTrue(torch.isfinite(output).all())


class TestGLM47FlashProjectionFactoryFunctions(BaseUnitTest):
    """Test cases for GLM-4.7-Flash projection layer factory functions."""
    
    def test_create_glm47_flash_projection_layer(self):
        """Test creating a GLM-4.7-Flash projection layer."""
        config = get_glm47_flash_projection_config(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        
        layer = create_glm47_flash_projection_layer(config)
        
        self.assertIsInstance(layer, GLM47FlashProjectionLayer)
        self.assertEqual(layer.input_dim, self.input_dim)
        self.assertEqual(layer.output_dim, self.output_dim)
    
    def test_create_glm47_flash_multi_projection_layer(self):
        """Test creating a GLM-4.7-Flash multi-projection layer."""
        config = get_glm47_flash_projection_config(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_projection_layers=3
        )
        
        layer = create_glm47_flash_multi_projection_layer(config)
        
        self.assertIsInstance(layer, GLM47FlashMultiProjectionLayer)
        self.assertEqual(layer.num_layers, 3)


class TestApplyGLM47FlashProjectionOptimizations(BaseUnitTest):
    """Test cases for applying GLM-4.7-Flash projection optimizations to a model."""
    
    def test_apply_glm47_flash_projection_optimizations(self):
        """Test applying projection optimizations to a model."""
        # Create a simple model with projection layers
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding_projection = nn.Linear(self.input_dim, self.output_dim)
                self.output_projection = nn.Linear(self.output_dim, self.input_dim)
                self.other_layer = nn.Linear(self.input_dim, self.input_dim)
        
        model = SimpleModel()
        config = get_glm47_flash_projection_config(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        
        # Apply projection optimizations
        optimized_model = apply_glm47_flash_projection_optimizations(model, config)
        
        # Check that projection layers were replaced
        self.assertIsInstance(optimized_model.embedding_projection, GLM47FlashProjectionLayer)
        self.assertIsInstance(optimized_model.output_projection, GLM47FlashProjectionLayer)
        self.assertIsInstance(optimized_model.other_layer, nn.Linear)


if __name__ == "__main__":
    unittest.main()