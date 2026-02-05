"""
Test suite for Qwen3-0.6B Projection Optimizations Implementation

This module tests the projection layer optimizations for the Qwen3-0.6B model.
"""

import unittest
import torch
import torch.nn as nn
from src.models.language.qwen3_0_6b.projection_optimizations import (
    Qwen3_0_6BProjectionLayer,
    Qwen3_0_6BMultiProjectionLayer,
    create_qwen3_0_6b_projection_layer,
    create_qwen3_0_6b_multi_projection_layer,
    apply_qwen3_0_6b_projection_optimizations
)
from src.models.language.qwen3_0_6b.projection_optimizations.config import get_qwen3_0_6b_projection_config


class BaseUnitTest(unittest.TestCase):
    """Base test class for Qwen3-0.6B projection optimizations."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 2
        self.seq_len = 10
        self.input_dim = 512
        self.output_dim = 256
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.input_dim)


class TestQwen3_0_6BProjectionLayer(BaseUnitTest):
    """Test cases for Qwen3-0.6B projection layer implementation."""
    
    def test_projection_layer_initialization(self):
        """Test that the projection layer initializes correctly."""
        layer = Qwen3_0_6BProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        
        self.assertIsInstance(layer, Qwen3_0_6BProjectionLayer)
        self.assertIsInstance(layer.input_projection, nn.Linear)
        self.assertIsInstance(layer.output_projection, nn.Linear)
        self.assertEqual(layer.input_projection.in_features, self.input_dim)
        self.assertEqual(layer.output_projection.out_features, self.output_dim)
    
    def test_projection_layer_forward_pass(self):
        """Test the forward pass of the projection layer."""
        layer = Qwen3_0_6BProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        
        output = layer(self.test_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_projection_layer_with_low_rank(self):
        """Test the projection layer with low-rank optimization."""
        layer = Qwen3_0_6BProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            use_low_rank=True,
            low_rank_dim=32  # Smaller rank for lightweight model
        )
        
        # Check that intermediate projection exists
        self.assertTrue(hasattr(layer, 'intermediate_projection'))
        self.assertIsInstance(layer.intermediate_projection, nn.Linear)
        
        output = layer(self.test_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_projection_layer_with_group_norm(self):
        """Test the projection layer with group normalization."""
        layer = Qwen3_0_6BProjectionLayer(
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
    
    def test_projection_layer_with_lightweight_optimizations(self):
        """Test the projection layer with lightweight optimizations."""
        layer = Qwen3_0_6BProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            use_lightweight_optimizations=True
        )
        
        self.assertIsNotNone(layer.depthwise_conv)
        self.assertIsInstance(layer.depthwise_conv, nn.Conv1d)
        self.assertIsInstance(layer.pointwise_conv, nn.Conv1d)
        
        output = layer(self.test_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertTrue(torch.isfinite(output).all())
    
    def test_projection_layer_different_activations(self):
        """Test the projection layer with different activation functions."""
        activations = ["silu", "gelu", "relu"]
        
        for activation in activations:
            layer = Qwen3_0_6BProjectionLayer(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                activation=activation
            )
            
            output = layer(self.test_input)
            
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
            self.assertTrue(torch.isfinite(output).all())


class TestQwen3_0_6BMultiProjectionLayer(BaseUnitTest):
    """Test cases for Qwen3-0.6B multi-projection layer implementation."""
    
    def test_multi_projection_layer_initialization(self):
        """Test that the multi-projection layer initializes correctly."""
        layer = Qwen3_0_6BMultiProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_layers=2
        )
        
        self.assertIsInstance(layer, Qwen3_0_6BMultiProjectionLayer)
        self.assertEqual(len(layer.projection_layers), 2)
        self.assertIsInstance(layer.final_projection, nn.Linear)
        
        for proj_layer in layer.projection_layers:
            self.assertIsInstance(proj_layer, Qwen3_0_6BProjectionLayer)
    
    def test_multi_projection_layer_forward_pass(self):
        """Test the forward pass of the multi-projection layer."""
        layer = Qwen3_0_6BMultiProjectionLayer(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_layers=2
        )
        
        output = layer(self.test_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertTrue(torch.isfinite(output).all())


class TestQwen3_0_6BProjectionFactoryFunctions(BaseUnitTest):
    """Test cases for Qwen3-0.6B projection layer factory functions."""
    
    def test_create_qwen3_0_6b_projection_layer(self):
        """Test creating a Qwen3-0.6B projection layer."""
        config = get_qwen3_0_6b_projection_config(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        
        layer = create_qwen3_0_6b_projection_layer(config)
        
        self.assertIsInstance(layer, Qwen3_0_6BProjectionLayer)
        self.assertEqual(layer.input_dim, self.input_dim)
        self.assertEqual(layer.output_dim, self.output_dim)
    
    def test_create_qwen3_0_6b_multi_projection_layer(self):
        """Test creating a Qwen3-0.6B multi-projection layer."""
        config = get_qwen3_0_6b_projection_config(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_projection_layers=3
        )
        
        layer = create_qwen3_0_6b_multi_projection_layer(config)
        
        self.assertIsInstance(layer, Qwen3_0_6BMultiProjectionLayer)
        self.assertEqual(layer.num_layers, 3)


class TestApplyQwen3_0_6BProjectionOptimizations(BaseUnitTest):
    """Test cases for applying Qwen3-0.6B projection optimizations to a model."""
    
    def test_apply_qwen3_0_6b_projection_optimizations(self):
        """Test applying projection optimizations to a model."""
        # Create a simple model with projection layers
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding_projection = nn.Linear(self.input_dim, self.output_dim)
                self.output_projection = nn.Linear(self.output_dim, self.input_dim)
                self.other_layer = nn.Linear(self.input_dim, self.input_dim)
        
        model = SimpleModel()
        config = get_qwen3_0_6b_projection_config(
            input_dim=self.input_dim,
            output_dim=self.output_dim
        )
        
        # Apply projection optimizations
        optimized_model = apply_qwen3_0_6b_projection_optimizations(model, config)
        
        # Check that projection layers were replaced
        self.assertIsInstance(optimized_model.embedding_projection, Qwen3_0_6BProjectionLayer)
        self.assertIsInstance(optimized_model.output_projection, Qwen3_0_6BProjectionLayer)
        self.assertIsInstance(optimized_model.other_layer, nn.Linear)


if __name__ == "__main__":
    unittest.main()