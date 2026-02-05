"""Test suite for Qwen3-Coder-30B Cross-Fusion Optimizations Implementation

This module tests the cross-fusion layer optimizations for the Qwen3-Coder-30B model.
"""

import unittest
import torch
import torch.nn as nn
from torch import Tensor

from src.models.coding.qwen3_coder_30b.cross_fusion_optimization import (
    CrossFusionConfig,
    Qwen3CoderCrossFusionOptimizer,
    CrossFusionManager,
    create_qwen3_coder_cross_fusion,
    apply_cross_fusion_to_model
)


class BaseUnitTest(unittest.TestCase):
    """Base unit test class with common utilities."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 2
        self.seq_len = 10
        self.hidden_size = 4096  # Qwen3-Coder-30B typical hidden size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestQwen3CoderCrossFusionOptimizer(BaseUnitTest):
    """Test cases for Qwen3-Coder-30B cross-fusion optimizer implementation."""
    
    def test_initialization(self):
        """Test initializing the Qwen3-Coder-30B cross-fusion optimizer."""
        config = CrossFusionConfig()
        config.hidden_size = self.hidden_size
        
        optimizer = Qwen3CoderCrossFusionOptimizer(config)
        
        self.assertIsInstance(optimizer, Qwen3CoderCrossFusionOptimizer)
        self.assertEqual(optimizer.config.hidden_size, self.hidden_size)
        
    def test_forward_pass_shapes(self):
        """Test that forward pass produces correct shapes."""
        config = CrossFusionConfig()
        config.hidden_size = self.hidden_size
        
        optimizer = Qwen3CoderCrossFusionOptimizer(config)
        
        # Create test tensors
        rep1 = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        rep2 = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        
        # Forward pass
        (fused_rep1, fused_rep2), fusion_loss = optimizer(rep1, rep2)
        
        # Check shapes
        self.assertEqual(fused_rep1.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(fused_rep2.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertIsInstance(fusion_loss, Tensor)
        self.assertEqual(fusion_loss.shape, torch.Size([]))  # Scalar
        
    def test_different_sequence_lengths(self):
        """Test fusion with different sequence lengths."""
        config = CrossFusionConfig()
        config.hidden_size = self.hidden_size
        
        optimizer = Qwen3CoderCrossFusionOptimizer(config)
        
        # Create test tensors with different sequence lengths
        rep1 = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        rep2 = torch.randn(self.batch_size, self.seq_len + 5, self.hidden_size)  # Different length
        
        # Forward pass
        (fused_rep1, fused_rep2), fusion_loss = optimizer(rep1, rep2)
        
        # Check shapes - should match original sequence lengths
        self.assertEqual(fused_rep1.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(fused_rep2.shape, (self.batch_size, self.seq_len + 5, self.hidden_size))
        self.assertIsInstance(fusion_loss, Tensor)
        
    def test_gradient_flow(self):
        """Test that gradients flow properly through the fusion layer."""
        config = CrossFusionConfig()
        config.hidden_size = self.hidden_size
        
        optimizer = Qwen3CoderCrossFusionOptimizer(config)
        
        # Create test tensors with gradient tracking
        rep1 = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        rep2 = torch.randn(self.batch_size, self.seq_len, self.hidden_size, requires_grad=True)
        
        # Forward pass
        (fused_rep1, fused_rep2), fusion_loss = optimizer(rep1, rep2)
        
        # Backward pass
        fusion_loss.backward()
        
        # Check that gradients were computed
        self.assertIsNotNone(rep1.grad)
        self.assertIsNotNone(rep2.grad)
        self.assertIsNotNone(optimizer.fusion_up_proj.weight.grad)
        self.assertIsNotNone(optimizer.fusion_gate_proj.weight.grad)
        self.assertIsNotNone(optimizer.fusion_down_proj.weight.grad)


class TestQwen3CoderCrossFusionManager(BaseUnitTest):
    """Test cases for Qwen3-Coder-30B cross-fusion manager implementation."""
    
    def test_initialization(self):
        """Test initializing the cross-fusion manager."""
        config = CrossFusionConfig()
        config.hidden_size = self.hidden_size
        
        manager = CrossFusionManager(config)
        
        self.assertIsInstance(manager, CrossFusionManager)
        self.assertEqual(len(manager.fusion_methods), 5)  # 5 default methods registered
        
    def test_fusion_method_registration(self):
        """Test registering and retrieving fusion methods."""
        config = CrossFusionConfig()
        config.hidden_size = self.hidden_size
        
        manager = CrossFusionManager(config)
        
        # Check that default methods are registered
        self.assertIn("qwen3_coder_specific", manager.fusion_methods)
        self.assertIn("contrastive", manager.fusion_methods)
        self.assertIn("attention", manager.fusion_methods)
        self.assertIn("learned_projection", manager.fusion_methods)
        self.assertIn("similarity_based", manager.fusion_methods)
        
        # Get a specific optimizer
        optimizer = manager.get_fusion_optimizer("qwen3_coder_specific")
        self.assertIsInstance(optimizer, Qwen3CoderCrossFusionOptimizer)
        
    def test_fuse_representations(self):
        """Test fusing representations using the manager."""
        config = CrossFusionConfig()
        config.hidden_size = self.hidden_size
        
        manager = CrossFusionManager(config)
        
        # Create test tensors
        rep1 = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        rep2 = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        
        # Fuse representations
        fused_rep1, fused_rep2, fusion_loss = manager.fuse_representations(
            rep1, rep2, method_name="qwen3_coder_specific"
        )
        
        # Check shapes
        self.assertEqual(fused_rep1.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(fused_rep2.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertIsInstance(fusion_loss, Tensor)
        
    def test_evaluate_fusion_quality(self):
        """Test evaluating fusion quality."""
        config = CrossFusionConfig()
        config.hidden_size = self.hidden_size
        
        manager = CrossFusionManager(config)
        
        # Create test tensors
        original_rep1 = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        original_rep2 = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        
        # Create slightly modified versions as "fused" reps
        fused_rep1 = original_rep1 + 0.1 * torch.randn_like(original_rep1)
        fused_rep2 = original_rep2 + 0.1 * torch.randn_like(original_rep2)
        
        # Evaluate fusion quality
        quality_metrics = manager.evaluate_fusion_quality(
            original_rep1, original_rep2, fused_rep1, fused_rep2
        )
        
        # Check that metrics are returned
        self.assertIsInstance(quality_metrics, dict)
        self.assertIn("rep1_preservation", quality_metrics)
        self.assertIn("rep2_preservation", quality_metrics)
        self.assertIn("cross_rep_similarity", quality_metrics)
        self.assertIn("original_cross_rep_similarity", quality_metrics)
        self.assertIn("fusion_improvement", quality_metrics)
        self.assertIn("overall_fusion_score", quality_metrics)


class TestQwen3CoderCrossFusionFactoryFunctions(BaseUnitTest):
    """Test cases for Qwen3-Coder-30B cross-fusion factory functions."""
    
    def test_create_qwen3_coder_cross_fusion(self):
        """Test creating a cross-fusion manager."""
        config = CrossFusionConfig()
        config.hidden_size = self.hidden_size
        
        manager = create_qwen3_coder_cross_fusion(config)
        
        self.assertIsInstance(manager, CrossFusionManager)
        self.assertEqual(len(manager.fusion_methods), 5)  # 5 default methods registered


class TestApplyQwen3CoderCrossFusionToModel(BaseUnitTest):
    """Test cases for applying Qwen3-Coder-30B cross-fusion to a model."""
    
    def test_apply_cross_fusion_to_model(self):
        """Test applying cross-fusion optimizations to a model."""
        config = CrossFusionConfig()
        config.hidden_size = self.hidden_size
        
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding_projection = nn.Linear(self.hidden_size, self.hidden_size)
                self.output_projection = nn.Linear(self.hidden_size, self.hidden_size)
                
        model = SimpleModel()
        
        # Apply cross-fusion optimizations
        optimized_model = apply_cross_fusion_to_model(model, config)
        
        # Check that fusion components were added
        self.assertTrue(hasattr(optimized_model, "cross_fusion_manager"))
        self.assertTrue(hasattr(optimized_model, "cross_fusion_optimizer"))
        self.assertTrue(hasattr(optimized_model, "perform_cross_fusion"))
        
        # Check that the model still has original components
        self.assertIsInstance(optimized_model.embedding_projection, nn.Linear)
        self.assertIsInstance(optimized_model.output_projection, nn.Linear)
        
        # Test the perform_cross_fusion method
        rep1 = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        rep2 = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        
        fused_rep1, fused_rep2, fusion_loss = optimized_model.perform_cross_fusion(rep1, rep2)
        
        self.assertEqual(fused_rep1.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertEqual(fused_rep2.shape, (self.batch_size, self.seq_len, self.hidden_size))
        self.assertIsInstance(fusion_loss, Tensor)


if __name__ == "__main__":
    unittest.main()