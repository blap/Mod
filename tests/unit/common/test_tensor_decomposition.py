"""
Tests for Tensor Decomposition System

This module contains comprehensive tests for the tensor decomposition system.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import numpy as np
from ..tensor_decomposition import (
    TensorDecomposer,
    AdaptiveTensorDecomposer,
    get_tensor_decomposer,
    decompose_model_weights,
    recompose_model_weights
)

# TestTensorDecomposer

    """Test cases for the TensorDecomposer class."""

    def setup_helper():
        """Set up test fixtures."""
        device = "cpu"  # Use CPU for testing to ensure compatibility
        decomposer = TensorDecomposer(
            decomposition_method="cp_decomposition",
            rank_ratio=0.5,
            device=device
        )

    def cp_decomposition_2d_tensor(self)():
        """Test CP decomposition on a 2D tensor."""
        # Create a random 2D tensor
        tensor = torch.randn(10, 8, device=device)
        
        # Decompose the tensor
        decomposed_data, metadata = decomposer.decompose_tensor(tensor, "test_2d")
        
        # Check that decomposition was successful
        assert_false(metadata.get("decomposition_failed"))
        assert_in("actual_compression_ratio", metadata)
        
        # Recompose the tensor
        recomposed_tensor = decomposer.recompose_tensor(decomposed_data, metadata)
        
        # Check that shapes match
        assert_equal(recomposed_tensor.shape, tensor.shape)
        
        # Check that the recomposed tensor is approximately equal to the original
        # (with some tolerance for approximation errors)
        reconstruction_error = torch.mean((tensor - recomposed_tensor) ** 2)
        assert_less(reconstruction_error.item(), 1.0)  # Reasonable threshold

    def tucker_decomposition_3d_tensor(self)():
        """Test Tucker decomposition on a 3D tensor."""
        # Create a random 3D tensor
        tensor = torch.randn(6, 4, 5, device=device)
        
        # Change decomposition method for this test
        tucker_decomposer = TensorDecomposer(
            decomposition_method="tucker_decomposition",
            rank_ratio=0.5,
            device=device
        )
        
        # Decompose the tensor
        decomposed_data, metadata = tucker_decomposer.decompose_tensor(tensor, "test_3d")
        
        # Check that decomposition was successful
        assert_false(metadata.get("decomposition_failed"))
        assert_in("actual_compression_ratio", metadata)
        
        # Recompose the tensor
        recomposed_tensor = tucker_decomposer.recompose_tensor(decomposed_data, metadata)
        
        # Check that shapes match
        assert_equal(recomposed_tensor.shape, tensor.shape)
        
        # Check that the recomposed tensor is approximately equal to the original
        reconstruction_error = torch.mean((tensor - recomposed_tensor) ** 2)
        assert_less(reconstruction_error.item(), 1.0)  # Reasonable threshold

    def matrix_svd_decomposition(self)():
        """Test SVD decomposition on a matrix."""
        # Create a random matrix
        tensor = torch.randn(12, 8, device=device)
        
        # Change decomposition method for this test
        svd_decomposer = TensorDecomposer(
            decomposition_method="matrix_svd",
            rank_ratio=0.5,
            device=device
        )
        
        # Decompose the tensor
        decomposed_data, metadata = svd_decomposer.decompose_tensor(tensor, "test_svd")
        
        # Check that decomposition was successful
        assert_false(metadata.get("decomposition_failed"))
        assert_in("actual_compression_ratio", metadata)
        
        # Recompose the tensor
        recomposed_tensor = svd_decomposer.recompose_tensor(decomposed_data, metadata)
        
        # Check that shapes match
        assert_equal(recomposed_tensor.shape, tensor.shape)
        
        # Check that the recomposed tensor is approximately equal to the original
        reconstruction_error = torch.mean((tensor - recomposed_tensor) ** 2)
        assert_less(reconstruction_error.item(), 1.0)  # Reasonable threshold

    def different_rank_ratios(self)():
        """Test decomposition with different rank ratios."""
        tensor = torch.randn(10, 8, device=device)
        
        # Test with different rank ratios
        for rank_ratio in [0.1, 0.3, 0.7, 0.9]:
            decomposer = TensorDecomposer(
                decomposition_method="matrix_svd",
                rank_ratio=rank_ratio,
                device=device
            )
            
            decomposed_data, metadata = decomposer.decompose_tensor(tensor, f"test_ratio_{rank_ratio}")
            
            # Check that decomposition was successful
            assert_false(metadata.get("decomposition_failed"))
            assert_in("actual_compression_ratio", metadata)

    def decomposition_statistics(self)():
        """Test decomposition statistics tracking."""
        tensor = torch.randn(10, 8, device=device)
        
        # Decompose the tensor
        _, metadata = decomposer.decompose_tensor(tensor, "stats_test")
        
        # Check that statistics are properly recorded
        stats = decomposer.get_decomposition_stats("stats_test")
        assert_in("original_size", stats)
        assert_in("decomposed_size", stats)
        assert_in("compression_ratio", stats)
        assert_in("saved_bytes", stats)
        
        # Check that saved bytes calculation is reasonable
        assertGreaterEqual(stats["saved_bytes"], 0)

    def invalid_decomposition_method(self)():
        """Test handling of invalid decomposition method."""
        invalid_decomposer = TensorDecomposer(
            decomposition_method="invalid_method",
            rank_ratio=0.5,
            device=device
        )
        
        tensor = torch.randn(10, 8, device=device)
        
        # This should fail gracefully and return the original tensor
        decomposed_data, metadata = invalid_decomposer.decompose_tensor(tensor, "invalid_test")
        
        # Check that failure is properly handled
        assert_true(metadata.get("decomposition_failed"))
        assert_in("error", metadata)

# TestAdaptiveTensorDecomposer

    """Test cases for the AdaptiveTensorDecomposer class."""

    def setup_helper():
        """Set up test fixtures."""
        device = "cpu"  # Use CPU for testing to ensure compatibility
        adaptive_decomposer = AdaptiveTensorDecomposer(
            decomposition_method="cp_decomposition",
            base_rank_ratio=0.5,
            device=device
        )

    def adaptive_rank_adjustment(self)():
        """Test adaptive rank ratio adjustment."""
        tensor = torch.randn(10, 8, device=device)
        
        # Test with different accuracy estimates
        for accuracy_estimate in [0.8, 0.9, 0.95, 0.98]:
            # Decompose with accuracy estimate
            _, metadata = adaptive_decomposer.decompose_tensor(
                tensor, 
                "adaptive_test", 
                accuracy_estimate=accuracy_estimate
            )
            
            # Check that decomposition was successful
            assert_false(metadata.get("decomposition_failed"))
            assert_in("actual_compression_ratio", metadata)

    def memory_usage_tracking(self)():
        """Test memory usage tracking."""
        # Test getting current memory usage
        memory_usage = adaptive_decomposer.get_current_memory_usage()
        
        # Memory usage should be between 0 and 1
        assertGreaterEqual(memory_usage, 0.0)
        assertLessEqual(memory_usage, 1.0)

# TestGlobalTensorDecomposer

    """Test cases for the global tensor decomposer functions."""

    def get_tensor_decomposer(self)():
        """Test getting the global tensor decomposer instance."""
        decomposer1 = get_tensor_decomposer()
        decomposer2 = get_tensor_decomposer()
        
        # Both should return the same instance
        assertIs(decomposer1, decomposer2)
        
        # Should be an instance of AdaptiveTensorDecomposer
        assert_is_instance(decomposer1, AdaptiveTensorDecomposer)

# TestModelWeightDecomposition

    """Test cases for model weight decomposition functions."""

    def setup_helper():
        """Set up test fixtures."""
        device = "cpu"  # Use CPU for testing to ensure compatibility
        
        # Create a simple test model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                linear1 = nn.Linear(10, 8)
                linear2 = nn.Linear(8, 4)
                conv = nn.Conv2d(3, 6, 3)
                
            def forward(self, x):
                x = torch.relu(linear1(x))
                x = torch.relu(linear2(x))
                return x
        
        model = SimpleModel()
        model.to(device)

    def decompose_model_weights(self)():
        """Test decomposing model weights."""
        # Decompose model weights
        decomposed_model, metadata = decompose_model_weights(
            model,
            rank_ratio=0.5,
            decomposition_method="matrix_svd",
            device=device
        )
        
        # Check that the function returns the model and metadata
        assert_is_not_none(decomposed_model)
        assert_is_instance(metadata)
        
        # Check that all named parameters are in the metadata
        for name, _ in model.named_parameters():
            assert_in(name, metadata)

    def recompose_model_weights(self)():
        """Test recomposing model weights."""
        # First decompose model weights
        _, metadata = decompose_model_weights(
            model,
            rank_ratio=0.5,
            decomposition_method="matrix_svd",
            device=device
        )
        
        # Then recompose (this is a placeholder function in our implementation)
        recomposed_model = recompose_model_weights(model, metadata)
        
        # Check that the function returns a model
        assert_is_not_none(recomposed_model)

if __name__ == "__main__":
    run_tests(test_functions)