"""
Integration Tests for Tensor Decomposition System

This module contains integration tests to verify that the tensor decomposition system
works properly across all four models (GLM-4-7, Qwen3-4b-instruct-2507, Qwen3-coder-30b, Qwen3-vl-2b).
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from ..tensor_decomposition import (
    TensorDecomposer,
    AdaptiveTensorDecomposer,
    get_tensor_decomposer,
    decompose_model_weights,
    recompose_model_weights
)

# Mock the model classes to avoid heavy dependencies
class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        linear1 = nn.Linear(10, 8)
        linear2 = nn.Linear(8, 4)
        conv = nn.Conv2d(3, 6, 3)
        
    def forward(self, x):
        x = torch.relu(linear1(x))
        x = torch.relu(linear2(x))
        return x

# TestTensorDecompositionIntegration

    """Integration tests for tensor decomposition system."""

    def setup_helper():
        """Set up test fixtures."""
        device = "cpu"  # Use CPU for testing to ensure compatibility
        mock_model = MockModel()

    def tensor_decomposer_basic_functionality(self)():
        """Test basic functionality of TensorDecomposer."""
        decomposer = TensorDecomposer(
            decomposition_method="matrix_svd",
            rank_ratio=0.5,
            device=device
        )
        
        # Test with a simple tensor
        tensor = torch.randn(10, 8, device=device)
        
        # Decompose and recompose
        decomposed_data, metadata = decomposer.decompose_tensor(tensor, "test_tensor")
        recomposed_tensor = decomposer.recompose_tensor(decomposed_data, metadata)
        
        # Verify shapes match
        assert_equal(recomposed_tensor.shape, tensor.shape)
        
        # Verify that the process completed successfully
        assert_false(metadata.get("decomposition_failed"))

    def adaptive_tensor_decomposer(self)():
        """Test AdaptiveTensorDecomposer functionality."""
        adaptive_decomposer = AdaptiveTensorDecomposer(
            decomposition_method="matrix_svd",
            base_rank_ratio=0.5,
            device=device
        )
        
        # Test with a simple tensor
        tensor = torch.randn(10, 8, device=device)
        
        # Decompose and recompose with accuracy estimate
        decomposed_data, metadata = adaptive_decomposer.decompose_tensor(
            tensor, "adaptive_test", accuracy_estimate=0.95
        )
        recomposed_tensor = adaptive_decomposer.recompose_tensor(decomposed_data, metadata)
        
        # Verify shapes match
        assert_equal(recomposed_tensor.shape, tensor.shape)

    def global_tensor_decomposer_singleton(self)():
        """Test that global tensor decomposer returns singleton instance."""
        decomposer1 = get_tensor_decomposer()
        decomposer2 = get_tensor_decomposer()
        
        # Both should return the same instance
        assertIs(decomposer1, decomposer2)
        
        # Should be an AdaptiveTensorDecomposer instance
        assert_is_instance(decomposer1, AdaptiveTensorDecomposer)

    def decompose_model_weights(self)():
        """Test decompose_model_weights function."""
        model = MockModel()
        
        # Decompose model weights
        decomposed_model, metadata = decompose_model_weights(
            model,
            rank_ratio=0.5,
            decomposition_method="matrix_svd",
            device=device
        )
        
        # Verify that metadata contains entries for all parameters
        param_names = {name for name, _ in model.named_parameters()}
        metadata_keys = set(metadata.keys())
        
        # All parameter names should be in metadata
        assert_equal(param_names, metadata_keys.intersection(param_names))

    def recompose_model_weights(self)():
        """Test recompose_model_weights function."""
        model = MockModel()
        
        # First decompose model weights
        _, metadata = decompose_model_weights(
            model,
            rank_ratio=0.5,
            decomposition_method="matrix_svd",
            device=device
        )
        
        # Then recompose (this is a placeholder in our implementation)
        recomposed_model = recompose_model_weights(model, metadata)
        
        # Verify that the function returns a model
        assert_is_not_none(recomposed_model)

    def different_decomposition_methods(self)():
        """Test different decomposition methods."""
        tensor = torch.randn(12)
        methods = ["matrix_svd", "cp_decomposition"]
        
        for method in methods:
            with subTest(method=method):
                decomposer = TensorDecomposer(
                    decomposition_method=method,
                    rank_ratio=0.5,
                    device=device
                )
                
                # Decompose and recompose
                decomposed_data, metadata = decomposer.decompose_tensor(tensor, f"test_{method}")
                recomposed_tensor = decomposer.recompose_tensor(decomposed_data, metadata)
                
                # Verify shapes match
                assert_equal(recomposed_tensor.shape, tensor.shape)

    def rank_ratio_impact(self)():
        """Test impact of different rank ratios."""
        tensor = torch.randn(10, 8, device=device)
        rank_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for rank_ratio in rank_ratios:
            with subTest(rank_ratio=rank_ratio):
                decomposer = TensorDecomposer(
                    decomposition_method="matrix_svd",
                    rank_ratio=rank_ratio,
                    device=device
                )
                
                # Decompose and recompose
                decomposed_data, metadata = decomposer.decompose_tensor(tensor, f"test_ratio_{rank_ratio}")
                
                # Verify that decomposition was successful
                assert_false(metadata.get("decomposition_failed"))
                assert_in("actual_compression_ratio", metadata)

    def decomposition_statistics(self)():
        """Test decomposition statistics tracking."""
        tensor = torch.randn(10, 8, device=device)
        decomposer = TensorDecomposer(
            decomposition_method="matrix_svd",
            rank_ratio=0.5,
            device=device
        )
        
        # Decompose tensor
        _, metadata = decomposer.decompose_tensor(tensor, "stats_test")
        
        # Check that statistics are properly recorded
        stats = decomposer.get_decomposition_stats("stats_test")
        assert_in("original_size", stats)
        assert_in("decomposed_size", stats)
        assert_in("compression_ratio", stats)
        assert_in("saved_bytes", stats)

    def error_handling(self)():
        """Test error handling in decomposition process."""
        # Create a decomposer with an invalid method
        invalid_decomposer = TensorDecomposer(
            decomposition_method="invalid_method",
            rank_ratio=0.5,
            device=device
        )
        
        tensor = torch.randn(10, 8, device=device)
        
        # This should handle the error gracefully
        decomposed_data, metadata = invalid_decomposer.decompose_tensor(tensor, "error_test")
        
        # Verify that error was handled properly
        assert_true(metadata.get("decomposition_failed"))
        assert_in("error", metadata)

    def memory_usage_tracking(self)():
        """Test memory usage tracking in adaptive decomposer."""
        adaptive_decomposer = AdaptiveTensorDecomposer(device=device)
        
        # Test memory usage tracking
        memory_usage = adaptive_decomposer.get_current_memory_usage()
        
        # Memory usage should be between 0 and 1
        assertGreaterEqual(memory_usage, 0.0)
        assertLessEqual(memory_usage, 1.0)

    def adaptive_rank_adjustment(self)():
        """Test adaptive rank ratio adjustment."""
        adaptive_decomposer = AdaptiveTensorDecomposer(
            base_rank_ratio=0.5,
            device=device
        )
        
        tensor = torch.randn(10, 8, device=device)
        
        # Test with different accuracy estimates
        for accuracy_estimate in [0.8, 0.9, 0.95, 0.98]:
            with subTest(accuracy=accuracy_estimate):
                # Decompose with accuracy estimate
                _, metadata = adaptive_decomposer.decompose_tensor(
                    tensor,
                    f"adaptive_test_{accuracy_estimate}",
                    accuracy_estimate=accuracy_estimate
                )
                
                # Verify that decomposition was successful
                assert_false(metadata.get("decomposition_failed"))
                assert_in("actual_compression_ratio", metadata)

# TestModelSpecificIntegration

    """Test tensor decomposition integration with model classes."""

    def setup_helper():
        """Set up test fixtures."""
        device = "cpu"

    @patch('torch.cuda.is_available', return_value=False)
    @patch('psutil.virtual_memory')
    def mock_model_with_decomposition(self, mock_psutil, mock_cuda)():
        """Test tensor decomposition with a mock model."""
        # Mock memory usage
        mock_memory = MagicMock()
        mock_memory.percent = 50.0
        mock_psutil.return_value = mock_memory
        mock_cuda.return_value = False
        
        # Create a mock model
        model = MockModel()
        
        # Apply decomposition to the model
        decomposed_model, metadata = decompose_model_weights(
            model,
            rank_ratio=0.5,
            decomposition_method="matrix_svd",
            device=device
        )
        
        # Verify that decomposition metadata was created for all parameters
        expected_params = set(name for name, _ in model.named_parameters())
        actual_params = set(metadata.keys())
        
        # Check that all parameters have metadata
        assert_equal(expected_params, actual_params.intersection(expected_params))
        
        # Verify that each parameter's metadata contains required fields
        for param_name in expected_params:
            param_metadata = metadata[param_name]
            assert_in("original_shape", param_metadata)
            assert_in("original_size", param_metadata)
            assert_in("decomposition_method", param_metadata)
            assert_in("actual_compression_ratio", param_metadata)

if __name__ == "__main__":
    run_tests(test_functions)