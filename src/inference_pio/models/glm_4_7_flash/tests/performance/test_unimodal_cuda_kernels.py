"""
Test suite for unimodal CUDA kernels.

This module contains comprehensive tests for the unimodal CUDA kernels
implemented for language models like GLM-4-7, Qwen3-4b-instruct-2507, and Qwen3-coder-30b.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from src.inference_pio.common.unimodal_cuda_kernels import (
    UnimodalAttentionKernel,
    UnimodalMLPKernel,
    UnimodalLayerNormKernel,
    UnimodalRMSNormKernel,
    UnimodalHardwareOptimizer,
    create_unimodal_cuda_kernels,
    apply_unimodal_cuda_optimizations_to_model,
    get_unimodal_cuda_optimization_report
)

# TestUnimodalAttentionKernel

    """Test cases for UnimodalAttentionKernel."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        batch_size = 2
        seq_len = 16
        d_model = 512
        nhead = 8
        
        attention_kernel = UnimodalAttentionKernel(
            d_model=d_model,
            nhead=nhead,
            dropout=0.1,
            use_flash_attention=True,
            causal=True
        )

    def forward_pass(self)():
        """Test the forward pass of the attention kernel."""
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, attn_weights = attention_kernel(query, key, value)
        
        # Check output shape
        assert_equal(output.shape, (batch_size))
        
        # Check attention weights shape
        assert_equal(attn_weights.shape, (batch_size))
        
        # Check that output is finite
        assert_true(torch.all(torch.isfinite(output)))

    def causal_masking(self)():
        """Test that causal masking works correctly."""
        query = torch.randn(batch_size)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, attn_weights = attention_kernel(query, key, value)
        
        # Check that upper triangular part (excluding diagonal) is masked
        # In causal attention, positions should not attend to future positions
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        masked_weights = attn_weights[:, :, mask]
        
        # The masked weights should be very small/negative (approaching -inf before softmax)
        # Since softmax makes them close to 0, we check that they're small
        assert_true(torch.all(masked_weights < 1e-3))

    def attention_with_mask(self)():
        """Test attention with custom attention mask."""
        query = torch.randn(batch_size)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)

        # Create a simple attention mask - need to match the expected format
        # The mask should be broadcastable to (batch, nhead, seq_len, seq_len)
        attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        attention_mask[0, :, :, 8:] = float('-inf')  # Mask out second half of first batch

        output, attn_weights = attention_kernel(
            query, key, value, attention_mask=attention_mask
        )

        assert_equal(output.shape, (batch_size))

# TestUnimodalMLPKernel

    """Test cases for UnimodalMLPKernel."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        batch_size = 2
        seq_len = 16
        d_model = 512
        intermediate_size = 2048
        
        mlp_kernel = UnimodalMLPKernel(
            d_model=d_model,
            intermediate_size=intermediate_size,
            activation="silu",
            use_swiglu=True,
            dropout=0.1
        )

    def forward_pass(self)():
        """Test the forward pass of the MLP kernel."""
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = mlp_kernel(x)
        
        # Check output shape
        assert_equal(output.shape, (batch_size))
        
        # Check that output is finite
        assert_true(torch.all(torch.isfinite(output)))

    def different_activations(self)():
        """Test MLP with different activation functions."""
        x = torch.randn(batch_size)
        
        # Test with different activations
        activations = ["relu", "gelu", "silu", "tanh", "sigmoid"]
        
        for activation in activations:
            mlp = UnimodalMLPKernel(
                d_model=d_model,
                intermediate_size=intermediate_size,
                activation=activation,
                use_swiglu=False
            )
            
            output = mlp(x)
            assert_equal(output.shape, (batch_size))
            assert_true(torch.all(torch.isfinite(output)))

    def swiglu_vs_standard(self)():
        """Test both SwiGLU and standard FFN modes."""
        x = torch.randn(batch_size)
        
        # Test SwiGLU mode
        swiglu_mlp = UnimodalMLPKernel(
            d_model=d_model,
            intermediate_size=intermediate_size,
            use_swiglu=True
        )
        swiglu_output = swiglu_mlp(x)
        
        # Test standard FFN mode
        standard_mlp = UnimodalMLPKernel(
            d_model=d_model,
            intermediate_size=intermediate_size,
            use_swiglu=False
        )
        standard_output = standard_mlp(x)
        
        assert_equal(swiglu_output.shape, (batch_size))
        assert_equal(standard_output.shape, (batch_size))

# TestUnimodalNormalizationKernels

    """Test cases for normalization kernels."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        batch_size = 2
        seq_len = 16
        d_model = 512

    def layernorm_kernel(self)():
        """Test UnimodalLayerNormKernel."""
        x = torch.randn(batch_size, seq_len, d_model)

        layernorm = UnimodalLayerNormKernel(
            normalized_shape=d_model,
            eps=1e-5
        )

        output = layernorm(x)

        assert_equal(output.shape, (batch_size))
        assert_true(torch.all(torch.isfinite(output)))

        # Check that normalization approximately preserves mean and std
        # Use a more realistic tolerance for the standard deviation check
        mean = output.mean(dim=-1)
        std = output.std(dim=-1, keepdim=True)

        assert_true(torch.allclose(mean), atol=1e-5))
        # Allow some tolerance for std since it's computed per sequence position
        assert_true(torch.allclose(std), atol=0.1))

    def rmsnorm_kernel(self)():
        """Test UnimodalRMSNormKernel."""
        x = torch.randn(batch_size, seq_len, d_model)
        
        rmsnorm = UnimodalRMSNormKernel(
            normalized_shape=d_model,
            eps=1e-5
        )
        
        output = rmsnorm(x)
        
        assert_equal(output.shape, (batch_size))
        assert_true(torch.all(torch.isfinite(output)))

    def normalization_consistency(self)():
        """Test that normalization produces consistent results."""
        x = torch.randn(batch_size)
        
        layernorm = UnimodalLayerNormKernel(normalized_shape=d_model)
        rmsnorm = UnimodalRMSNormKernel(normalized_shape=d_model)
        
        ln_output = layernorm(x)
        rn_output = rmsnorm(x)
        
        # Both should have finite outputs
        assert_true(torch.all(torch.isfinite(ln_output)))
        assertTrue(torch.all(torch.isfinite(rn_output)))

# TestUnimodalHardwareOptimizer

    """Test cases for UnimodalHardwareOptimizer."""

    def hardware_detection(self)():
        """Test hardware capability detection."""
        hw_optimizer = UnimodalHardwareOptimizer()
        
        report = hw_optimizer.get_optimization_report()
        
        # Check that report has expected keys
        assert_in("compute_capability")
        assertIn("tensor_cores_supported", report)
        assert_in("optimization_level", report)
        assert_in("recommended_kernels", report)
        
        # Check that compute capability is a tuple of two integers
        assert_is_instance(report["compute_capability"], tuple)
        assert_equal(len(report["compute_capability"]), 2)
        assert_true(all(isinstance(cap) for cap in report["compute_capability"]))
        
        # Check that tensor cores support is boolean
        assert_is_instance(report["tensor_cores_supported"], bool)
        
        # Check that optimization level is a string
        assert_is_instance(report["optimization_level"], str)
        
        # Check that recommended kernels is a list
        assert_is_instance(report["recommended_kernels"], list)

# TestUnimodalCUDAUtilities

    """Test cases for utility functions."""

    def create_unimodal_cuda_kernels(self)():
        """Test creation of unimodal CUDA kernels."""
        d_model = 512
        nhead = 8
        intermediate_size = 2048
        
        kernels = create_unimodal_cuda_kernels(
            d_model=d_model,
            nhead=nhead,
            intermediate_size=intermediate_size,
            use_flash_attention=True,
            use_swiglu=True
        )
        
        # Check that all expected kernels are created
        expected_keys = ["attention", "mlp", "layernorm", "rmsnorm"]
        for key in expected_keys:
            assert_in(key, kernels)
            assert_is_not_none(kernels[key])

    def get_optimization_report(self)():
        """Test getting optimization report."""
        # Create a simple model for testing
        model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512),
            num_layers=2
        )
        
        report = get_unimodal_cuda_optimization_report(
            model=model,
            d_model=512,
            nhead=8,
            intermediate_size=2048,
            model_type="test"
        )
        
        # Check that report has expected structure
        assert_in("model_type", report)
        assert_in("hardware_info", report)
        assert_in("modules_identified_for_optimization", report)
        assert_in("optimization_config", report)
        
        assert_equal(report["model_type"], "test")

# TestModelOptimizationApplication

    """Test cases for applying optimizations to models."""

    def apply_unimodal_cuda_optimizations_to_model(self)():
        """Test applying unimodal CUDA optimizations to a model."""
        # Create a simple test model
        class SimpleTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                embedding = nn.Embedding(1000, 512)
                attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
                layernorm = nn.LayerNorm(512)
                mlp = nn.Sequential(
                    nn.Linear(512, 2048),
                    nn.GELU(),
                    nn.Linear(2048, 512)
                )
                
            def forward(self, x):
                x = embedding(x)
                attn_out, _ = attention(x, x, x)
                x = x + attn_out
                x = layernorm(x)
                x = x + mlp(x)
                return x
        
        model = SimpleTestModel()
        
        # Apply unimodal CUDA optimizations
        optimized_model = apply_unimodal_cuda_optimizations_to_model(
            model=model,
            d_model=512,
            nhead=8,
            intermediate_size=2048,
            model_type="test"
        )
        
        # The optimized model should still be functional
        input_ids = torch.randint(0, 1000, (2, 16))
        output = optimized_model(input_ids)
        
        assert_equal(output.shape, (2))
        assert_true(torch.all(torch.isfinite(output)))

if __name__ == "__main__":
    # Run the tests
    run_tests(test_functions)