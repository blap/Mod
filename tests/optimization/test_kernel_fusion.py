"""
Test suite for kernel fusion techniques in Qwen3-VL model
"""
import torch
import torch.nn as nn
import pytest
from typing import Tuple
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.qwen3_vl.optimization.kernel_fusion import (
    FusedLayerNormLinear,
    FusedAttentionSoftmax,
    FusedMLPBlock,
    FusedQKVMatmul,
    FusedResidualAddLayerNorm,
    FusedDecoderLayer,
    KernelFusionManager,
    apply_kernel_fusion_to_model
)


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.hidden_size = 512
        self.intermediate_size = 2048
        self.num_attention_heads = 8
        self.num_hidden_layers = 2
        self.layer_norm_eps = 1e-5
        self.vocab_size = 32000
        self.max_position_embeddings = 512
        self.rope_theta = 1000000
        self.use_cache = True


def test_fused_layer_norm_linear():
    """Test the fused LayerNorm + Linear module"""
    config = MockConfig()
    fused_module = FusedLayerNormLinear(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        eps=config.layer_norm_eps
    )
    
    # Create test input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output = fused_module(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, config.intermediate_size), \
        f"Expected output shape {(batch_size, seq_len, config.intermediate_size)}, got {output.shape}"
    
    print("✓ FusedLayerNormLinear test passed")


def test_fused_attention_softmax():
    """Test the fused Attention + Softmax module"""
    config = MockConfig()
    fused_attention = FusedAttentionSoftmax(config)
    
    # Create test inputs
    batch_size, seq_len = 2, 10
    head_dim = config.hidden_size // config.num_attention_heads
    
    query = torch.randn(batch_size, config.num_attention_heads, seq_len, head_dim)
    key = torch.randn(batch_size, config.num_attention_heads, seq_len, head_dim)
    value = torch.randn(batch_size, config.num_attention_heads, seq_len, head_dim)
    
    # Forward pass
    output = fused_attention(query, key, value)
    
    # Check output shape
    assert output.shape == (batch_size, config.num_attention_heads, seq_len, head_dim), \
        f"Expected output shape {(batch_size, config.num_attention_heads, seq_len, head_dim)}, got {output.shape}"
    
    print("✓ FusedAttentionSoftmax test passed")


def test_fused_mlp_block():
    """Test the fused MLP block"""
    config = MockConfig()
    fused_mlp = FusedMLPBlock(config)
    
    # Create test input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    residual = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output = fused_mlp(x, residual)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Expected output shape {(batch_size, seq_len, config.hidden_size)}, got {output.shape}"
    
    print("✓ FusedMLPBlock test passed")


def test_fused_qkv_matmul():
    """Test the fused QKV matmul module"""
    config = MockConfig()
    fused_qkv = FusedQKVMatmul(config)
    
    # Create test input
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    query, key, value = fused_qkv(hidden_states)
    
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Check output shapes
    expected_shape = (batch_size, config.num_attention_heads, seq_len, head_dim)
    assert query.shape == expected_shape, f"Expected query shape {expected_shape}, got {query.shape}"
    assert key.shape == expected_shape, f"Expected key shape {expected_shape}, got {key.shape}"
    assert value.shape == expected_shape, f"Expected value shape {expected_shape}, got {value.shape}"
    
    print("✓ FusedQKVMatmul test passed")


def test_fused_residual_add_layernorm():
    """Test the fused residual addition + layer norm module"""
    config = MockConfig()
    fused_norm = FusedResidualAddLayerNorm(config.hidden_size, config.layer_norm_eps)
    
    # Create test inputs
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    residual = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output = fused_norm(hidden_states, residual)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, config.hidden_size), \
        f"Expected output shape {(batch_size, seq_len, config.hidden_size)}, got {output.shape}"
    
    print("✓ FusedResidualAddLayerNorm test passed")


def test_fused_decoder_layer():
    """Test the fused decoder layer"""
    config = MockConfig()
    layer_idx = 0
    fused_layer = FusedDecoderLayer(config, layer_idx)
    
    # Create test inputs
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass
    output = fused_layer(hidden_states)
    
    # Check output type and shape
    assert isinstance(output, tuple), "Output should be a tuple"
    assert len(output) >= 1, "Output tuple should have at least one element"
    assert output[0].shape == (batch_size, seq_len, config.hidden_size), \
        f"Expected output shape {(batch_size, seq_len, config.hidden_size)}, got {output[0].shape}"
    
    print("✓ FusedDecoderLayer test passed")


def test_kernel_fusion_manager():
    """Test the kernel fusion manager"""
    config = MockConfig()
    
    # Create a simple mock model for testing
    class MockModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([
                nn.Module() for _ in range(config.num_hidden_layers)
            ])
            # Add mock attributes to layers
            for layer in self.language_model.layers:
                layer.self_attn = nn.Module()
                layer.mlp = nn.Module()
                layer.input_layernorm = nn.LayerNorm(config.hidden_size)
                layer.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
    
    model = MockModel(config)
    
    # Apply kernel fusion
    fusion_manager = KernelFusionManager(config)
    fused_model = fusion_manager.fuse_model(model)
    
    # Check that layers were replaced
    for layer in fused_model.language_model.layers:
        assert isinstance(layer, FusedDecoderLayer), \
            "Layers should be replaced with FusedDecoderLayer instances"
    
    # Get fusion stats
    stats = fusion_manager.get_fusion_stats(fused_model)
    assert stats["fused_layers"] == config.num_hidden_layers, \
        f"Expected {config.num_hidden_layers} fused layers, got {stats['fused_layers']}"
    
    print("✓ KernelFusionManager test passed")


def test_apply_kernel_fusion_to_model():
    """Test the apply_kernel_fusion_to_model function"""
    config = MockConfig()
    
    # Create a simple mock model for testing
    class MockModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.language_model = nn.Module()
            self.language_model.layers = nn.ModuleList([
                nn.Module() for _ in range(config.num_hidden_layers)
            ])
            # Add mock attributes to layers
            for layer in self.language_model.layers:
                layer.self_attn = nn.Module()
                layer.mlp = nn.Module()
                layer.input_layernorm = nn.LayerNorm(config.hidden_size)
                layer.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
    
    model = MockModel(config)
    
    # Apply kernel fusion
    fused_model = apply_kernel_fusion_to_model(model, config)
    
    # Check that layers were replaced
    for layer in fused_model.language_model.layers:
        assert isinstance(layer, FusedDecoderLayer), \
            "Layers should be replaced with FusedDecoderLayer instances"
    
    print("✓ apply_kernel_fusion_to_model test passed")


def test_cuda_fallback():
    """Test that CUDA fallback works correctly"""
    config = MockConfig()
    
    # Create a fused module
    fused_module = FusedLayerNormLinear(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        eps=config.layer_norm_eps
    )
    
    # Create test input on CPU
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Forward pass on CPU (should use PyTorch fallback)
    output = fused_module(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, config.intermediate_size), \
        f"Expected output shape {(batch_size, seq_len, config.intermediate_size)}, got {output.shape}"
    
    print("✓ CUDA fallback test passed")


def run_all_tests():
    """Run all tests"""
    print("Running kernel fusion tests...")
    
    test_fused_layer_norm_linear()
    test_fused_attention_softmax()
    test_fused_mlp_block()
    test_fused_qkv_matmul()
    test_fused_residual_add_layernorm()
    test_fused_decoder_layer()
    test_kernel_fusion_manager()
    test_apply_kernel_fusion_to_model()
    test_cuda_fallback()
    
    print("\n✓ All kernel fusion tests passed!")


if __name__ == "__main__":
    run_all_tests()