"""
Comprehensive Test Suite for CUDA Kernels

This module contains comprehensive tests for all CUDA kernels implemented for the various models.
"""

import unittest
import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.common.cuda_kernels.kernel_framework import (
    BaseCUDAKernel,
    AttentionKernel,
    LinearProjectionKernel,
    ActivationKernel,
    KVCacheKernel,
    NormalizationKernel,
    MLPLayerKernel,
    create_standardized_cuda_kernels,
)

from src.models.specialized.glm_4_7_flash.cuda_kernels.custom_kernels import (
    GLM47FlashAttentionKernel,
    GLM47FlashMLPKernel,
    GLM47FlashRMSNormKernel,
    GLM47FlashRotaryEmbedding,
    GLM47FlashLinearKernel,
    create_glm47_flash_cuda_kernels,
)

from src.models.language.qwen3_4b_instruct_2507.cuda_kernels.custom_kernels import (
    Qwen34BInstructAttentionKernel,
    Qwen34BInstructMLPKernel,
    Qwen34BInstructRMSNormKernel,
    Qwen34BInstructRotaryEmbedding,
    Qwen34BInstructLinearKernel,
    create_qwen3_4b_instruct_cuda_kernels,
)

from src.models.language.qwen3_0_6b.cuda_kernels.custom_kernels import (
    Qwen306BAttentionKernel,
    Qwen306BMLPKernel,
    Qwen306BRMSNormKernel,
    Qwen306BRotaryEmbedding,
    Qwen306BLinearKernel,
    create_qwen3_06b_cuda_kernels,
)

from src.models.coding.qwen3_coder_30b.cuda_kernels.custom_kernels import (
    Qwen3Coder30BAttentionKernel,
    Qwen3Coder30BMLPKernel,
    Qwen3Coder30BRMSNormKernel,
    Qwen3Coder30BRotaryEmbedding,
    Qwen3Coder30BLinearKernel,
    create_qwen3_coder_30b_cuda_kernels,
)

from src.models.coding.qwen3_coder_next.cuda_kernels.custom_kernels import (
    Qwen3CoderNextAttentionKernel,
    Qwen3CoderNextMLPKernel,
    Qwen3CoderNextRMSNormKernel,
    Qwen3CoderNextRotaryEmbedding,
    Qwen3CoderNextLinearKernel,
    create_qwen3_coder_next_cuda_kernels,
)


class TestStandardizedKernels(unittest.TestCase):
    """Test cases for standardized CUDA kernels."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 512
        self.nhead = 8
        self.intermediate_size = 2048
        
    def test_attention_kernel(self):
        """Test the standardized attention kernel."""
        kernel = AttentionKernel(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=0.1,
            use_flash_attention=True,
        )
        
        # Create random inputs
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output, attn_weights = kernel(query, key, value, need_weights=True)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check attention weights shape
        self.assertEqual(attn_weights.shape, (self.batch_size, self.nhead, self.seq_len, self.seq_len))
        
    def test_linear_projection_kernel(self):
        """Test the standardized linear projection kernel."""
        kernel = LinearProjectionKernel(
            in_features=self.d_model,
            out_features=self.d_model,
            bias=True,
        )
        
        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = kernel(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_activation_kernel(self):
        """Test the standardized activation kernel."""
        # Test SiLU activation
        silu_kernel = ActivationKernel(activation_type="silu")
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = silu_kernel(x)
        self.assertEqual(output.shape, x.shape)
        
        # Test GELU activation
        gelu_kernel = ActivationKernel(activation_type="gelu")
        output = gelu_kernel(x)
        self.assertEqual(output.shape, x.shape)
        
    def test_normalization_kernel(self):
        """Test the standardized normalization kernel."""
        # Test RMSNorm
        rmsnorm_kernel = NormalizationKernel(
            normalized_shape=self.d_model,
            norm_type="rms",
            eps=1e-6,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = rmsnorm_kernel(x)
        self.assertEqual(output.shape, x.shape)
        
        # Test LayerNorm
        layernorm_kernel = NormalizationKernel(
            normalized_shape=self.d_model,
            norm_type="layer",
            eps=1e-6,
        )
        output = layernorm_kernel(x)
        self.assertEqual(output.shape, x.shape)
        
    def test_mlp_layer_kernel(self):
        """Test the standardized MLP layer kernel."""
        # Test with SwiGLU
        mlp_kernel = MLPLayerKernel(
            d_model=self.d_model,
            intermediate_size=self.intermediate_size,
            use_swiglu=True,
            activation_type="silu",
            dropout=0.1,
        )
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = mlp_kernel(x)
        self.assertEqual(output.shape, x.shape)
        
        # Test without SwiGLU
        mlp_kernel = MLPLayerKernel(
            d_model=self.d_model,
            intermediate_size=self.intermediate_size,
            use_swiglu=False,
            activation_type="gelu",
            dropout=0.1,
        )
        output = mlp_kernel(x)
        self.assertEqual(output.shape, x.shape)
        
    def test_create_standardized_cuda_kernels(self):
        """Test the factory function for standardized kernels."""
        kernels = create_standardized_cuda_kernels(
            d_model=self.d_model,
            nhead=self.nhead,
            intermediate_size=self.intermediate_size,
            max_batch_size=self.batch_size,
            max_seq_len=self.seq_len,
            use_flash_attention=True,
            use_swiglu=True,
            norm_type="rms",
            activation_type="silu",
        )
        
        # Check that all expected kernels are created
        expected_kernels = ["attention", "mlp", "norm", "kv_cache"]
        for kernel_name in expected_kernels:
            self.assertIn(kernel_name, kernels)
            self.assertIsInstance(kernels[kernel_name], BaseCUDAKernel)


class TestGLM47FlashKernels(unittest.TestCase):
    """Test cases for GLM-4.7-Flash CUDA kernels."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 2048
        self.nhead = 32
        self.intermediate_size = 5504
        
    def test_glm47_flash_attention_kernel(self):
        """Test the GLM-4.7-Flash attention kernel."""
        kernel = GLM47FlashAttentionKernel(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=0.1,
            use_flash_attention=True,
            use_rotary_embeddings=True,
        )
        
        # Create random inputs
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        position_ids = torch.arange(self.seq_len).expand(self.batch_size, -1)
        
        # Forward pass
        output, attn_weights = kernel(query, key, value, position_ids=position_ids, need_weights=True)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check attention weights shape
        self.assertEqual(attn_weights.shape, (self.batch_size, self.nhead, self.seq_len, self.seq_len))
        
    def test_glm47_flash_mlp_kernel(self):
        """Test the GLM-4.7-Flash MLP kernel."""
        kernel = GLM47FlashMLPKernel(
            d_model=self.d_model,
            intermediate_size=self.intermediate_size,
            activation_type="gelu",
            dropout=0.1,
        )
        
        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = kernel(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_glm47_flash_rmsnorm_kernel(self):
        """Test the GLM-4.7-Flash RMSNorm kernel."""
        kernel = GLM47FlashRMSNormKernel(
            normalized_shape=self.d_model,
            eps=1e-5,
        )
        
        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = kernel(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_glm47_flash_rotary_embedding(self):
        """Test the GLM-4.7-Flash rotary embedding."""
        kernel = GLM47FlashRotaryEmbedding(dim=self.d_model // self.nhead)
        
        # Create random inputs
        q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        k = torch.randn(self.batch_size, self.seq_len, self.d_model)
        position_ids = torch.arange(self.seq_len).expand(self.batch_size, -1)
        
        # Forward pass
        q_rotated, k_rotated = kernel(q, k, position_ids)
        
        # Check output shapes
        self.assertEqual(q_rotated.shape, q.shape)
        self.assertEqual(k_rotated.shape, k.shape)
        
    def test_create_glm47_flash_cuda_kernels(self):
        """Test the factory function for GLM-4.7-Flash kernels."""
        kernels = create_glm47_flash_cuda_kernels(
            d_model=self.d_model,
            nhead=self.nhead,
            intermediate_size=self.intermediate_size,
            use_flash_attention=True,
            use_rotary_embeddings=True,
        )
        
        # Check that all expected kernels are created
        expected_kernels = ["attention", "mlp", "rmsnorm", "linear"]
        for kernel_name in expected_kernels:
            self.assertIn(kernel_name, kernels)
            self.assertIsInstance(kernels[kernel_name], BaseCUDAKernel)


class TestQwen34BInstructKernels(unittest.TestCase):
    """Test cases for Qwen3-4B-Instruct-2507 CUDA kernels."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 2048
        self.nhead = 16
        self.intermediate_size = 5504
        
    def test_qwen3_4b_instruct_attention_kernel(self):
        """Test the Qwen3-4B-Instruct-2507 attention kernel."""
        kernel = Qwen34BInstructAttentionKernel(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=0.1,
            use_flash_attention=True,
            use_rotary_embeddings=True,
            sliding_window_size=4096,
        )
        
        # Create random inputs
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        position_ids = torch.arange(self.seq_len).expand(self.batch_size, -1)
        
        # Forward pass
        output, attn_weights = kernel(query, key, value, position_ids=position_ids, need_weights=True)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check attention weights shape
        self.assertEqual(attn_weights.shape, (self.batch_size, self.nhead, self.seq_len, self.seq_len))
        
    def test_qwen3_4b_instruct_mlp_kernel(self):
        """Test the Qwen3-4B-Instruct-2507 MLP kernel."""
        kernel = Qwen34BInstructMLPKernel(
            d_model=self.d_model,
            intermediate_size=self.intermediate_size,
            activation_type="silu",
            dropout=0.1,
        )
        
        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = kernel(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_qwen3_4b_instruct_rmsnorm_kernel(self):
        """Test the Qwen3-4B-Instruct-2507 RMSNorm kernel."""
        kernel = Qwen34BInstructRMSNormKernel(
            normalized_shape=self.d_model,
            eps=1e-6,
        )
        
        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = kernel(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_qwen3_4b_instruct_rotary_embedding(self):
        """Test the Qwen3-4B-Instruct-2507 rotary embedding."""
        kernel = Qwen34BInstructRotaryEmbedding(dim=self.d_model // self.nhead)
        
        # Create random inputs
        q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        k = torch.randn(self.batch_size, self.seq_len, self.d_model)
        position_ids = torch.arange(self.seq_len).expand(self.batch_size, -1)
        
        # Forward pass
        q_rotated, k_rotated = kernel(q, k, position_ids)
        
        # Check output shapes
        self.assertEqual(q_rotated.shape, q.shape)
        self.assertEqual(k_rotated.shape, k.shape)
        
    def test_create_qwen3_4b_instruct_cuda_kernels(self):
        """Test the factory function for Qwen3-4B-Instruct-2507 kernels."""
        kernels = create_qwen3_4b_instruct_cuda_kernels(
            d_model=self.d_model,
            nhead=self.nhead,
            intermediate_size=self.intermediate_size,
            use_flash_attention=True,
            use_rotary_embeddings=True,
            sliding_window_size=4096,
        )
        
        # Check that all expected kernels are created
        expected_kernels = ["attention", "mlp", "rmsnorm", "linear"]
        for kernel_name in expected_kernels:
            self.assertIn(kernel_name, kernels)
            self.assertIsInstance(kernels[kernel_name], BaseCUDAKernel)


class TestQwen306BKernels(unittest.TestCase):
    """Test cases for Qwen3-0.6B CUDA kernels."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 1536
        self.nhead = 12
        self.intermediate_size = 4096
        
    def test_qwen3_06b_attention_kernel(self):
        """Test the Qwen3-0.6B attention kernel."""
        kernel = Qwen306BAttentionKernel(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=0.1,
            use_flash_attention=True,
            use_rotary_embeddings=True,
        )
        
        # Create random inputs
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        position_ids = torch.arange(self.seq_len).expand(self.batch_size, -1)
        
        # Forward pass
        output, attn_weights = kernel(query, key, value, position_ids=position_ids, need_weights=True)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check attention weights shape
        self.assertEqual(attn_weights.shape, (self.batch_size, self.nhead, self.seq_len, self.seq_len))
        
    def test_qwen3_06b_mlp_kernel(self):
        """Test the Qwen3-0.6B MLP kernel."""
        kernel = Qwen306BMLPKernel(
            d_model=self.d_model,
            intermediate_size=self.intermediate_size,
            activation_type="silu",
            dropout=0.1,
        )
        
        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = kernel(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_qwen3_06b_rmsnorm_kernel(self):
        """Test the Qwen3-0.6B RMSNorm kernel."""
        kernel = Qwen306BRMSNormKernel(
            normalized_shape=self.d_model,
            eps=1e-6,
        )
        
        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = kernel(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_qwen3_06b_rotary_embedding(self):
        """Test the Qwen3-0.6B rotary embedding."""
        kernel = Qwen306BRotaryEmbedding(dim=self.d_model // self.nhead)
        
        # Create random inputs
        q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        k = torch.randn(self.batch_size, self.seq_len, self.d_model)
        position_ids = torch.arange(self.seq_len).expand(self.batch_size, -1)
        
        # Forward pass
        q_rotated, k_rotated = kernel(q, k, position_ids)
        
        # Check output shapes
        self.assertEqual(q_rotated.shape, q.shape)
        self.assertEqual(k_rotated.shape, k.shape)
        
    def test_create_qwen3_06b_cuda_kernels(self):
        """Test the factory function for Qwen3-0.6B kernels."""
        kernels = create_qwen3_06b_cuda_kernels(
            d_model=self.d_model,
            nhead=self.nhead,
            intermediate_size=self.intermediate_size,
            use_flash_attention=True,
            use_rotary_embeddings=True,
        )
        
        # Check that all expected kernels are created
        expected_kernels = ["attention", "mlp", "rmsnorm", "linear"]
        for kernel_name in expected_kernels:
            self.assertIn(kernel_name, kernels)
            self.assertIsInstance(kernels[kernel_name], BaseCUDAKernel)


class TestQwen3Coder30BKernels(unittest.TestCase):
    """Test cases for Qwen3-Coder-30B CUDA kernels."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 4096
        self.nhead = 32
        self.intermediate_size = 11008
        
    def test_qwen3_coder_30b_attention_kernel(self):
        """Test the Qwen3-Coder-30B attention kernel."""
        kernel = Qwen3Coder30BAttentionKernel(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=0.1,
            use_flash_attention=True,
            use_rotary_embeddings=True,
            use_sliding_window=True,
            sliding_window_size=4096,
        )
        
        # Create random inputs
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        position_ids = torch.arange(self.seq_len).expand(self.batch_size, -1)
        
        # Forward pass
        output, attn_weights = kernel(query, key, value, position_ids=position_ids, need_weights=True)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check attention weights shape
        self.assertEqual(attn_weights.shape, (self.batch_size, self.nhead, self.seq_len, self.seq_len))
        
    def test_qwen3_coder_30b_mlp_kernel(self):
        """Test the Qwen3-Coder-30B MLP kernel."""
        kernel = Qwen3Coder30BMLPKernel(
            d_model=self.d_model,
            intermediate_size=self.intermediate_size,
            activation_type="silu",
            dropout=0.1,
        )
        
        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = kernel(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_qwen3_coder_30b_rmsnorm_kernel(self):
        """Test the Qwen3-Coder-30B RMSNorm kernel."""
        kernel = Qwen3Coder30BRMSNormKernel(
            normalized_shape=self.d_model,
            eps=1e-5,
        )
        
        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = kernel(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_qwen3_coder_30b_rotary_embedding(self):
        """Test the Qwen3-Coder-30B rotary embedding."""
        kernel = Qwen3Coder30BRotaryEmbedding(dim=self.d_model // self.nhead)
        
        # Create random inputs
        q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        k = torch.randn(self.batch_size, self.seq_len, self.d_model)
        position_ids = torch.arange(self.seq_len).expand(self.batch_size, -1)
        
        # Forward pass
        q_rotated, k_rotated = kernel(q, k, position_ids)
        
        # Check output shapes
        self.assertEqual(q_rotated.shape, q.shape)
        self.assertEqual(k_rotated.shape, k.shape)
        
    def test_create_qwen3_coder_30b_cuda_kernels(self):
        """Test the factory function for Qwen3-Coder-30B kernels."""
        kernels = create_qwen3_coder_30b_cuda_kernels(
            d_model=self.d_model,
            nhead=self.nhead,
            intermediate_size=self.intermediate_size,
            use_flash_attention=True,
            use_rotary_embeddings=True,
            use_sliding_window=True,
            sliding_window_size=4096,
        )
        
        # Check that all expected kernels are created
        expected_kernels = ["attention", "mlp", "rmsnorm", "linear"]
        for kernel_name in expected_kernels:
            self.assertIn(kernel_name, kernels)
            self.assertIsInstance(kernels[kernel_name], BaseCUDAKernel)


class TestQwen3CoderNextKernels(unittest.TestCase):
    """Test cases for Qwen3-Coder-Next CUDA kernels."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 16
        self.d_model = 8192
        self.nhead = 64
        self.intermediate_size = 22016
        
    def test_qwen3_coder_next_attention_kernel(self):
        """Test the Qwen3-Coder-Next attention kernel."""
        kernel = Qwen3CoderNextAttentionKernel(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=0.1,
            use_flash_attention=True,
            use_rotary_embeddings=True,
            use_sliding_window=True,
            sliding_window_size=4096,
            use_moe=True,
            num_experts=8,
            top_k=2,
        )
        
        # Create random inputs
        query = torch.randn(self.batch_size, self.seq_len, self.d_model)
        key = torch.randn(self.batch_size, self.seq_len, self.d_model)
        value = torch.randn(self.batch_size, self.seq_len, self.d_model)
        position_ids = torch.arange(self.seq_len).expand(self.batch_size, -1)
        
        # Forward pass
        output, attn_weights = kernel(query, key, value, position_ids=position_ids, need_weights=True)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check attention weights shape
        self.assertEqual(attn_weights.shape, (self.batch_size, self.nhead, self.seq_len, self.seq_len))
        
    def test_qwen3_coder_next_mlp_kernel(self):
        """Test the Qwen3-Coder-Next MLP kernel."""
        kernel = Qwen3CoderNextMLPKernel(
            d_model=self.d_model,
            intermediate_size=self.intermediate_size,
            activation_type="silu",
            dropout=0.1,
            use_moe=True,
            num_experts=8,
            top_k=2,
        )
        
        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = kernel(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_qwen3_coder_next_rmsnorm_kernel(self):
        """Test the Qwen3-Coder-Next RMSNorm kernel."""
        kernel = Qwen3CoderNextRMSNormKernel(
            normalized_shape=self.d_model,
            eps=1e-5,
        )
        
        # Create random input
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Forward pass
        output = kernel(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
    def test_qwen3_coder_next_rotary_embedding(self):
        """Test the Qwen3-Coder-Next rotary embedding."""
        kernel = Qwen3CoderNextRotaryEmbedding(dim=self.d_model // self.nhead)
        
        # Create random inputs
        q = torch.randn(self.batch_size, self.seq_len, self.d_model)
        k = torch.randn(self.batch_size, self.seq_len, self.d_model)
        position_ids = torch.arange(self.seq_len).expand(self.batch_size, -1)
        
        # Forward pass
        q_rotated, k_rotated = kernel(q, k, position_ids)
        
        # Check output shapes
        self.assertEqual(q_rotated.shape, q.shape)
        self.assertEqual(k_rotated.shape, k.shape)
        
    def test_create_qwen3_coder_next_cuda_kernels(self):
        """Test the factory function for Qwen3-Coder-Next kernels."""
        kernels = create_qwen3_coder_next_cuda_kernels(
            d_model=self.d_model,
            nhead=self.nhead,
            intermediate_size=self.intermediate_size,
            use_flash_attention=True,
            use_rotary_embeddings=True,
            use_sliding_window=True,
            sliding_window_size=4096,
            use_moe=True,
            num_experts=8,
            top_k=2,
        )
        
        # Check that all expected kernels are created
        expected_kernels = ["attention", "mlp", "rmsnorm", "linear"]
        for kernel_name in expected_kernels:
            self.assertIn(kernel_name, kernels)
            self.assertIsInstance(kernels[kernel_name], BaseCUDAKernel)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)