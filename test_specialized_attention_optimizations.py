"""
Test suite for specialized attention optimizations across all models.
This test verifies that each model correctly incorporates its specialized attention mechanism.
"""

import torch
import unittest
from src.models.specialized.glm_4_7_flash.attention.flash_attention import FlashAttention, FlashAttentionConfig
from src.models.language.qwen3_4b_instruct_2507.attention.grouped_query_attention import GroupedQueryAttention, GroupedQueryAttentionConfig
from src.models.coding.qwen3_coder_30b.attention.multi_query_attention import MultiQueryAttention, MultiQueryAttentionConfig
from src.models.language.qwen3_0_6b.attention.sparse_attention import SparseAttention, SparseAttentionConfig
from src.models.coding.qwen3_coder_next.attention.sliding_window_attention import SlidingWindowAttention, SlidingWindowAttentionConfig


class TestSpecializedAttentionOptimizations(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.batch_size = 2
        self.seq_len = 16
        self.embed_dim = 512
        self.num_heads = 8
        
        # Create sample input tensors
        self.query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        self.value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
    
    def test_flash_attention(self):
        """Test Flash Attention implementation for GLM-4.7-Flash."""
        config = FlashAttentionConfig(
            use_flash_attention=True,
            flash_attention_dropout=0.1,
            flash_num_heads=self.num_heads
        )
        
        attention = FlashAttention(
            embed_dim=self.embed_dim,
            num_heads=config.flash_num_heads,
            dropout=config.flash_attention_dropout
        )
        
        output, attn_weights = attention(self.query, self.key, self.value)
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
        
        # Verify attention weights shape if returned
        if attn_weights is not None:
            self.assertEqual(attn_weights.shape[0], self.batch_size)
            self.assertEqual(attn_weights.shape[1], self.seq_len)
            self.assertEqual(attn_weights.shape[2], self.seq_len)
    
    def test_grouped_query_attention(self):
        """Test Grouped Query Attention implementation for Qwen3-4B-Instruct-2507."""
        config = GroupedQueryAttentionConfig(
            use_grouped_query_attention=True,
            gqa_num_heads=self.num_heads,
            gqa_num_kv_groups=4,  # Group queries
            gqa_attention_dropout=0.1
        )
        
        attention = GroupedQueryAttention(
            embed_dim=self.embed_dim,
            num_heads=config.gqa_num_heads,
            num_kv_groups=config.gqa_num_kv_groups,
            dropout=config.gqa_attention_dropout
        )
        
        output, attn_weights = attention(self.query, self.key, self.value)
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
        
        # Verify attention weights shape if returned
        if attn_weights is not None:
            self.assertEqual(attn_weights.shape[0], self.batch_size)
            self.assertEqual(attn_weights.shape[1], self.seq_len)
            self.assertEqual(attn_weights.shape[2], self.seq_len)
    
    def test_multi_query_attention(self):
        """Test Multi-Query Attention implementation for Qwen3-Coder-30B."""
        config = MultiQueryAttentionConfig(
            use_multi_query_attention=True,
            mqa_num_heads=self.num_heads,
            mqa_attention_dropout=0.1
        )
        
        attention = MultiQueryAttention(
            embed_dim=self.embed_dim,
            num_heads=config.mqa_num_heads,
            dropout=config.mqa_attention_dropout
        )
        
        output, attn_weights = attention(self.query, self.key, self.value)
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
        
        # Verify attention weights shape if returned
        if attn_weights is not None:
            self.assertEqual(attn_weights.shape[0], self.batch_size)
            self.assertEqual(attn_weights.shape[1], self.seq_len)
            self.assertEqual(attn_weights.shape[2], self.seq_len)
    
    def test_sparse_attention(self):
        """Test Sparse Attention implementation for Qwen3-0.6B."""
        config = SparseAttentionConfig(
            use_sparse_attention=True,
            sparse_num_heads=self.num_heads,
            sparse_block_size=32,
            sparse_local_window_size=64,
            sparse_attention_dropout=0.1
        )
        
        attention = SparseAttention(
            embed_dim=self.embed_dim,
            num_heads=config.sparse_num_heads,
            block_size=config.sparse_block_size,
            local_window_size=config.sparse_local_window_size,
            dropout=config.sparse_attention_dropout
        )
        
        output, attn_weights = attention(self.query, self.key, self.value)
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
        
        # Sparse attention may not return attention weights
        # Verify it runs without errors
    
    def test_sliding_window_attention(self):
        """Test Sliding Window Attention implementation for Qwen3-Coder-Next."""
        config = SlidingWindowAttentionConfig(
            use_sliding_window_attention=True,
            sliding_num_heads=self.num_heads,
            sliding_window_size=128,
            sliding_attention_dropout=0.1
        )
        
        attention = SlidingWindowAttention(
            embed_dim=self.embed_dim,
            num_heads=config.sliding_num_heads,
            window_size=config.sliding_window_size,
            dropout=config.sliding_attention_dropout
        )
        
        output, attn_weights = attention(self.query, self.key, self.value)
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
        
        # Sliding window attention may not return attention weights
        # Verify it runs without errors


class TestAttentionFactories(unittest.TestCase):
    """Test the factory functions for creating attention layers."""
    
    def test_flash_attention_factory(self):
        """Test Flash Attention factory function."""
        from src.models.specialized.glm_4_7_flash.attention.flash_attention import create_flash_attention_layer
        
        config = FlashAttentionConfig()
        layer = create_flash_attention_layer(config, embed_dim=512)
        
        self.assertIsInstance(layer, FlashAttention)
        self.assertEqual(layer.embed_dim, 512)
    
    def test_gqa_factory(self):
        """Test Grouped Query Attention factory function."""
        from src.models.language.qwen3_4b_instruct_2507.attention.grouped_query_attention import create_gqa_layer
        
        config = GroupedQueryAttentionConfig()
        layer = create_gqa_layer(config, embed_dim=512)
        
        self.assertIsInstance(layer, GroupedQueryAttention)
        self.assertEqual(layer.embed_dim, 512)
    
    def test_mqa_factory(self):
        """Test Multi-Query Attention factory function."""
        from src.models.coding.qwen3_coder_30b.attention.multi_query_attention import create_mqa_layer
        
        config = MultiQueryAttentionConfig()
        layer = create_mqa_layer(config, embed_dim=512)
        
        self.assertIsInstance(layer, MultiQueryAttention)
        self.assertEqual(layer.embed_dim, 512)
    
    def test_sparse_attention_factory(self):
        """Test Sparse Attention factory function."""
        from src.models.language.qwen3_0_6b.attention.sparse_attention import create_sparse_attention_layer
        
        config = SparseAttentionConfig()
        layer = create_sparse_attention_layer(config, embed_dim=512)
        
        self.assertIsInstance(layer, SparseAttention)
        self.assertEqual(layer.embed_dim, 512)
    
    def test_sliding_window_attention_factory(self):
        """Test Sliding Window Attention factory function."""
        from src.models.coding.qwen3_coder_next.attention.sliding_window_attention import create_sliding_window_attention_layer
        
        config = SlidingWindowAttentionConfig()
        layer = create_sliding_window_attention_layer(config, embed_dim=512)
        
        self.assertIsInstance(layer, SlidingWindowAttention)
        self.assertEqual(layer.embed_dim, 512)


if __name__ == '__main__':
    unittest.main()