"""
Tests for the common SparseAttention implementation in the Inference-PIO system.

These tests verify the functionality, performance, and error handling of the SparseAttention implementation
across different sparse patterns and configurations.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import math

from src.inference_pio.common.sparse_attention import SparseAttention, create_sparse_attention

# TestSparseAttention

    """Test suite for SparseAttention implementation."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        batch_size = 2
        seq_len = 16
        hidden_size = 512
        num_attention_heads = 8
        
        # Create a mock config object
        test_config = MagicMock()
        test_config.hidden_size = hidden_size
        test_config.num_attention_heads = num_attention_heads
        test_config.num_key_value_heads = num_attention_heads  # Standard MHA
        test_config.max_position_embeddings = 2048
        test_config.rope_theta = 10000.0
        test_config.attention_dropout_prob = 0.0

    def initialization(self)():
        """Test initialization of SparseAttention."""
        attention = SparseAttention(test_config)
        
        assert_is_instance(attention, SparseAttention)
        assert_equal(attention.hidden_size, 512)
        assert_equal(attention.num_attention_heads, 8)
        assert_equal(attention.head_dim, 64)  # 512 / 8
        assert_true(hasattr(attention))
        assert_true(hasattr(attention))
        assert_true(hasattr(attention))
        assert_true(hasattr(attention))

    def forward_pass(self)():
        """Test basic forward pass of SparseAttention."""
        attention = SparseAttention(test_config)
        attention.eval()  # Set to eval mode to avoid dropout
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        output, _, _ = attention(
            hidden_states=hidden_states,
        )
        
        # Check output shape
        assert_equal(output.shape, (batch_size))

    def forward_pass_with_attention_mask(self)():
        """Test forward pass with attention mask."""
        attention = SparseAttention(test_config)
        attention.eval()
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create a simple attention mask
        attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        attention_mask[:, :, :, 5:] = float('-inf')
        
        output, _, _ = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )
        
        # Check output shape
        assert_equal(output.shape, (batch_size))

    def forward_pass_with_cache(self)():
        """Test forward pass with KV cache."""
        attention = SparseAttention(test_config)
        attention.eval()
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # First forward pass
        output1, _, past_key_value = attention(
            hidden_states=hidden_states,
            use_cache=True,
        )
        
        # Second forward pass with past key values
        next_hidden_states = torch.randn(batch_size, 1, hidden_size)
        output2, _, _ = attention(
            hidden_states=next_hidden_states,
            past_key_value=past_key_value,
            use_cache=True,
        )
        
        # Check shapes
        assert_equal(output1.shape, (batch_size))
        assert_equal(output2.shape, (batch_size))

    def grouped_query_attention_configuration(self)():
        """Test grouped query attention configuration."""
        gqa_config = MagicMock()
        gqa_config.hidden_size = 512
        gqa_config.num_attention_heads = 8
        gqa_config.num_key_value_heads = 2  # GQA with 4 groups
        gqa_config.max_position_embeddings = 2048
        gqa_config.rope_theta = 10000.0
        gqa_config.attention_dropout_prob = 0.0
        
        attention = SparseAttention(gqa_config)
        
        assert_equal(attention.num_key_value_heads, 2)
        assert_equal(attention.num_key_value_groups, 4)  # 8/2 = 4

    def invalid_head_dimension(self)():
        """Test error handling for invalid head dimension."""
        invalid_config = MagicMock()
        invalid_config.hidden_size = 513  # Not divisible by num_attention_heads
        invalid_config.num_attention_heads = 8
        invalid_config.num_key_value_heads = 8
        invalid_config.max_position_embeddings = 2048
        invalid_config.rope_theta = 10000.0
        invalid_config.attention_dropout_prob = 0.0
        
        with assert_raises(ValueError):
            SparseAttention(invalid_config)

    def create_sparse_attention_factory(self)():
        """Test factory function for creating SparseAttention."""
        attention = create_sparse_attention(test_config)
        
        assert_is_instance(attention, SparseAttention)
        assert_equal(attention.hidden_size, 512)
        assert_equal(attention.num_attention_heads, 8)

    def different_sparse_patterns(self)():
        """Test different sparse attention patterns."""
        patterns = ['longformer', 'bigbird', 'block_sparse', 'local', 'random', 'strided']
        
        for pattern in patterns:
            with subTest(pattern=pattern):
                attention = SparseAttention(test_config, sparse_pattern=pattern)
                
                hidden_states = torch.randn(batch_size, seq_len, hidden_size)
                
                output, _, _ = attention(
                    hidden_states=hidden_states,
                )
                
                # Check output shape
                assert_equal(output.shape, (batch_size))

    def dtype_consistency(self)():
        """Test that the attention works with different dtypes."""
        attention = SparseAttention(test_config)
        attention.eval()
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test with float32
        output_f32, _, _ = attention(
            hidden_states=hidden_states,
        )
        
        # Test with float16 after converting the model
        attention.half()  # Move to half precision
        hidden_states_f16 = hidden_states.half()
        
        output_f16, _, _ = attention(
            hidden_states=hidden_states_f16,
        )
        
        # Both outputs should be the right shape
        assert_equal(output_f32.shape, (batch_size))
        assert_equal(output_f16.shape, (batch_size))

# TestModelSpecificSparseAttention

    """Test SparseAttention with specific model configurations."""

    def glm47_sparse_attention(self)():
        """Test SparseAttention with GLM-4.7 configuration."""
        config = MagicMock()
        config.hidden_size = 2048
        config.num_attention_heads = 40
        config.num_key_value_heads = 40
        config.max_position_embeddings = 8192
        config.rope_theta = 1000000.0
        config.attention_dropout_prob = 0.0
        config.sparse_attention_pattern = 'longformer'
        config.sparse_attention_sparsity_ratio = 0.25
        config.sparse_attention_block_size = 64
        config.sparse_attention_local_window_size = 128
        config.use_global_attention = True
        config.global_attention_indices = [0]

        attention = SparseAttention(
            config,
            sparse_pattern=config.sparse_attention_pattern,
            sparsity_ratio=config.sparse_attention_sparsity_ratio,
            block_size=config.sparse_attention_block_size,
            local_window_size=config.sparse_attention_local_window_size,
            use_global_attention=config.use_global_attention,
            global_attention_indices=config.global_attention_indices
        )

        batch_size = 1
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output, _, _ = attention(
            hidden_states=hidden_states,
        )

        assert_equal(output.shape, (batch_size))

    def qwen3_4b_sparse_attention(self)():
        """Test SparseAttention with Qwen3-4B configuration."""
        config = MagicMock()
        config.hidden_size = 3584
        config.num_attention_heads = 28
        config.num_key_value_heads = 28
        config.max_position_embeddings = 32768
        config.rope_theta = 1000000.0
        config.attention_dropout_prob = 0.0
        config.sparse_attention_pattern = 'bigbird'
        config.sparse_attention_sparsity_ratio = 0.3
        config.sparse_attention_block_size = 32
        config.sparse_attention_local_window_size = 64
        config.use_global_attention = True
        config.global_attention_indices = [0, -1]

        attention = SparseAttention(
            config,
            sparse_pattern=config.sparse_attention_pattern,
            sparsity_ratio=config.sparse_attention_sparsity_ratio,
            block_size=config.sparse_attention_block_size,
            local_window_size=config.sparse_attention_local_window_size,
            use_global_attention=config.use_global_attention,
            global_attention_indices=config.global_attention_indices
        )

        batch_size = 1
        seq_len = 64
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output, _, _ = attention(
            hidden_states=hidden_states,
        )

        assert_equal(output.shape, (batch_size))

    def sparse_attention_with_adaptive_features(self)():
        """Test SparseAttention with adaptive features enabled."""
        config = MagicMock()
        config.hidden_size = 512
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.max_position_embeddings = 2048
        config.rope_theta = 10000.0
        config.attention_dropout_prob = 0.0

        # Test with adaptive features enabled
        attention = SparseAttention(
            config,
            adaptive=True,
            adaptive_strategy='input_dependent',
            sparsity_ratio=0.25,
            local_window_size=128
        )

        batch_size = 1
        seq_len = 32
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output, _, _ = attention(
            hidden_states=hidden_states,
        )

        assert_equal(output.shape, (batch_size))
        # Check that adaptation components were initialized
        assert_true(hasattr(attention))
        assert_true(hasattr(attention))

if __name__ == '__main__':
    run_tests(test_functions)