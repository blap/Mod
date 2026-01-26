"""
Tests for the Adaptive SparseAttention implementation in the Inference-PIO system.

These tests verify the functionality, performance, and error handling of the AdaptiveSparseAttention implementation
across different adaptive strategies and configurations.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import math

from src.inference_pio.common.adaptive_sparse_attention import AdaptiveSparseAttention, create_adaptive_sparse_attention

# TestAdaptiveSparseAttention

    """Test suite for AdaptiveSparseAttention implementation."""

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
        """Test initialization of AdaptiveSparseAttention."""
        attention = AdaptiveSparseAttention(test_config)

        assert_is_instance(attention, AdaptiveSparseAttention)
        assert_equal(attention.hidden_size, 512)
        assert_equal(attention.num_attention_heads, 8)
        assert_equal(attention.head_dim, 64)  # 512 / 8
        assert_true(hasattr(attention))
        assert_true(hasattr(attention))
        assert_true(hasattr(attention))
        assert_true(hasattr(attention))
        assert_true(hasattr(attention))
        assert_true(hasattr(attention))

    def forward_pass(self)():
        """Test basic forward pass of AdaptiveSparseAttention."""
        attention = AdaptiveSparseAttention(test_config)
        attention.eval()  # Set to eval mode to avoid dropout

        hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        output, _, _ = attention(
            hidden_states=hidden_states,
        )

        # Check output shape
        assert_equal(output.shape, (batch_size))

    def forward_pass_with_attention_mask(self)():
        """Test forward pass with attention mask."""
        attention = AdaptiveSparseAttention(test_config)
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
        attention = AdaptiveSparseAttention(test_config)
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

        attention = AdaptiveSparseAttention(gqa_config)

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
            AdaptiveSparseAttention(invalid_config)

    def create_adaptive_sparse_attention_factory(self)():
        """Test factory function for creating AdaptiveSparseAttention."""
        attention = create_adaptive_sparse_attention(test_config)

        assert_is_instance(attention, AdaptiveSparseAttention)
        assert_equal(attention.hidden_size, 512)
        assert_equal(attention.num_attention_heads, 8)

    def different_adaptive_strategies(self)():
        """Test different adaptive strategies."""
        strategies = ['input_dependent', 'dynamic', 'static']

        for strategy in strategies:
            with subTest(strategy=strategy):
                attention = AdaptiveSparseAttention(
                    test_config,
                    adaptive_strategy=strategy
                )

                hidden_states = torch.randn(batch_size, seq_len, hidden_size)

                output, _, _ = attention(
                    hidden_states=hidden_states,
                )

                # Check output shape
                assert_equal(output.shape, (batch_size))

    def dtype_consistency(self)():
        """Test that the attention works with different dtypes."""
        attention = AdaptiveSparseAttention(test_config)
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

    def analyze_input_characteristics(self)():
        """Test the input characteristic analysis function."""
        attention = AdaptiveSparseAttention(test_config)
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        features = attention._analyze_input_characteristics(hidden_states)
        
        # Check that all expected features are returned
        expected_keys = ['variance', 'entropy', 'magnitude', 'position_variance', 'activation_density', 'seq_len', 'batch_size']
        for key in expected_keys:
            assert_in(key, features)
        
        # Check that numerical values are reasonable
        assert_equal(features['seq_len'], seq_len)
        assert_equal(features['batch_size'], batch_size)
        assertGreaterEqual(features['activation_density'], 0.0)
        assertLessEqual(features['activation_density'], 1.0)

    def select_adaptive_pattern(self)():
        """Test the adaptive pattern selection function."""
        attention = AdaptiveSparseAttention(test_config)
        
        # Test with short sequence
        short_features = {
            'variance': 0.1,
            'entropy': 0.5,
            'magnitude': 1.0,
            'position_variance': 0.2,
            'activation_density': 0.3,
            'seq_len': 64,
            'batch_size': batch_size
        }
        
        pattern, sparsity = attention._select_adaptive_pattern(short_features)
        assert_in(pattern, ["longformer")
        assertGreaterEqual(sparsity, 0.0)
        assertLessEqual(sparsity, 1.0)

        # Test with long sequence
        long_features = {
            'variance': 0.1,
            'entropy': 0.5,
            'magnitude': 1.0,
            'position_variance': 0.2,
            'activation_density': 0.8,  # High activation density
            'seq_len': 2048,
            'batch_size': batch_size
        }
        
        pattern, sparsity = attention._select_adaptive_pattern(long_features)
        assert_in(pattern, ["longformer")
        assertGreaterEqual(sparsity, 0.0)
        assertLessEqual(sparsity, 1.0)

# TestModelSpecificAdaptiveSparseAttention

    """Test AdaptiveSparseAttention with specific model configurations."""

    def glm47_adaptive_sparse_attention(self)():
        """Test AdaptiveSparseAttention with GLM-4.7 configuration."""
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

        attention = AdaptiveSparseAttention(
            config,
            adaptive_strategy='input_dependent',
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

    def qwen3_4b_adaptive_sparse_attention(self)():
        """Test AdaptiveSparseAttention with Qwen3-4B configuration."""
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

        attention = AdaptiveSparseAttention(
            config,
            adaptive_strategy='dynamic',
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

    def qwen3_coder_adaptive_sparse_attention(self)():
        """Test AdaptiveSparseAttention with Qwen3-Coder configuration."""
        config = MagicMock()
        config.hidden_size = 4096
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.max_position_embeddings = 32768
        config.rope_theta = 1000000.0
        config.attention_dropout_prob = 0.0
        config.sparse_attention_pattern = 'block_sparse'
        config.sparse_attention_sparsity_ratio = 0.2
        config.sparse_attention_block_size = 64
        config.sparse_attention_local_window_size = 128
        config.use_global_attention = True
        config.global_attention_indices = [0]

        attention = AdaptiveSparseAttention(
            config,
            adaptive_strategy='static',
            sparsity_ratio=config.sparse_attention_sparsity_ratio,
            block_size=config.sparse_attention_block_size,
            local_window_size=config.sparse_attention_local_window_size,
            use_global_attention=config.use_global_attention,
            global_attention_indices=config.global_attention_indices
        )

        batch_size = 1
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output, _, _ = attention(
            hidden_states=hidden_states,
        )

        assert_equal(output.shape, (batch_size))

    def qwen3_vl_adaptive_sparse_attention(self)():
        """Test AdaptiveSparseAttention with Qwen3-VL configuration."""
        config = MagicMock()
        config.hidden_size = 3584
        config.num_attention_heads = 28
        config.num_key_value_heads = 28
        config.max_position_embeddings = 4096
        config.rope_theta = 1000000.0
        config.attention_dropout_prob = 0.0
        config.sparse_attention_pattern = 'local'
        config.sparse_attention_sparsity_ratio = 0.4
        config.sparse_attention_block_size = 32
        config.sparse_attention_local_window_size = 64
        config.use_global_attention = True
        config.global_attention_indices = [0]

        attention = AdaptiveSparseAttention(
            config,
            adaptive=True,
            adaptive_strategy='input_dependent',
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

if __name__ == '__main__':
    run_tests(test_functions)