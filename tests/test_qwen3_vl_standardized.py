"""
Standardized tests for Qwen3-VL model components using real implementation.
"""

import pytest
import torch
from src.qwen3_vl.attention.flash_attention_2 import FlashAttention2
from src.qwen3_vl.core.config import Qwen3VLConfig

class TestQwen3VLConfig:
    """Tests for Qwen3VL configuration."""

    def test_config_initialization(self):
        """Test basic configuration initialization."""
        config = Qwen3VLConfig(
            hidden_size=1024,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=2048,
            vocab_size=50000
        )

        assert config.hidden_size == 1024
        assert config.num_attention_heads == 32
        assert config.num_key_value_heads == 32
        assert config.max_position_embeddings == 2048
        assert config.vocab_size == 50000

    def test_config_default_values(self):
        """Test configuration with default values."""
        config = Qwen3VLConfig()
        # Check that default values are properly set
        assert config.hidden_size > 0
        assert config.num_attention_heads > 0


class TestFlashAttention2:
    """Tests for FlashAttention 2 implementation."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        config = Qwen3VLConfig(
            hidden_size=256,
            num_attention_heads=8,
            num_key_value_heads=8,
            max_position_embeddings=128,
            attention_dropout_prob=0.0,
            rope_theta=10000.0
        )
        return config

    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for testing."""
        batch_size = 2
        seq_len = 64
        hidden_size = 256
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        position_ids = torch.arange(seq_len).expand((batch_size, seq_len))
        
        return {
            'hidden_states': hidden_states,
            'position_ids': position_ids
        }

    def test_flash_attention_2_initialization(self, sample_config):
        """Test FlashAttention 2 initialization."""
        attention = FlashAttention2(sample_config, layer_idx=0)
        
        assert attention.num_heads == 8
        assert attention.head_dim == 256 // 8
        assert attention.layer_idx == 0

    def test_flash_attention_2_forward_pass(self, sample_config, sample_inputs):
        """Test FlashAttention 2 forward pass."""
        attention = FlashAttention2(sample_config, layer_idx=0)
        
        output, weights, past_key_value = attention(
            hidden_states=sample_inputs['hidden_states'],
            attention_mask=None,
            position_ids=sample_inputs['position_ids'],
            output_attentions=True
        )
        
        assert output.shape == sample_inputs['hidden_states'].shape
        assert weights is not None
        assert torch.all(torch.isfinite(output))

if __name__ == "__main__":
    pytest.main([__file__])
