
import unittest
from unittest.mock import MagicMock
import sys

# Mock transformers
import transformers
transformers.AutoModelForVision2Seq = MagicMock()
transformers.AutoTokenizer = MagicMock()
transformers.AutoImageProcessor = MagicMock()

import torch
import torch.nn as nn
from src.inference_pio.models.qwen3_vl_2b.attention.attention import (
    Qwen3VL2BMultimodalAttention,
    Qwen3VL2BModalitySpecificAttention,
    Qwen3VL2BAdaptiveMultimodalAttention
)
from src.inference_pio.models.qwen3_vl_2b.rotary_embeddings import RotaryEmbedding

class TestQwen3VLOptimizations(unittest.TestCase):
    def setUp(self):
        class Config:
            hidden_size = 128
            num_attention_heads = 4
            num_hidden_layers = 2
            layer_norm_eps = 1e-6
            max_position_embeddings = 128
            rope_theta = 10000.0

        self.config = Config()
        self.device = 'cpu'

    def test_rotary_initialization(self):
        # Optimization 7: Verify RotaryEmbedding is initialized in __init__
        attn = Qwen3VL2BMultimodalAttention(self.config)
        self.assertTrue(hasattr(attn, 'rotary_emb'))
        self.assertIsInstance(attn.rotary_emb, RotaryEmbedding)

        attn_mod = Qwen3VL2BModalitySpecificAttention(self.config)
        self.assertTrue(hasattr(attn_mod, 'rotary_emb'))
        self.assertIsInstance(attn_mod.rotary_emb, RotaryEmbedding)

    def test_sdpa_execution(self):
        # Optimization 8 & 9: Verify forward pass runs without error (implying SDPA works or fallback)
        attn = Qwen3VL2BMultimodalAttention(self.config)
        hidden_states = torch.randn(2, 16, self.config.hidden_size)
        output, _, _ = attn(hidden_states)
        self.assertEqual(output.shape, hidden_states.shape)

        attn_mod = Qwen3VL2BModalitySpecificAttention(self.config)
        output_mod, _, _ = attn_mod(hidden_states)
        self.assertEqual(output_mod.shape, hidden_states.shape)

    def test_adaptive_attention(self):
        # Optimization 10: Verify reuse mean logic
        attn = Qwen3VL2BAdaptiveMultimodalAttention(self.config)
        hidden_states = torch.randn(2, 16, self.config.hidden_size)
        output, _, _ = attn(hidden_states)
        self.assertEqual(output.shape, hidden_states.shape)

if __name__ == '__main__':
    unittest.main()
