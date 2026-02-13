"""
Unit Tests for Qwen3-Coder-Next Model
"""

import unittest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_coder_next.config import create_qwen3_coder_next_config
from src.inference_pio.models.qwen3_coder_next.model import Qwen3CoderNextForCausalLM

class TestQwen3CoderNextModel(unittest.TestCase):
    def setUp(self):
        # Small config for fast testing
        self.config = create_qwen3_coder_next_config(
            hidden_size=64,
            num_hidden_layers=2,
            vocab_size=100,
            num_attention_heads=4,
            intermediate_size=128,
            max_position_embeddings=128,
            deltanet_query_key_heads=4,
            deltanet_value_heads=4,
            deltanet_head_dim=16,
            hybrid_block_pattern=["deltanet", "attention"],
            num_experts=4,
            num_experts_per_tok=2
        )
        self.model = Qwen3CoderNextForCausalLM(self.config)

    def test_forward_pass_shape(self):
        # Batch=1, Seq=5
        input_ids = Tensor([1, 5])
        input_ids.load([1.0, 5.0, 10.0, 2.0, 3.0])

        logits, pkv = self.model(input_ids)

        self.assertEqual(logits.shape, (1, 5, 100))
        # Default use_cache=False -> pkv is None
        self.assertIsNone(pkv)

    def test_forward_pass_with_cache(self):
        input_ids = Tensor([1, 1])
        input_ids.load([1.0])

        logits, pkv = self.model(input_ids, use_cache=True)

        self.assertIsNotNone(pkv)
        self.assertEqual(len(pkv), 2) # 2 layers

    def test_generate_simple(self):
        # Test generation loop runs without error
        input_ids = Tensor([1, 2])
        input_ids.load([1.0, 2.0])

        # Generates 3 new tokens -> total 5
        output_ids = self.model.generate(input_ids, max_new_tokens=3)

        self.assertEqual(output_ids.shape, (1, 5))

if __name__ == '__main__':
    unittest.main()
