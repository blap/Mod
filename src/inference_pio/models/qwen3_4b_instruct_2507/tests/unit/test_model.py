"""
Unit Tests for Qwen3-4B-Instruct-2507 Model
"""

import unittest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen3_4B_Instruct_2507_Config
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen3_4B_Instruct_2507_Model

class TestQwen34BModel(unittest.TestCase):
    def setUp(self):
        self.config = Qwen3_4B_Instruct_2507_Config(
            hidden_size=64, num_attention_heads=4, num_hidden_layers=2, vocab_size=100, intermediate_size=128
        )
        self.model = Qwen3_4B_Instruct_2507_Model(self.config)

    def test_forward(self):
        input_ids = Tensor([1, 5])
        input_ids.load([1.0]*5)
        # model calls forward on Qwen3ForCausalLM
        out, pkv = self.model(input_ids)
        self.assertEqual(out.shape, (1, 5, 100)) # Logits [B, S, Vocab]

    def test_generate(self):
        input_ids = Tensor([1, 2])
        input_ids.load([1.0, 2.0])
        # model calls generate on Qwen3ForCausalLM
        out = self.model.generate(input_ids, max_new_tokens=3)
        self.assertEqual(out.shape, (1, 5))

if __name__ == '__main__':
    unittest.main()
