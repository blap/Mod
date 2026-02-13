"""
Unit Tests for GLM-4.7-Flash Model
"""

import unittest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig
from src.inference_pio.models.glm_4_7_flash.model import GLM47FlashModel

class TestGLM47FlashModel(unittest.TestCase):
    def setUp(self):
        self.config = GLM47FlashConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2, # Fixed
            vocab_size=100,
            max_position_embeddings=128
        )
        self.model = GLM47FlashModel(self.config)

    def test_forward_pass(self):
        input_ids = Tensor([1, 10]) # Batch=1, Seq=10
        input_ids.load([1.0, 5.0, 10.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        output = self.model(input_ids)

        # Output is hidden state [B, S, Hidden]
        self.assertEqual(output.shape, (1, 10, 64))

if __name__ == '__main__':
    unittest.main()
