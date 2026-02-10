"""
Unit Tests for GLM-4.7-Flash Plugin
"""

import unittest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.glm_4_7_flash.plugin import GLM_4_7_Flash_Plugin
from src.inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig

class TestGLM47FlashPlugin(unittest.TestCase):
    def test_lifecycle(self):
        plugin = GLM_4_7_Flash_Plugin()

        # Test Load with small config
        config = GLM47FlashConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=100,
            intermediate_size=128,
            max_position_embeddings=128
        )
        plugin.load_model(config=config)
        self.assertIsNotNone(plugin._model)

        # Test Infer
        input_ids = Tensor([1, 2])
        input_ids.load([1.0, 2.0])

        out = plugin.infer(input_ids)
        self.assertIsNotNone(out)
        # Default generation adds 10 tokens
        self.assertEqual(out.shape[1], 12)

if __name__ == '__main__':
    unittest.main()
