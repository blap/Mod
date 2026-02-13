"""
Unit Tests for Qwen3-Coder-30B Plugin
"""

import unittest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig

class TestQwen3Coder30BPlugin(unittest.TestCase):
    def test_lifecycle(self):
        plugin = Qwen3_Coder_30B_Plugin()

        # Test Initialize
        plugin.initialize(
            hidden_size=64, num_attention_heads=4, num_hidden_layers=2, vocab_size=100, intermediate_size=128
        )
        self.assertIsNotNone(plugin._model)

        # Test generate (Real execution)
        # Using Tensor input to bypass tokenization for pure model test
        input_ids = Tensor([1, 2])
        input_ids.load([1.0, 2.0])

        out = plugin.infer(input_ids)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape[1], 12) # 2 input + 10 generated

        # Test generate_text (mock tokenizer path, but real model run)
        out_text = plugin.generate_text("test", max_new_tokens=5)
        self.assertIn("Generated", out_text)

        # Cleanup
        plugin.cleanup()
        self.assertIsNone(plugin._model)

if __name__ == '__main__':
    unittest.main()
