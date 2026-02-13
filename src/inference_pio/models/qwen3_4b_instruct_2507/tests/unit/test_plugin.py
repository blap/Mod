"""
Unit Tests for Qwen3-4B-Instruct-2507 Plugin
"""

import unittest
import os
import sys
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin

class TestQwen34BPlugin(unittest.TestCase):
    def test_lifecycle(self):
        plugin = Qwen3_4B_Instruct_2507_Plugin()
        
        # Initialize
        plugin.initialize(
            hidden_size=64, num_attention_heads=4, num_hidden_layers=2, vocab_size=100, intermediate_size=128
        )
        self.assertIsNotNone(plugin._model)

        # Mock Tokenizer for testing text generation flow
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1.0, 2.0] # Tensor load expects floats
        mock_tokenizer.decode.return_value = "Mock Output"
        plugin._model._tokenizer = mock_tokenizer

        # Test generate_text
        out = plugin.generate_text("Test prompt", max_new_tokens=3)
        self.assertEqual(out, "Mock Output")

        # Verify encode called
        mock_tokenizer.encode.assert_called_with("Test prompt")

        # Cleanup
        plugin.cleanup()
        self.assertIsNone(plugin._model)

if __name__ == '__main__':
    unittest.main()
