"""
Unit Tests for Qwen3-Coder-Next Plugin
"""

import unittest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_coder_next.plugin import Qwen3CoderNextPlugin

class TestQwen3CoderNextPlugin(unittest.TestCase):
    def test_plugin_lifecycle(self):
        plugin = Qwen3CoderNextPlugin()

        # Test Initialize
        plugin.initialize(
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
            num_activated_experts=2,
            num_experts_per_tok=2,
            model_path="dummy/path/to/model"
        )
        self.assertIsNotNone(plugin.config)
        self.assertEqual(plugin.config.hidden_size, 64)

        # Test Load Model (mock path/no weights)
        # This will fail to load real weights but catch exception and init random weights
        plugin.load_model()
        self.assertIsNotNone(plugin._model)

        # Test Infer with Tensor
        input_ids = Tensor([1, 2])
        input_ids.load([1.0, 2.0])

        # Default generate args used inside infer (max_new_tokens=10)
        out = plugin.infer(input_ids)
        self.assertIsNotNone(out)
        self.assertEqual(out.shape[1], 12) # 2 (input) + 10 (default max_new_tokens)

        # Cleanup
        plugin.cleanup()
        self.assertIsNone(plugin._model)

if __name__ == '__main__':
    unittest.main()
