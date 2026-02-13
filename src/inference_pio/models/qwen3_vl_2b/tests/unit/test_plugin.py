"""
Unit Tests for Qwen3-VL-2B Plugin
"""

import unittest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

class TestQwen3VL2BPlugin(unittest.TestCase):
    def test_lifecycle(self):
        plugin = Qwen3_VL_2B_Plugin()

        # Test Load
        c = Qwen3VL2BConfig(
            hidden_size=64, num_attention_heads=4, num_hidden_layers=2, vocab_size=100,
            vision_hidden_size=32, vision_num_attention_heads=4, vision_num_hidden_layers=2,
            vision_patch_size=2, vision_image_size=8
        )
        plugin.load_model(config=c)
        self.assertIsNotNone(plugin._model)

        # Test Infer (Text only)
        ids = Tensor([1, 2])
        ids.load([1.0, 2.0])
        out = plugin.infer(ids)
        self.assertEqual(out.shape, (1, 12)) # 2+10

        # Test Infer (Multimodal)
        pixels = Tensor([1, 3, 8, 8])
        pixels.fill(0.5)
        out_mm = plugin.infer((ids, pixels))
        self.assertEqual(out_mm.shape, (1, 12))

if __name__ == '__main__':
    unittest.main()
