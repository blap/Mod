"""
Unit Tests for Qwen3-VL-2B Model
"""

import unittest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel

class TestQwen3VL2BModel(unittest.TestCase):
    def setUp(self):
        self.config = Qwen3VL2BConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=100,
            vision_hidden_size=32,
            vision_num_attention_heads=4,
            vision_num_hidden_layers=2,
            vision_patch_size=2,
            vision_image_size=8 # 4x4 patches
        )
        self.model = Qwen3VL2BModel(self.config)

    def test_forward_text_only(self):
        input_ids = Tensor([1, 10])
        input_ids.load([1.0]*10)

        out, pkv = self.model(input_ids)
        self.assertEqual(out.shape, (1, 10, 64))

    def test_forward_multimodal(self):
        input_ids = Tensor([1, 5])
        input_ids.load([1.0]*5)

        # Pixel values: [B, 3, H, W]
        pixels = Tensor([1, 3, 8, 8])
        pixels.fill(0.5)

        out, pkv = self.model(input_ids, pixel_values=pixels)

        # Vis encoder: H=8, W=8, P=2 -> 4x4 patches = 16 patches
        # 16 vision tokens + 5 text tokens = 21 tokens
        self.assertEqual(out.shape, (1, 21, 64))

    def test_generate_multimodal(self):
        input_ids = Tensor([1, 2])
        input_ids.load([1.0, 2.0])
        pixels = Tensor([1, 3, 8, 8])
        pixels.fill(0.5)

        # Generates 3 new tokens
        out = self.model.generate(input_ids, pixel_values=pixels, max_new_tokens=3)

        # Output contains only text tokens (input + generated)
        self.assertEqual(out.shape, (1, 5))

if __name__ == '__main__':
    unittest.main()
