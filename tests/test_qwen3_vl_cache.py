import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel, Qwen3VL2BConfig
from inference_pio.core.engine.backend import Tensor

class TestQwen3VLCache(unittest.TestCase):
    def test_forward_no_cache(self):
        config = Qwen3VL2BConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=1,
            vocab_size=100,
            vision_hidden_size=32,
            vision_num_attention_heads=2,
            vision_num_hidden_layers=1,
            vision_patch_size=14,
            vision_image_size=28
        )
        model = Qwen3VL2BModel(config)

        input_ids = Tensor([1, 5])
        input_ids.fill(1.0)
        pixel_values = Tensor([1, 3, 28, 28])
        pixel_values.fill(0.1)

        # Test 1: Full forward with vision
        h, cache = model(input_ids, pixel_values=pixel_values, use_cache=True)
        self.assertIsNotNone(cache)
        k, v = cache[0]
        # Vision Tokens: (28/14)^2 = 4 patches.
        # Total Seq: 4 (vis) + 5 (text) = 9
        self.assertEqual(k.shape[1], 9)

        # Test 2: Next token
        next_input = Tensor([1, 1])
        next_input.fill(2.0)
        h_next, cache_next = model(next_input, past_key_values=cache, use_cache=True)

        k_next, v_next = cache_next[0]
        self.assertEqual(k_next.shape[1], 10)

        print("Qwen3-VL-2B KV Cache Test Passed")

if __name__ == '__main__':
    unittest.main()
