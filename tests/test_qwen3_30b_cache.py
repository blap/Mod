import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel, Qwen3Coder30BConfig
from inference_pio.models.qwen3_0_6b.architecture import Qwen3ForCausalLM, Qwen3Model
from inference_pio.core.engine.backend import Tensor

class TestKVCache30B(unittest.TestCase):
    def test_qwen_30b_cache(self):
        config = Qwen3Coder30BConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=1,
            vocab_size=100,
            intermediate_size=128,
            max_position_embeddings=128
        )
        model = Qwen3Coder30BModel(config)

        # 1. Forward full sequence
        input_ids = Tensor([1, 5])
        input_ids.fill(1.0)
        logits, cache = model(input_ids, use_cache=True)

        self.assertIsNotNone(cache)
        k, v = cache[0]
        self.assertEqual(k.shape[1], 5)

        # 2. Next token
        next_input = Tensor([1, 1])
        next_input.fill(2.0)

        logits_next, cache_next = model(next_input, past_key_values=cache, use_cache=True)
        k_next, v_next = cache_next[0]
        self.assertEqual(k_next.shape[1], 6)
        print("Qwen3-Coder-30B KV Cache Test Passed")

class TestKVCache06B(unittest.TestCase):
    def test_qwen_06b_cache(self):
        # Mock config
        class Config:
            hidden_size = 64
            num_attention_heads = 4
            num_hidden_layers = 1
            vocab_size = 100
            intermediate_size = 128
            max_position_embeddings = 128
            layer_norm_eps = 1e-6

        config = Config()
        model = Qwen3ForCausalLM(config)

        # 1. Forward
        input_ids = Tensor([1, 5])
        input_ids.fill(1.0)
        logits, cache = model(input_ids, use_cache=True)

        self.assertIsNotNone(cache)
        k, v = cache[0]
        self.assertEqual(k.shape[1], 5)

        # 2. Next token
        next_input = Tensor([1, 1])
        next_input.fill(2.0)

        logits_next, cache_next = model(next_input, past_key_values=cache, use_cache=True)
        k_next, v_next = cache_next[0]
        self.assertEqual(k_next.shape[1], 6)
        print("Qwen3-0.6B KV Cache Test Passed")

if __name__ == '__main__':
    unittest.main()
