import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from inference_pio.models.qwen3_coder_next.model import Qwen3CoderNextForCausalLM, Qwen3CoderNextConfig
from inference_pio.models.glm_4_7_flash.architecture import GLMForCausalLM
from inference_pio.core.engine.backend import Tensor

class TestKVCache(unittest.TestCase):
    def test_qwen_cache(self):
        config = Qwen3CoderNextConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=1,
            vocab_size=100,
            intermediate_size=128,
            max_position_embeddings=128,
            deltanet_head_dim=16, # reduced for test speed
            deltanet_query_key_heads=4
        )
        model = Qwen3CoderNextForCausalLM(config)

        # 1. Forward full sequence
        input_ids = Tensor([1, 5]) # Length 5
        input_ids.fill(1.0)
        logits, cache = model(input_ids, use_cache=True)

        self.assertIsNotNone(cache)
        self.assertEqual(len(cache), 1) # 1 layer

        # Layer 0 is DeltaNet (default pattern starts with deltanet)
        state = cache[0]
        # State should be a Tensor [B, H, D, D]
        # [1, 4, 16, 16]
        self.assertTrue(isinstance(state, Tensor))
        self.assertEqual(state.shape, (1, 4, 16, 16))

        # 2. Generate step (next token)
        next_input = Tensor([1, 1])
        next_input.fill(2.0)

        logits_next, cache_next = model(next_input, past_key_values=cache, use_cache=True)
        state_next = cache_next[0]

        # State should be updated (shape remains same)
        self.assertEqual(state_next.shape, (1, 4, 16, 16))

        print("Qwen3-Coder-Next KV Cache Test Passed")

    def test_glm_cache(self):
        # Mock config
        class Config:
            hidden_size = 64
            num_attention_heads = 4
            num_hidden_layers = 1
            vocab_size = 100
            intermediate_size = 128
            max_position_embeddings = 128
            layer_norm_eps = 1e-5

        config = Config()
        model = GLMForCausalLM(config)

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

        print("GLM-4.7-Flash KV Cache Test Passed")

if __name__ == '__main__':
    unittest.main()
