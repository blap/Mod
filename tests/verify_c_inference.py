import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import logging

sys.path.insert(0, os.getcwd())
logging.basicConfig(level=logging.INFO)

class TestCInference(unittest.TestCase):
    def setUp(self):
        # Mock torch and numpy as missing
        self.patcher = patch.dict(sys.modules, {'torch': None, 'numpy': None})
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_c_engine_init(self):
        from src.inference_pio.core.engine.backend import Tensor, zeros
        t = zeros([2, 2])
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.size, 4)
        print("C Tensor initialized")

    def test_qwen3_06b_c_init(self):
        from src.inference_pio.models.qwen3_0_6b.architecture import Qwen3ForCausalLM
        class Config:
            hidden_size = 64
            num_attention_heads = 4
            num_hidden_layers = 2
            vocab_size = 1000
            intermediate_size = 128
            max_position_embeddings = 512
            rope_theta = 10000.0
            num_key_value_heads = 4
            rms_norm_eps = 1e-6

        config = Config()
        try:
            model = Qwen3ForCausalLM(config)
            print("Qwen3-0.6B C Model Initialized")
        except Exception as e:
            self.fail(f"Init failed: {e}")

if __name__ == '__main__':
    unittest.main()
