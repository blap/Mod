import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config

class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Qwen3_0_6B_Config()
        self.assertEqual(c.hidden_size, 1024)
        self.assertEqual(c.num_attention_heads, 16)
        self.assertEqual(c.num_hidden_layers, 24)

if __name__ == '__main__':
    unittest.main()
