import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig

class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Qwen3Coder30BConfig()
        self.assertEqual(c.hidden_size, 4096)

if __name__ == '__main__':
    unittest.main()
