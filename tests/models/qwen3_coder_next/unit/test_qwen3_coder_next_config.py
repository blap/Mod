import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig

class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Qwen3CoderNextConfig()
        self.assertTrue(hasattr(c, 'hybrid_block_pattern'))

if __name__ == '__main__':
    unittest.main()
