import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen3_4B_Instruct_2507_Config

class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Qwen3_4B_Instruct_2507_Config()
        self.assertEqual(c.hidden_size, 3072)

if __name__ == '__main__':
    unittest.main()
