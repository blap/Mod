import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BConfig

class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = Qwen3VL2BConfig()
        self.assertEqual(c.vision_image_size, 448)

if __name__ == '__main__':
    unittest.main()
