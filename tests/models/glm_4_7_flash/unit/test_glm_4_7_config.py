import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
from src.inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig

class TestConfig(unittest.TestCase):
    def test_defaults(self):
        c = GLM47FlashConfig()
        self.assertTrue(c.hidden_size > 0)

if __name__ == '__main__':
    unittest.main()
