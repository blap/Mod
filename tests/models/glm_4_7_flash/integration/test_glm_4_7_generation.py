import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.glm_4_7_flash.model import GLM47FlashModel
from src.inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig

class TestGeneration(unittest.TestCase):
    def test_generate(self):
        c = GLM47FlashConfig(num_hidden_layers=1)
        m = GLM47FlashModel(c)
        input_ids = Tensor([1, 5]); input_ids.fill(1.0)
        out = m.generate(input_ids, max_new_tokens=5)
        self.assertEqual(out.shape, (1, 10))

if __name__ == '__main__':
    unittest.main()
