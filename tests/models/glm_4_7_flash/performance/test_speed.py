import unittest
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.glm_4_7_flash.model import GLM47FlashModel
from src.inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig

class TestSpeed(unittest.TestCase):
    def test_tps(self):
        c = GLM47FlashConfig(num_hidden_layers=1)
        m = GLM47FlashModel(c)
        input_ids = Tensor([1, 5]); input_ids.fill(1.0)
        start = time.time()
        m.generate(input_ids, max_new_tokens=10)
        tps = 10 / (time.time() - start)
        self.assertGreater(tps, 0.001)

if __name__ == '__main__':
    unittest.main()
