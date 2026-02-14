import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.glm_4_7_flash.model import GLM47FlashModel
from src.inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig

class TestLayers(unittest.TestCase):
    def test_layer_forward(self):
        c = GLM47FlashConfig(num_hidden_layers=1)
        m = GLM47FlashModel(c)
        layer = m.layers[0]
        x = Tensor([1, 5, c.hidden_size]); x.fill(0.1)
        out = layer(x)
        self.assertEqual(out.shape, (1, 5, c.hidden_size))

if __name__ == '__main__':
    unittest.main()
