import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel, Qwen3VL2BConfig

class TestLayers(unittest.TestCase):
    def test_layer_forward(self):
        c = Qwen3VL2BConfig(num_hidden_layers=1)
        m = Qwen3VL2BModel(c)
        layer = m.layers[0]
        x = Tensor([1, 5, c.hidden_size]); x.fill(0.1)
        out, _ = layer(x)
        self.assertEqual(out.shape, (1, 5, c.hidden_size))

if __name__ == '__main__':
    unittest.main()
