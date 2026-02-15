import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig

class TestLayers(unittest.TestCase):
    def test_layer_forward(self):
        c = Qwen3Coder30BConfig(num_hidden_layers=1)
        m = Qwen3Coder30BModel(c)
        layer = m.layers[0]
        x = Tensor([1, 5, 4096]); x.fill(0.1)
        out, _ = layer(x)
        self.assertEqual(out.shape, (1, 5, 4096))

if __name__ == '__main__':
    unittest.main()
