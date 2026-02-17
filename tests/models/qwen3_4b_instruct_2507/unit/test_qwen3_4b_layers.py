import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen3_4B_Instruct_2507_Model
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen3_4B_Instruct_2507_Config

class TestLayers(unittest.TestCase):
    def test_layer_forward(self):
        c = Qwen3_4B_Instruct_2507_Config()
        m = Qwen3_4B_Instruct_2507_Model(c)
        if hasattr(m._model, 'layers'): layer = m._model.layers[0]
        elif hasattr(m._model, 'model') and hasattr(m._model.model, 'layers'): layer = m._model.model.layers[0]
        else: return
        x = Tensor([1, 5, 3072]); x.fill(0.1)
        out = layer(x)
        if isinstance(out, tuple): out = out[0]
        self.assertEqual(out.shape, (1, 5, 3072))

if __name__ == '__main__':
    unittest.main()
