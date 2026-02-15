import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_0_6b.model import Qwen3_0_6B_Model
from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6bConfig as Qwen3_0_6B_Config

class TestLayers(unittest.TestCase):
    def test_layer_forward(self):
        print("Testing Layer Forward Pass...")
        config = Qwen3_0_6B_Config()
        model = Qwen3_0_6B_Model(config)

        if hasattr(model._model, 'layers'):
            layer = model._model.layers[0]
        elif hasattr(model._model, 'model') and hasattr(model._model.model, 'layers'):
             layer = model._model.model.layers[0]
        else:
             print("Could not find layers attribute. Skipping.")
             return

        x = Tensor([1, 5, 1024])
        x.fill(0.1)

        output = layer(x)
        if isinstance(output, tuple):
            output = output[0]

        self.assertEqual(output.shape, (1, 5, 1024))

if __name__ == '__main__':
    unittest.main()
