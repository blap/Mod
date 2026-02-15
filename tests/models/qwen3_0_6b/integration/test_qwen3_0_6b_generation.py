import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from src.inference_pio.core.engine.backend import Tensor
from src.inference_pio.models.qwen3_0_6b.model import Qwen3_0_6B_Model
from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6bConfig as Qwen3_0_6B_Config

class TestGeneration(unittest.TestCase):
    def test_generate(self):
        print("Testing Full Generation...")
        config = Qwen3_0_6B_Config()
        model = Qwen3_0_6B_Model(config)

        tokenizer = model._tokenizer
        if tokenizer:
             # Basic check if tokenizer works
             try:
                 ids = tokenizer.encode("Test prompt")
             except:
                 ids = [1, 2, 3]

             if isinstance(ids, list):
                 input_tensor = Tensor([1, len(ids)])
                 input_tensor.load([float(x) for x in ids])
             else:
                 input_tensor = ids
        else:
             print("Tokenizer not found. Using dummy input.")
             input_tensor = Tensor([1, 5])
             input_tensor.fill(1.0)

        max_new_tokens = 5
        output = model.generate(input_tensor, max_new_tokens=max_new_tokens)

        expected_len = input_tensor.shape[1] + max_new_tokens
        self.assertEqual(output.shape, (1, expected_len))

if __name__ == '__main__':
    unittest.main()
