import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from inference_pio.models.qwen3_vl_2b.vision_transformer_kernels import VisionSelfAttentionKernel, VisionTransformerConfig
from inference_pio.core.engine.backend import Tensor

class TestVisionShape(unittest.TestCase):
    def test_forward_shape(self):
        config = VisionTransformerConfig(
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=1,
            patch_size=14,
            image_size=28,
            intermediate_size=128
        )
        model = VisionSelfAttentionKernel(config)

        # Input: [B=1, Seq=4, Hidden=64]
        x = Tensor([1, 4, 64])
        x.fill(0.1)

        # Run forward
        try:
            out = model(x)
            # Check output shape [1, 4, 64]
            self.assertEqual(out.shape, (1, 4, 64))
            print("Vision Attention Shape Test Passed")
        except Exception as e:
            self.fail(f"Vision Attention Failed: {e}")

if __name__ == '__main__':
    unittest.main()
