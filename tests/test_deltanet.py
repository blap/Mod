import sys
import os
import unittest
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from inference_pio.models.qwen3_coder_next.model import Qwen3CoderNextDeltaNet, Qwen3CoderNextConfig
from inference_pio.core.engine.backend import Tensor

class TestDeltaNet(unittest.TestCase):
    def test_deltanet_recurrence(self):
        # Config
        config = Qwen3CoderNextConfig(
            hidden_size=64,
            num_attention_heads=4,
            deltanet_query_key_heads=4,
            deltanet_value_heads=4,
            deltanet_head_dim=16
        )
        model = Qwen3CoderNextDeltaNet(config)

        # Input: [B=1, S=5, H=64]
        x = Tensor([1, 5, 64])
        x.fill(0.1)

        # Forward
        out, state = model(x)

        # Check shapes
        # Out: [1, 5, 64]
        self.assertEqual(out.shape, (1, 5, 64))
        # State: [1, 4, 16, 16] (B, H, D, D)
        self.assertEqual(state.shape, (1, 4, 16, 16))

        print("DeltaNet Recurrence Test Passed")

if __name__ == '__main__':
    unittest.main()
