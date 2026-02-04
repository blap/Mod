import unittest
import torch
from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig
from src.inference_pio.models.qwen3_coder_next.model import Qwen3CoderNextModel

class TestQwen3CoderNextStructure(unittest.TestCase):
    def setUp(self):
        # Use a tiny config for memory efficiency during test
        self.config = Qwen3CoderNextConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            deltanet_query_key_heads=4,
            deltanet_value_heads=4,
            deltanet_head_dim=32,
            attention_head_dim=64,
            attention_rope_dim=64,
            num_key_value_heads=2,
            expert_intermediate_size=64,
            num_experts=4,
            num_activated_experts=2,
            hybrid_block_pattern=["deltanet", "deltanet", "deltanet", "attention"]
        )

    def test_model_instantiation(self):
        model = Qwen3CoderNextModel(self.config)
        self.assertEqual(len(model.layers), 4)

        # Verify Hybrid Pattern
        self.assertEqual(model.layers[0].layer_type, "deltanet")
        self.assertEqual(model.layers[3].layer_type, "attention")

    def test_forward_pass(self):
        model = Qwen3CoderNextModel(self.config)
        input_ids = torch.randint(0, 1000, (1, 10))
        output = model(input_ids)

        # Check output shape [Batch, Seq, Vocab] (but model returns hidden states usually before head)
        # The provided model.py implementation returns norm(hidden_states)
        if isinstance(output, dict):
            self.assertEqual(output["last_hidden_state"].shape, (1, 10, 256))
        else:
            self.assertEqual(output[0].shape, (1, 10, 256))

if __name__ == '__main__':
    unittest.main()
