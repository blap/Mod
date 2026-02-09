"""
Integration test for Qwen3-Coder-Next using Custom Backend
"""

import pytest
from src.inference_pio.core.engine.backend import Tensor, Linear, cat
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_coder_next.model import Qwen3CoderNextModel
from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig

def test_tensor_ops():
    """Verify basic backend ops needed for model."""
    t1 = Tensor([2, 2])
    t1.load([1.0, 2.0, 3.0, 4.0])

    t2 = Tensor([2, 2])
    t2.fill(0.5)

    res = t1 * t2
    data = res.to_list()
    assert data == [0.5, 1.0, 1.5, 2.0]

    # Matmul
    # [[1, 2], [3, 4]] x [[0.5, 0.5], [0.5, 0.5]] = [[1.5, 1.5], [3.5, 3.5]]
    mm = t1.matmul(t2)
    mm_data = mm.to_list()
    assert mm_data[0] == 1.5
    assert mm_data[3] == 3.5

def test_30b_model_structure():
    """Verify 30B model can instantiate without torch."""
    config = Qwen3Coder30BConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64
    )

    model = Qwen3Coder30BModel(config)

    # Check parameters
    assert model.embed_tokens.weight.shape == (100, 32)
    assert len(model.layers) == 2

    # Forward Pass Dummy
    input_ids = Tensor([1, 5]) # Batch 1, Seq 5
    input_ids.load([1.0, 10.0, 50.0, 2.0, 5.0])

    # Should run without error
    out = model(input_ids)
    assert out.shape == (1, 5, 32)

def test_next_model_structure():
    """Verify Next model can instantiate without torch."""
    config = Qwen3CoderNextConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128
    )

    model = Qwen3CoderNextModel(config)

    # Check parameters
    assert model.embed_tokens.weight.shape == (100, 32)
    assert len(model.layers) == 2

    # Forward Pass Dummy
    input_ids = Tensor([1, 5]) # Batch 1, Seq 5
    input_ids.load([1.0, 10.0, 50.0, 2.0, 5.0])

    # Should run without error
    out = model(input_ids)
    assert out.shape == (1, 5, 32)

if __name__ == "__main__":
    test_tensor_ops()
    test_30b_model_structure()
    test_next_model_structure()
    print("Tests Passed!")
