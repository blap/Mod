"""
Basic test for core functionality
"""
import pytest
from qwen3_vl.core.config import Qwen3VLConfig
from qwen3_vl.core.inference import Qwen3VLInference, create_dummy_model


def test_config_creation():
    """Test that configuration can be created."""
    config = Qwen3VLConfig()
    assert config.num_hidden_layers == 32
    assert config.num_attention_heads == 32


def test_dummy_model_creation():
    """Test that dummy model can be created."""
    model = create_dummy_model()
    assert model is not None


def test_inference_creation():
    """Test that inference instance can be created."""
    model = create_dummy_model()
    inference = Qwen3VLInference(model)
    assert inference is not None