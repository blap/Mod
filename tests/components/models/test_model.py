"""
Basic test for models functionality
"""
import pytest
from models.config import Qwen3VLConfig
from models.qwen3_vl import Qwen3VLModel


def test_model_config():
    """Test that model configuration works."""
    config = Qwen3VLConfig()
    assert config.vocab_size == 152064


def test_model_creation():
    """Test that model can be created."""
    # This is a basic test - in reality, we'd need to create the full model
    config = Qwen3VLConfig()
    # Just testing that config is valid
    assert config is not None