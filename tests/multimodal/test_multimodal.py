"""
Basic test for multimodal functionality
"""
import pytest
import torch
from qwen3_vl.multimodal.cross_modal_compression import CrossModalCompression


def test_cross_modal_compression():
    """Test cross modal compression functionality."""
    # This is a basic test - in reality, we'd have more complex testing
    compression = CrossModalCompression()
    assert compression is not None