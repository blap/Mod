"""
Pytest configuration file for Qwen3-VL project.

This file contains shared fixtures and configuration for all tests in the project.
It imports the standardized test configuration to ensure consistency.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path
import sys

# Import the standardized test configuration
from tests.test_config import *  # Import all standardized fixtures and utilities

# Additional project-specific fixtures can be added here
@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    from src.qwen3_vl.components.configuration import Qwen3VLConfig

    return Qwen3VLConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=128,
        vocab_size=1000,
        intermediate_size=512,
        num_hidden_layers=4
    )


@pytest.fixture
def sample_text_batch():
    """Create a sample batch of text for testing."""
    return [
        "This is a sample text for testing.",
        "Another sample text for the batch.",
        "Third text in the batch.",
        "Final text in the batch."
    ]


@pytest.fixture
def sample_image_batch():
    """Create a sample batch of images for testing."""
    from PIL import Image
    import numpy as np

    images = []
    for i in range(4):
        # Create a simple test image
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        images.append(img)
    return images


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.device = torch.device('cpu')
    model.parameters = Mock(return_value=iter([torch.nn.Parameter(torch.randn(10, 10))]))
    model.generate = Mock(return_value=torch.randint(0, 1000, (4, 10)))
    model.forward = Mock(return_value=torch.randn(4, 10))
    return model


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    import shutil
    shutil.rmtree(temp_dir)