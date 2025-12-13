"""
Standardized test configuration for Qwen3-VL project.
This file defines common test fixtures, utilities, and configuration patterns.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import project modules
from src.qwen3_vl.config.config import Qwen3VLConfig


@pytest.fixture(scope="session")
def device():
    """Global fixture to determine the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def default_config():
    """Default configuration for testing."""
    config = Qwen3VLConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=128,
        vocab_size=1000,
        intermediate_size=512,
        num_hidden_layers=4
    )
    return config


@pytest.fixture
def sample_text_inputs():
    """Sample text inputs for testing."""
    return {
        'input_ids': torch.randint(0, 1000, (2, 32)),  # batch_size=2, seq_len=32
        'attention_mask': torch.ones(2, 32),
        'position_ids': torch.arange(32).unsqueeze(0).expand(2, 32)
    }


@pytest.fixture
def sample_image_inputs():
    """Sample image inputs for testing."""
    return torch.randn(2, 3, 224, 224)  # batch_size=2, channels=3, height=224, width=224


@pytest.fixture
def sample_multimodal_inputs():
    """Sample multimodal inputs combining text and images."""
    return {
        'input_ids': torch.randint(0, 1000, (2, 32)),
        'pixel_values': torch.randn(2, 3, 224, 224),
        'attention_mask': torch.ones(2, 32),
        'position_ids': torch.arange(32).unsqueeze(0).expand(2, 32)
    }


@pytest.fixture
def mock_model():
    """Mock model for testing without loading actual weights."""
    model = Mock()
    model.device = torch.device('cpu')
    model.forward = Mock(return_value=torch.randn(2, 32, 256))
    model.generate = Mock(return_value=torch.randint(0, 1000, (2, 64)))
    model.state_dict = Mock(return_value={})
    model.load_state_dict = Mock()
    model.train = Mock()
    model.eval = Mock()
    return model


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup is handled by test runner


@pytest.fixture
def sample_tensor():
    """Sample tensor for basic operations."""
    return torch.randn(2, 16, 128)  # batch_size=2, seq_len=16, hidden_size=128


@pytest.fixture
def sample_attention_mask():
    """Sample attention mask for testing."""
    # Create a causal mask
    seq_len = 32
    mask = torch.tril(torch.ones(1, 1, seq_len, seq_len))
    mask = (1.0 - mask) * torch.finfo(torch.float32).min
    return mask


def create_test_config(**kwargs) -> Qwen3VLConfig:
    """Helper function to create a test configuration with custom parameters."""
    default_params = {
        'hidden_size': 256,
        'num_attention_heads': 8,
        'num_key_value_heads': 8,
        'max_position_embeddings': 128,
        'vocab_size': 1000,
        'intermediate_size': 512,
        'num_hidden_layers': 4
    }
    
    # Override defaults with provided kwargs
    default_params.update(kwargs)
    
    return Qwen3VLConfig(**default_params)


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], 
                       msg: str = "Tensor shape mismatch"):
    """Helper to assert tensor shape."""
    assert tensor.shape == expected_shape, f"{msg}: expected {expected_shape}, got {tensor.shape}"


def assert_tensor_finite(tensor: torch.Tensor, msg: str = "Tensor contains non-finite values"):
    """Helper to assert tensor contains only finite values."""
    assert torch.all(torch.isfinite(tensor)), msg


def assert_tensors_close(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                        atol: float = 1e-6, rtol: float = 1e-5,
                        msg: str = "Tensors are not close"):
    """Helper to assert two tensors are close."""
    assert torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol), msg


def measure_gpu_memory():
    """Helper to measure GPU memory usage if available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB
    return 0.0


def reset_gpu_memory():
    """Helper to reset GPU memory stats if available."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def benchmark_function(func, *args, num_runs=5, **kwargs):
    """Helper to benchmark a function's execution time."""
    import time
    
    # Warmup
    for _ in range(2):
        func(*args, **kwargs)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        
        result = func(*args, **kwargs)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        
        times.append(end - start)
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'times': times,
        'result': result
    }


# Parametrized test configurations for different model sizes
MODEL_SIZES = [
    {"hidden_size": 128, "num_attention_heads": 4, "max_position_embeddings": 64},
    {"hidden_size": 256, "num_attention_heads": 8, "max_position_embeddings": 128},
    {"hidden_size": 512, "num_attention_heads": 16, "max_position_embeddings": 256},
]


@pytest.fixture(params=MODEL_SIZES)
def model_size_config(request):
    """Parametrized fixture for testing different model sizes."""
    params = request.param
    return create_test_config(**params)


# Test markers for categorizing tests
def pytest_configure(config):
    """Configuration hook for pytest."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "multimodal: mark test as multimodal-specific"
    )