"""
Final Verification Test for Image Tokenization System in Qwen3-VL-2B Model

This test verifies that the efficient image tokenization system is properly
integrated with the Qwen3-VL-2B model and functions as expected.
"""

import os
import sys

# Add root to path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from src.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.models.qwen3_vl_2b.image_tokenization import (
    ImageTokenizationConfig,
    ImageTokenizer,
)
from src.models.qwen3_vl_2b.model import Qwen3VL2BModel


def test_image_tokenization_creation():
    """Test creating the image tokenization system."""
    print("Testing image tokenization system creation...")

    config = ImageTokenizationConfig(
        image_size=448, patch_size=14, max_image_tokens=1024, token_dim=1024
    )

    tokenizer = ImageTokenizer(config, image_processor=MagicMock())

    assert tokenizer is not None, "Image tokenizer should be created"
    assert tokenizer.config == config, "Config should match"

    print("PASS: Image tokenization system created successfully")


def test_image_tokenization_functionality():
    """Test the image tokenization functionality."""
    print("Testing image tokenization functionality...")

    config = ImageTokenizationConfig(
        image_size=224, patch_size=16, max_image_tokens=256, token_dim=512
    )

    # Mock image processor behavior
    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

    tokenizer = ImageTokenizer(config, image_processor=mock_processor)

    # Create a test image
    image = Image.new("RGB", (224, 224), color="red")

    # Tokenize the image
    result = tokenizer.tokenize(image)

    assert "pixel_values" in result, "Result should contain pixel_values"
    pixel_values = result["pixel_values"]
    # The output dimension depends on the mock and internal logic.
    # Qwen3-VL often uses patches, but here we just check it returns something valid.
    assert pixel_values is not None

    print(f"PASS: Image tokenized successfully, shape: {pixel_values.shape}")


def test_qwen3_vl_2b_integration():
    """Test integration with Qwen3-VL-2B model."""
    print("Testing Qwen3-VL-2B model integration...")

    # Create config
    model_config = Qwen3VL2BConfig()
    model_config.model_path = "dummy_path"
    model_config.use_flash_attention_2 = False
    model_config.use_cuda_kernels = False
    model_config.enable_disk_offloading = False
    model_config.enable_intelligent_pagination = False
    model_config.use_multimodal_attention = False
    model_config.enable_image_tokenization = True  # Enable image tokenization

    # Mock the model loading
    with patch(
        "src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained"
    ) as mock_model, patch(
        "src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer, patch(
        "src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained"
    ) as mock_image_proc:

        # Set up mocks
        mock_model_instance = MagicMock()
        mock_model_instance.gradient_checkpointing_enable = lambda: None
        mock_model_instance.config = MagicMock()
        mock_model_instance.config.hidden_size = 2048
        mock_model_instance.config.num_attention_heads = 16
        mock_model_instance.config.num_hidden_layers = 24

        # Ensure either AutoModel or Qwen2VL returns this
        mock_model.return_value = mock_model_instance

        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "</s>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_image_proc_instance = MagicMock()
        mock_image_proc.return_value = mock_image_proc_instance

        # Create the model
        # We need to ensure we don't trigger real downloads or other init issues
        model = Qwen3VL2BModel(model_config)

        # Verify integration
        assert (
            model._image_tokenizer is not None
        ), "Image tokenizer should be initialized"
        assert hasattr(
            model._model, "image_tokenizer"
        ), "Model should have image_tokenizer attribute"
        assert (
            model._model.image_tokenizer is not None
        ), "Model's image_tokenizer should not be None"

        print("PASS: Qwen3-VL-2B model integration successful")


def test_plugin_integration():
    """Test integration with the Qwen3-VL-2B plugin."""
    print("Testing plugin integration...")

    from src.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin

    # Check that the class has the required methods
    plugin_cls = Qwen3_VL_2B_Instruct_Plugin

    # Test encode_image method exists
    assert hasattr(plugin_cls, "encode_image"), "Plugin should have encode_image method"
    assert callable(
        getattr(plugin_cls, "encode_image")
    ), "encode_image should be callable"

    # Test tokenize method exists (for text)
    assert hasattr(plugin_cls, "tokenize"), "Plugin should have tokenize method"
    assert callable(getattr(plugin_cls, "tokenize")), "tokenize should be callable"

    print("PASS: Plugin integration successful")


def test_performance_metrics():
    """Test performance metrics functionality."""
    print("Testing performance metrics...")

    config = ImageTokenizationConfig()
    tokenizer = ImageTokenizer(config, image_processor=MagicMock())

    # Check initial metrics
    initial_stats = tokenizer.get_performance_stats()
    assert "total_tokenization_time" in initial_stats
    assert "num_tokenization_calls" in initial_stats
    assert "average_tokenization_time" in initial_stats
    assert initial_stats["num_tokenization_calls"] == 0

    # Reset metrics
    tokenizer.reset_performance_stats()
    reset_stats = tokenizer.get_performance_stats()
    assert reset_stats["num_tokenization_calls"] == 0
    assert reset_stats["total_tokenization_time"] == 0.0

    print("PASS: Performance metrics functionality verified")


def run_all_tests():
    """Run all verification tests."""
    print("Running final verification tests for Image Tokenization System...\n")

    test_image_tokenization_creation()
    print()

    test_image_tokenization_functionality()
    print()

    test_qwen3_vl_2b_integration()
    print()

    test_plugin_integration()
    print()

    test_performance_metrics()
    print()

    print(
        "SUCCESS: All verification tests passed! The Image Tokenization System is successfully integrated with Qwen3-VL-2B."
    )


if __name__ == "__main__":
    run_all_tests()
