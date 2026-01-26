"""
Tests for Image Tokenization Integration with Qwen3-VL-2B Model

This module contains tests for the integration of the efficient image tokenization
system with the Qwen3-VL-2B model.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from PIL import Image
import tempfile
import os

from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.common.image_tokenization import (
    ImageTokenizationConfig,
    ImageTokenizer,
    create_image_tokenizer
)

# TestQwen3VL2BImageTokenizationIntegration

    """Test cases for image tokenization integration with Qwen3-VL-2B model."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        # Create a minimal config for testing
        config = Qwen3VL2BConfig()
        # Override model path to avoid loading actual model during tests
        config.model_path = "dummy_path"
        
        # Disable some heavy optimizations for faster testing
        config.use_flash_attention_2 = False
        config.use_cuda_kernels = False
        config.enable_disk_offloading = False
        config.enable_intelligent_pagination = False
        config.enable_tensor_parallelism = False
        config.use_multimodal_attention = False

    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained')
    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained')
    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained')
    def model_initialization_with_image_tokenization(self, mock_image_proc, mock_tokenizer, mock_model)():
        """Test that the model initializes with image tokenization system."""
        # Mock the model components
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = '</s>'
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_image_proc_instance = Mock()
        mock_image_proc.return_value = mock_image_proc_instance

        # Create the model
        model = Qwen3VL2BModel(config)

        # Verify that image tokenization was initialized
        assert_is_not_none(model._image_tokenizer)
        assert_is_instance(model._image_tokenizer)

    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained')
    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained')
    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained')
    def image_tokenization_disabled(self, mock_image_proc)():
        """Test that image tokenization can be disabled."""
        # Disable image tokenization in config
        config.enable_image_tokenization = False
        
        # Mock the model components
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = '</s>'
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_image_proc_instance = Mock()
        mock_image_proc.return_value = mock_image_proc_instance

        # Create the model
        model = Qwen3VL2BModel(config)

        # Verify that image tokenization was not initialized
        assert_is_none(model._image_tokenizer)

    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained')
    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained')
    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained')
    def image_tokenization_with_different_configs(self)():
        """Test image tokenization with different configuration options."""
        # Modify config for different settings
        config.image_size = 224
        config.patch_size = 16
        config.max_image_tokens = 256
        config.enable_image_quantization = True
        config.image_quantization_bits = 8
        config.enable_image_compression = True
        config.image_compression_ratio = 0.7

        # Mock the model components
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = '</s>'
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_image_proc_instance = Mock()
        mock_image_proc.return_value = mock_image_proc_instance

        # Create the model
        model = Qwen3VL2BModel(config)

        # Verify that image tokenization was initialized with correct config
        assert_is_not_none(model._image_tokenizer)
        assert_equal(model._image_tokenizer.config.image_size)
        assert_equal(model._image_tokenizer.config.patch_size, 16)
        assert_equal(model._image_tokenizer.config.max_image_tokens, 256)
        assert_true(model._image_tokenizer.config.enable_quantization)
        assert_equal(model._image_tokenizer.config.quantization_bits)
        assert_true(model._image_tokenizer.config.enable_compression)
        assert_equal(model._image_tokenizer.config.compression_ratio)

    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained')
    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained')
    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained')
    def image_tokenization_attached_to_model(self, mock_image_proc, mock_tokenizer, mock_model)():
        """Test that image tokenizer is attached to the model."""
        # Mock the model components
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = '</s>'
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_image_proc_instance = Mock()
        mock_image_proc.return_value = mock_image_proc_instance

        # Create the model
        model = Qwen3VL2BModel(config)

        # Verify that image tokenizer is attached to the model
        assert_true(hasattr(model._model))
        assert_is_instance(model._model.image_tokenizer, ImageTokenizer)

# TestImageTokenizationMethods

    """Test cases for image tokenization methods."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = ImageTokenizationConfig(
            image_size=224,
            patch_size=16,
            max_image_tokens=256,
            token_dim=512
        )
        tokenizer = ImageTokenizer(config)

    def tokenize_pil_image(self)():
        """Test tokenizing a PIL image."""
        image = Image.new('RGB', (224, 224), color='red')
        
        result = tokenizer.tokenize(image)
        
        assert_in('pixel_values', result)
        assert_equal(result['pixel_values'].shape, (1))

    def tokenize_image_path(self)():
        """Test tokenizing an image from a file path."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image = Image.new('RGB', (224, 224), color='blue')
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            result = tokenizer.tokenize(tmp_path)
            assert_in('pixel_values', result)
            assert_equal(result['pixel_values'].shape, (1))
        finally:
            os.unlink(tmp_path)

    def batch_tokenize_images(self)():
        """Test batch tokenizing multiple images."""
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
            Image.new('RGB', (224, 224), color='blue')
        ]
        
        result = tokenizer.batch_tokenize(images)
        
        assert_in('pixel_values', result)
        assert_equal(result['pixel_values'].shape, (3))

    def tokenize_with_quantization(self)():
        """Test tokenizing with quantization enabled."""
        # Create config with quantization enabled
        config = ImageTokenizationConfig(
            image_size=224,
            patch_size=16,
            max_image_tokens=256,
            token_dim=512,
            enable_quantization=True,
            quantization_bits=8
        )
        tokenizer = ImageTokenizer(config)
        
        image = Image.new('RGB', (224, 224), color='red')
        result = tokenizer.tokenize(image)
        
        assert_in('pixel_values', result)
        assert_equal(result['pixel_values'].shape, (1))

    def tokenize_with_compression(self)():
        """Test tokenizing with compression enabled."""
        # Create config with compression enabled
        config = ImageTokenizationConfig(
            image_size=224,
            patch_size=16,
            max_image_tokens=256,
            token_dim=512,
            enable_compression=True,
            compression_ratio=0.5
        )
        tokenizer = ImageTokenizer(config)
        
        image = Image.new('RGB', (224, 224), color='red')
        result = tokenizer.tokenize(image)
        
        assert_in('pixel_values', result)
        # The shape might be different due to compression
        assert_equal(result['pixel_values'].shape[0], 1)  # Batch dimension
        assert_equal(result['pixel_values'].shape[1], 3)  # Channel dimension

# TestPerformanceMetrics

    """Test cases for performance metrics."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = ImageTokenizationConfig(
            image_size=224,
            patch_size=16,
            max_image_tokens=256,
            token_dim=512
        )
        tokenizer = ImageTokenizer(config)

    def performance_tracking(self)():
        """Test that performance metrics are tracked."""
        # Initially, no calls should have been made
        initial_stats = tokenizer.get_performance_stats()
        assert_equal(initial_stats['num_tokenization_calls'], 0)
        
        # Perform a tokenization
        image = Image.new('RGB', (224, 224), color='red')
        tokenizer.tokenize(image)
        
        # Now there should be one call
        stats = tokenizer.get_performance_stats()
        assert_equal(stats['num_tokenization_calls'], 1)
        
        # Perform another tokenization
        tokenizer.tokenize(image)
        
        # Now there should be two calls
        stats = tokenizer.get_performance_stats()
        assert_equal(stats['num_tokenization_calls'], 2)

    def performance_reset(self)():
        """Test that performance metrics can be reset."""
        # Perform some tokenizations
        image = Image.new('RGB', (224, 224), color='red')
        tokenizer.tokenize(image)
        tokenizer.tokenize(image)
        
        # Verify metrics are tracked
        stats = tokenizer.get_performance_stats()
        assert_equal(stats['num_tokenization_calls'], 2)
        
        # Reset metrics
        tokenizer.reset_performance_stats()
        
        # Verify metrics are reset
        reset_stats = tokenizer.get_performance_stats()
        assert_equal(reset_stats['num_tokenization_calls'], 0)
        assert_equal(reset_stats['total_tokenization_time'], 0.0)

if __name__ == '__main__':
    run_tests(test_functions)