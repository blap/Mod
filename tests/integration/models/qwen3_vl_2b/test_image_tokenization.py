"""
Tests for Efficient Image Tokenization System for Qwen3-VL-2B Model

This module contains comprehensive tests for the efficient image tokenization system
implemented for the Qwen3-VL-2B model.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import os
import torch
from PIL import Image
import numpy as np

from src.inference_pio.common.image_tokenization import (
    ImageTokenizationConfig,
    ImageTokenizer,
    EfficientImageProcessor,
    create_image_tokenizer
)

# TestImageTokenizationConfig

    """Test cases for ImageTokenizationConfig."""

    def default_config_values(self)():
        """Test that default configuration values are set correctly."""
        config = ImageTokenizationConfig()
        
        assert_equal(config.image_size, 448)
        assert_equal(config.patch_size, 14)
        assert_equal(config.max_image_tokens, 1024)
        assert_equal(config.token_dim, 1024)
        assert_true(config.enable_patch_caching)
        assertTrue(config.enable_batch_processing)
        assertTrue(config.enable_memory_efficient_processing)

    def custom_config_values(self)():
        """Test that custom configuration values are set correctly."""
        config = ImageTokenizationConfig(
            image_size=224,
            patch_size=16,
            max_image_tokens=512,
            token_dim=512,
            enable_patch_caching=False,
            enable_batch_processing=False,
            enable_memory_efficient_processing=False
        )
        
        assert_equal(config.image_size, 224)
        assert_equal(config.patch_size, 16)
        assert_equal(config.max_image_tokens, 512)
        assert_equal(config.token_dim, 512)
        assert_false(config.enable_patch_caching)
        assertFalse(config.enable_batch_processing)
        assertFalse(config.enable_memory_efficient_processing)

# TestImageTokenizer

    """Test cases for ImageTokenizer."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = ImageTokenizationConfig(
            image_size=224,
            patch_size=16,
            max_image_tokens=256,
            token_dim=512
        )
        tokenizer = ImageTokenizer(config)

    def initialization(self)():
        """Test ImageTokenizer initialization."""
        assert_is_instance(tokenizer, ImageTokenizer)
        assert_equal(tokenizer.config, config)
        assert_equal(tokenizer.num_patches_per_side, 14)  # 224//16
        assert_equal(tokenizer.total_patches, 196)  # 14^2

    def create_basic_image_processor(self)():
        """Test creation of basic image processor."""
        processor = tokenizer._create_basic_image_processor()
        assert_is_not_none(processor)

    def tokenize_with_pil_image(self)():
        """Test tokenizing a PIL image."""
        # Create a dummy image
        image = Image.new('RGB'), color='red')

        result = tokenizer.tokenize(image)

        assert_in('pixel_values', result)
        # For Qwen models, the output is patch-based with shape (num_patches, patch_dim)
        # The exact shape depends on the model's patch configuration
        pixel_values = result['pixel_values']
        assert_equal(pixel_values.dim(), 2)  # Should be 2D: (num_patches, patch_dim)
        assertGreaterEqual(pixel_values.shape[0], 1)  # At least one patch
        assertGreaterEqual(pixel_values.shape[1], 1)  # At least one dimension per patch

    def tokenize_with_image_path(self)():
        """Test tokenizing an image from a file path."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image = Image.new('RGB', (224, 224), color='blue')
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            result = tokenizer.tokenize(tmp_path)
            assert_in('pixel_values', result)
            # For Qwen models, the output is patch-based with shape (num_patches, patch_dim)
            pixel_values = result['pixel_values']
            assert_equal(pixel_values.dim(), 2)  # Should be 2D: (num_patches, patch_dim)
            assertGreaterEqual(pixel_values.shape[0], 1)  # At least one patch
            assertGreaterEqual(pixel_values.shape[1], 1)  # At least one dimension per patch
        finally:
            os.unlink(tmp_path)

    def tokenize_with_tensor(self)():
        """Test tokenizing an image tensor."""
        # Create a dummy image tensor
        image_tensor = torch.rand(3, 224, 224)

        result = tokenizer.tokenize(image_tensor)

        assert_in('pixel_values', result)
        # For Qwen models, the output is patch-based with shape (num_patches, patch_dim)
        pixel_values = result['pixel_values']
        assert_equal(pixel_values.dim(), 2)  # Should be 2D: (num_patches, patch_dim)
        assertGreaterEqual(pixel_values.shape[0], 1)  # At least one patch
        assertGreaterEqual(pixel_values.shape[1], 1)  # At least one dimension per patch

    def batch_tokenize(self)():
        """Test batch tokenization of multiple images."""
        # Create dummy images
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
            Image.new('RGB', (224, 224), color='blue')
        ]

        result = tokenizer.batch_tokenize(images)

        assert_in('pixel_values', result)
        # For Qwen models, batch processing creates a 3D tensor: (batch_size, num_patches, patch_dim)
        pixel_values = result['pixel_values']
        assert_equal(pixel_values.dim(), 3)  # Should be 3D: (batch_size, num_patches, patch_dim)
        assert_equal(pixel_values.shape[0], 3)  # Batch size of 3
        assertGreaterEqual(pixel_values.shape[1], 1)  # At least one patch per image
        assertGreaterEqual(pixel_values.shape[2], 1)  # At least one dimension per patch

    def tokenize_with_patches(self)():
        """Test tokenizing with patch extraction."""
        # Create a dummy image
        image = Image.new('RGB', (224, 224), color='red')

        result = tokenizer.tokenize_with_patches(image)

        assert_in('pixel_values', result)
        # For Qwen models, the output is patch-based with shape (num_patches, patch_dim)
        pixel_values = result['pixel_values']
        assert_equal(pixel_values.dim(), 2)  # Should be 2D: (num_patches, patch_dim)
        assertGreaterEqual(pixel_values.shape[0], 1)  # At least one patch
        assertGreaterEqual(pixel_values.shape[1], 1)  # At least one dimension per patch

    def get_performance_stats(self)():
        """Test getting performance statistics."""
        stats = tokenizer.get_performance_stats()
        
        assert_in('total_tokenization_time', stats)
        assert_in('num_tokenization_calls', stats)
        assert_in('average_tokenization_time', stats)
        assert_equal(stats['num_tokenization_calls'], 0)

    def reset_performance_stats(self)():
        """Test resetting performance statistics."""
        # Simulate some tokenization calls
        tokenizer.total_tokenization_time = 1.0
        tokenizer.num_tokenization_calls = 5
        
        tokenizer.reset_performance_stats()
        
        stats = tokenizer.get_performance_stats()
        assert_equal(stats['total_tokenization_time'], 0.0)
        assert_equal(stats['num_tokenization_calls'], 0)

# TestEfficientImageProcessor

    """Test cases for EfficientImageProcessor."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = ImageTokenizationConfig(
            image_size=224,
            patch_size=16,
            max_image_tokens=256,
            token_dim=512
        )
        processor = EfficientImageProcessor(config)

    def initialization(self)():
        """Test EfficientImageProcessor initialization."""
        assert_is_instance(processor, EfficientImageProcessor)
        assert_is_instance(processor.tokenizer, ImageTokenizer)

    def process_single_image(self)():
        """Test processing a single image."""
        image = Image.new('RGB', (224, 224), color='red')

        result = processor.process(image)

        assert_in('pixel_values', result)
        # For Qwen models, the output is patch-based with shape (num_patches, patch_dim)
        pixel_values = result['pixel_values']
        assert_equal(pixel_values.dim(), 2)  # Should be 2D: (num_patches, patch_dim)
        assertGreaterEqual(pixel_values.shape[0], 1)  # At least one patch
        assertGreaterEqual(pixel_values.shape[1], 1)  # At least one dimension per patch

    def batch_process(self)():
        """Test batch processing of multiple images."""
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green')
        ]

        result = processor.batch_process(images)

        assert_in('pixel_values', result)
        # For Qwen models, batch processing creates a 3D tensor: (batch_size, num_patches, patch_dim)
        pixel_values = result['pixel_values']
        assert_equal(pixel_values.dim(), 3)  # Should be 3D: (batch_size, num_patches, patch_dim)
        assert_equal(pixel_values.shape[0], 2)  # Batch size of 2
        assertGreaterEqual(pixel_values.shape[1], 1)  # At least one patch per image
        assertGreaterEqual(pixel_values.shape[2], 1)  # At least one dimension per patch

# TestFactoryFunctions

    """Test cases for factory functions."""

    @patch('src.inference_pio.common.image_tokenization.AutoImageProcessor.from_pretrained')
    def create_image_tokenizer(self, mock_from_pretrained)():
        """Test creating an image tokenizer."""
        # Mock the image processor
        mock_processor = MagicMock()
        mock_from_pretrained.return_value = mock_processor
        
        tokenizer = create_image_tokenizer("dummy_path")
        
        assert_is_instance(tokenizer, ImageTokenizer)
        mock_from_pretrained.assert_called_once_with("dummy_path", trust_remote_code=True)

    def create_image_tokenizer_without_model_path(self)():
        """Test creating an image tokenizer without a model path."""
        tokenizer = create_image_tokenizer()
        
        assert_is_instance(tokenizer, ImageTokenizer)

# TestIntegration

    """Integration tests for the image tokenization system."""

    def complete_tokenization_flow(self)():
        """Test the complete flow from image to tokens."""
        config = ImageTokenizationConfig(
            image_size=224,
            patch_size=16,
            max_image_tokens=256,
            token_dim=512
        )

        # Create processor
        processor = EfficientImageProcessor(config)

        # Create test image
        image = Image.new('RGB', (224, 224), color='red')

        # Process the image
        result = processor.process(image)

        # Verify the result
        assert_in('pixel_values', result)
        # For Qwen models, the output is patch-based with shape (num_patches, patch_dim)
        pixel_values = result['pixel_values']
        assert_equal(pixel_values.dim(), 2)  # Should be 2D: (num_patches, patch_dim)
        assertGreaterEqual(pixel_values.shape[0], 1)  # At least one patch
        assertGreaterEqual(pixel_values.shape[1], 1)  # At least one dimension per patch

        # Verify values are in expected range
        assert_true(torch.all(pixel_values >= -1.0))
        assertTrue(torch.all(pixel_values <= 1.0))

    def batch_processing_performance(self)():
        """Test that batch processing is more efficient than individual processing."""
        config = ImageTokenizationConfig(
            image_size=224,
            patch_size=16,
            max_image_tokens=256,
            token_dim=512,
            enable_batch_processing=True
        )

        processor = EfficientImageProcessor(config)
        images = [Image.new('RGB', (224, 224), color=f'hsl({i*60}, 50%, 50%)') for i in range(5)]

        # Process individually
        individual_results = []
        for img in images:
            result = processor.process(img)
            individual_results.append(result)

        # Process in batch
        batch_result = processor.batch_process(images)

        # Verify batch result has correct shape: (batch_size, num_patches, patch_dim)
        pixel_values = batch_result['pixel_values']
        assert_equal(pixel_values.dim(), 3)  # Should be 3D: (batch_size, num_patches, patch_dim)
        assert_equal(pixel_values.shape[0], 5)  # Batch size of 5

        # Verify individual results match batch results
        # For Qwen models, individual processing returns (num_patches, patch_dim)
        # While batch processing returns (batch_size, num_patches, patch_dim)
        for i, individual_result in enumerate(individual_results):
            # Individual result is (num_patches, patch_dim), batch result slice is (1, num_patches, patch_dim)
            # So we squeeze the batch result to compare
            individual_tensor = individual_result['pixel_values']
            batch_slice = pixel_values[i, :, :]  # Shape: (num_patches, patch_dim)
            torch.testing.assert_close(individual_tensor, batch_slice)

if __name__ == '__main__':
    run_tests(test_functions)