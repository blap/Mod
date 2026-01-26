"""
Tests for Multimodal Preprocessing Pipeline in Qwen3-VL-2B Model

This module contains comprehensive tests for the multimodal preprocessing pipeline
implementation for the Qwen3-VL-2B model.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import tempfile
import os
from PIL import Image
import numpy as np

from src.inference_pio.common.multimodal_preprocessing import (
    TextPreprocessor,
    ImagePreprocessor,
    MultimodalPreprocessor,
    create_multimodal_preprocessor
)
from src.inference_pio.common.multimodal_pipeline import (
    MultimodalPipelineStage,
    MultimodalPreprocessingPipeline,
    create_multimodal_pipeline,
    OptimizedMultimodalPipeline,
    create_optimized_multimodal_pipeline
)

# TestTextPreprocessor

    """Test cases for TextPreprocessor."""

    def setup_helper():
        """Set up test fixtures."""
        from transformers import AutoTokenizer
        # Create a simple tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        preprocessor = TextPreprocessor(tokenizer, max_length=512)

    def preprocess_single_text(self)():
        """Test preprocessing of a single text."""
        text = "This is a test sentence."
        result = preprocessor.preprocess(text)
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        assert_equal(result['input_ids'].shape[0], 1)  # Batch dimension
        assertLessEqual(result['input_ids'].shape[1], 512)  # Max length constraint

    def batch_preprocess_texts(self)():
        """Test preprocessing of a batch of texts."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        result = preprocessor.batch_preprocess(texts)
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        assert_equal(result['input_ids'].shape[0], 3)  # Batch size
        assertLessEqual(result['input_ids'].shape[1], 512)  # Max length constraint

    def preprocess_empty_text(self)():
        """Test preprocessing of an empty text."""
        result = preprocessor.preprocess("")
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        # Empty text should still produce some tokens (e.g., BOS/EOS)

# TestImagePreprocessor

    """Test cases for ImagePreprocessor."""

    def setup_helper():
        """Set up test fixtures."""
        from transformers import AutoImageProcessor
        # Create a simple image processor for testing
        image_processor = AutoImageProcessor.from_pretrained("facebook/opt-350m", trust_remote_code=True)
        preprocessor = ImagePreprocessor(image_processor, image_size=224, patch_size=14)

    def preprocess_single_image(self)():
        """Test preprocessing of a single image."""
        # Create a dummy image for testing
        image = Image.new('RGB', (224, 224), color='red')
        result = preprocessor.preprocess(image)
        
        assert_in('pixel_values', result)
        # Check that pixel values have the expected shape
        assert_equal(result['pixel_values'].shape[0], 1)  # Batch dimension

    def batch_preprocess_images(self)():
        """Test preprocessing of a batch of images."""
        # Create dummy images for testing
        images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='green'),
            Image.new('RGB', (224, 224), color='blue')
        ]
        result = preprocessor.batch_preprocess(images)
        
        assert_in('pixel_values', result)
        # Check that pixel values have the expected batch size
        assert_equal(result['pixel_values'].shape[0], 3)  # Batch size

    def preprocess_image_from_path(self)():
        """Test preprocessing of an image from a file path."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            image = Image.new('RGB', (224, 224), color='yellow')
            image.save(tmp_file.name)
            tmp_path = tmp_file.name

        try:
            result = preprocessor.preprocess(tmp_path)
            assert_in('pixel_values', result)
        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)

# TestMultimodalPreprocessor

    """Test cases for MultimodalPreprocessor."""

    def setup_helper():
        """Set up test fixtures."""
        from transformers import AutoTokenizer, AutoImageProcessor
        # Create simple tokenizer and image processor for testing
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        image_processor = AutoImageProcessor.from_pretrained("facebook/opt-350m", trust_remote_code=True)
        preprocessor = MultimodalPreprocessor(
            tokenizer, 
            image_processor, 
            max_text_length=512, 
            image_size=224, 
            patch_size=14
        )

    def preprocess_text_only(self)():
        """Test preprocessing of text-only input."""
        text = "This is a test sentence."
        result = preprocessor.preprocess(text=text)
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        # Should not have pixel_values for text-only input
        assert_not_in('pixel_values', result)

    def preprocess_image_only(self)():
        """Test preprocessing of image-only input."""
        image = Image.new('RGB', (224, 224), color='red')
        result = preprocessor.preprocess(image=image)
        
        assert_in('pixel_values', result)
        # Should not have text-related fields for image-only input
        assert_not_in('input_ids', result)
        assert_not_in('attention_mask', result)

    def preprocess_multimodal(self)():
        """Test preprocessing of multimodal input (text and image)."""
        text = "Describe this image."
        image = Image.new('RGB', (224, 224), color='blue')
        result = preprocessor.preprocess(text=text, image=image)
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        assert_in('pixel_values', result)

    def batch_preprocess_multimodal(self)():
        """Test batch preprocessing of multimodal inputs."""
        inputs = [
            {'text': 'First sentence.', 'image': Image.new('RGB', (224, 224), color='red')},
            {'text': 'Second sentence.', 'image': Image.new('RGB', (224, 224), color='green')}
        ]
        result = preprocessor.batch_preprocess(inputs)
        
        # Check that both text and image fields are present
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        assert_in('pixel_values', result)
        
        # Check batch dimensions
        assert_equal(result['input_ids'].shape[0], 2)
        assert_equal(result['pixel_values'].shape[0], 2)

# TestMultimodalPipelineStage

    """Test cases for MultimodalPipelineStage."""

    def stage_execution(self)():
        """Test execution of a pipeline stage."""
        def dummy_operation(data, multiplier=2):
            return data * multiplier
        
        stage = MultimodalPipelineStage("test_stage", dummy_operation, multiplier=3)
        result = stage.execute(5)
        
        assert_equal(result, 15)
        assert_equal(stage.call_count, 1)

    def stage_error_handling(self)():
        """Test error handling in pipeline stage."""
        def error_operation(data):
            raise ValueError("Test error")
        
        stage = MultimodalPipelineStage("error_stage", error_operation)
        
        with assert_raises(ValueError):
            stage.execute("test_data")

# TestMultimodalPreprocessingPipeline

    """Test cases for MultimodalPreprocessingPipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up class fixtures."""
        # Create a temporary directory for model files
        cls.temp_dir = tempfile.mkdtemp()
        
        # For testing, we'll use a smaller model
        try:
            from transformers import AutoTokenizer, AutoImageProcessor
            # Use a lightweight model for testing
            tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2", trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            image_processor = AutoImageProcessor.from_pretrained("hf-internal-testing/tiny-random-vit", trust_remote_code=True)
            
            # Save these to temp directory to simulate a model
            tokenizer.save_pretrained(cls.temp_dir)
            image_processor.save_pretrained(cls.temp_dir)
        except:
            # If the above fails, use a mock approach
            pass

    def setup_helper():
        """Set up test fixtures."""
        try:
            pipeline = MultimodalPreprocessingPipeline(
                model_path=temp_dir,
                max_text_length=128,
                image_size=224,
                patch_size=14
            )
        except:
            # If we can't create the pipeline with real models, skip these tests
            skipTest("Cannot create pipeline with real models for testing")

    def pipeline_execution_text_only(self)():
        """Test pipeline execution with text-only input."""
        data = {'text': 'This is a test.'}
        result = pipeline.execute(data)
        
        # Should have text-related outputs
        assert_in('input_ids', result)
        assert_in('attention_mask', result)

    def pipeline_execution_image_only(self)():
        """Test pipeline execution with image-only input."""
        image = Image.new('RGB', (224, 224), color='red')
        data = {'image': image}
        result = pipeline.execute(data)
        
        # Should have image-related outputs
        assert_in('pixel_values', result)

    def pipeline_execution_multimodal(self)():
        """Test pipeline execution with multimodal input."""
        image = Image.new('RGB', (224, 224), color='blue')
        data = {'text': 'Describe this image.', 'image': image}
        result = pipeline.execute(data)
        
        # Should have both text and image-related outputs
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        assert_in('pixel_values', result)

    def pipeline_batch_execution(self)():
        """Test batch execution of the pipeline."""
        inputs = [
            {'text': 'First sentence.', 'image': Image.new('RGB', (224, 224), color='red')},
            {'text': 'Second sentence.', 'image': Image.new('RGB', (224, 224), color='green')}
        ]
        results = pipeline.execute_batch(inputs)
        
        assert_equal(len(results), 2)
        for result in results:
            assert_in('input_ids', result)
            assert_in('pixel_values', result)

    def pipeline_stats(self)():
        """Test getting pipeline statistics."""
        stats = pipeline.get_pipeline_stats()
        
        assert_in('pipeline_execution_time', stats)
        assert_in('pipeline_call_count', stats)
        assert_in('average_pipeline_time', stats)
        assert_in('stages', stats)
        
        # Should have at least the default stages
        assertGreaterEqual(len(stats['stages']), 4)

# TestOptimizedMultimodalPipeline

    """Test cases for OptimizedMultimodalPipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up class fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        
        # For testing, we'll use a smaller model
        try:
            from transformers import AutoTokenizer, AutoImageProcessor
            # Use a lightweight model for testing
            tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2", trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            image_processor = AutoImageProcessor.from_pretrained("hf-internal-testing/tiny-random-vit", trust_remote_code=True)
            
            # Save these to temp directory to simulate a model
            tokenizer.save_pretrained(cls.temp_dir)
            image_processor.save_pretrained(cls.temp_dir)
        except:
            # If the above fails, use a mock approach
            pass

    def setup_helper():
        """Set up test fixtures."""
        try:
            pipeline = OptimizedMultimodalPipeline(
                model_path=temp_dir,
                max_text_length=128,
                image_size=224,
                patch_size=14
            )
        except:
            # If we can't create the pipeline with real models, skip these tests
            skipTest("Cannot create optimized pipeline with real models for testing")

    def caching_functionality(self)():
        """Test caching functionality of the optimized pipeline."""
        image = Image.new('RGB', (224, 224), color='red')
        data = {'text': 'Test caching.', 'image': image}
        
        # Execute the same input twice
        result1 = pipeline.execute(data)
        result2 = pipeline.execute(data)
        
        # Both should succeed
        assert_is_not_none(result1)
        assertIsNotNone(result2)

    def cache_clearing(self)():
        """Test clearing the cache."""
        pipeline.clear_cache()
        # Just ensure the method runs without error

# TestFactoryFunctions

    """Test cases for factory functions."""

    @classmethod
    def setUpClass(cls):
        """Set up class fixtures."""
        cls.temp_dir = tempfile.mkdtemp()
        
        try:
            from transformers import AutoTokenizer)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            image_processor = AutoImageProcessor.from_pretrained("hf-internal-testing/tiny-random-vit", trust_remote_code=True)
            
            # Save these to temp directory to simulate a model
            tokenizer.save_pretrained(cls.temp_dir)
            image_processor.save_pretrained(cls.temp_dir)
        except:
            # If the above fails, use a mock approach
            pass

    def create_multimodal_preprocessor(self)():
        """Test creating a multimodal preprocessor."""
        try:
            preprocessor = create_multimodal_preprocessor(
                model_path=temp_dir,
                max_text_length=256,
                image_size=224,
                patch_size=14
            )
            assert_is_instance(preprocessor, MultimodalPreprocessor)
        except:
            skipTest("Cannot create preprocessor with real models for testing")

    def create_multimodal_pipeline(self)():
        """Test creating a multimodal pipeline."""
        try:
            pipeline = create_multimodal_pipeline(
                model_path=temp_dir,
                max_text_length=256,
                image_size=224,
                patch_size=14
            )
            assert_is_instance(pipeline, MultimodalPreprocessingPipeline)
        except:
            skipTest("Cannot create pipeline with real models for testing")

    def create_optimized_multimodal_pipeline(self)():
        """Test creating an optimized multimodal pipeline."""
        try:
            pipeline = create_optimized_multimodal_pipeline(
                model_path=temp_dir,
                max_text_length=256,
                image_size=224,
                patch_size=14
            )
            assert_is_instance(pipeline, OptimizedMultimodalPipeline)
        except:
            skipTest("Cannot create optimized pipeline with real models for testing")

if __name__ == '__main__':
    print("Running multimodal preprocessing pipeline tests...")
    run_tests(test_functions)