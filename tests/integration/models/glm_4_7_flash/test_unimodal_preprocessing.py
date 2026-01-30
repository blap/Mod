"""
Tests for Unimodal Preprocessing Pipeline

This module contains comprehensive tests for the unimodal preprocessing pipeline
implementation for language models like GLM-4-7, Qwen3-4b-instruct-2507, and Qwen3-coder-30b.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import os
import torch
from transformers import AutoTokenizer

from src.inference_pio.common.unimodal_preprocessing import (
    TextPreprocessor,
    UnimodalPreprocessor,
    create_unimodal_preprocessor,
    apply_unimodal_preprocessing_to_model
)

# TestTextPreprocessor

    """Test cases for the TextPreprocessor class."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        # Create a mock tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
        tokenizer.pad_token = tokenizer.eos_token
        preprocessor = TextPreprocessor(tokenizer, max_length=512)

    def preprocess_single_text(self)():
        """Test preprocessing of a single text."""
        text = "Hello world!"
        result = preprocessor.preprocess(text)
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        assert_equal(result['input_ids'].shape[0], 1)  # Batch dimension
        assert_equal(result['attention_mask'].shape[0], 1)  # Batch dimension

    def preprocess_empty_text(self)():
        """Test preprocessing of an empty text."""
        text = ""
        result = preprocessor.preprocess(text)
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)

    def preprocess_long_text(self)():
        """Test preprocessing of a long text with truncation."""
        long_text = "Hello world! " * 1000  # Very long text
        result = preprocessor.preprocess(long_text, truncation=True)
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        # Should be truncated to max_length
        assertLessEqual(result['input_ids'].shape[1], 512)

    def batch_preprocess(self)():
        """Test preprocessing of a batch of texts."""
        texts = ["Hello world!", "How are you?", "Fine, thanks!"]
        result = preprocessor.batch_preprocess(texts)
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        assert_equal(result['input_ids'].shape[0], len(texts))
        assert_equal(result['attention_mask'].shape[0], len(texts))

    def batch_preprocess_empty_list(self)():
        """Test preprocessing of an empty list of texts."""
        texts = []
        result = preprocessor.batch_preprocess(texts)
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        assert_equal(result['input_ids'].shape[0], 0)

# TestUnimodalPreprocessor

    """Test cases for the UnimodalPreprocessor class."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        # Create a mock tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
        tokenizer.pad_token = tokenizer.eos_token
        preprocessor = UnimodalPreprocessor(tokenizer, max_text_length=512)

    def preprocess_text(self)():
        """Test preprocessing of text."""
        text = "Hello world!"
        result = preprocessor.preprocess(text)
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        assert_equal(result['input_ids'].shape[0], 1)
        assert_equal(result['attention_mask'].shape[0], 1)

    def batch_preprocess_texts(self)():
        """Test preprocessing of a batch of texts."""
        texts = ["Hello world!", "How are you?", "Fine, thanks!"]
        result = preprocessor.batch_preprocess(texts)
        
        assert_in('input_ids', result)
        assert_in('attention_mask', result)
        assert_equal(result['input_ids'].shape[0], len(texts))
        assert_equal(result['attention_mask'].shape[0], len(texts))

    def get_performance_stats(self)():
        """Test getting performance statistics."""
        stats = preprocessor.get_performance_stats()
        
        assert_in('total_preprocessing_time', stats)
        assert_in('num_preprocessing_calls', stats)
        assert_in('average_preprocessing_time', stats)
        assert_equal(stats['num_preprocessing_calls'], 0)

    def reset_performance_stats(self)():
        """Test resetting performance statistics."""
        # First, make some preprocessing calls to increase the count
        preprocessor.preprocess("Hello world!")
        preprocessor.preprocess("Another text.")
        
        # Check that stats are not zero
        stats = preprocessor.get_performance_stats()
        assert_greater(stats['num_preprocessing_calls'], 0)
        
        # Reset stats
        preprocessor.reset_performance_stats()
        
        # Check that stats are reset
        stats = preprocessor.get_performance_stats()
        assert_equal(stats['num_preprocessing_calls'], 0)
        assert_equal(stats['total_preprocessing_time'], 0.0)
        assert_equal(stats['average_preprocessing_time'], 0.0)

# TestUnimodalPreprocessorFactory

    """Test cases for the unimodal preprocessor factory functions."""

    @patch('src.inference_pio.common.unimodal_preprocessing.AutoTokenizer.from_pretrained')
    def create_unimodal_preprocessor(self, mock_from_pretrained)():
        """Test creating a unimodal preprocessor."""
        # Mock the tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_from_pretrained.return_value = mock_tokenizer
        
        preprocessor = create_unimodal_preprocessor('dummy_path', max_text_length=256)
        
        assert_is_instance(preprocessor, UnimodalPreprocessor)
        assert_equal(preprocessor.text_preprocessor.max_length, 256)
        mock_from_pretrained.assert_called_once_with('dummy_path', trust_remote_code=True)

    @patch('src.inference_pio.common.unimodal_preprocessing.AutoTokenizer.from_pretrained')
    def create_unimodal_preprocessor_with_model_type(self, mock_from_pretrained)():
        """Test creating a unimodal preprocessor with model type."""
        # Mock the tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_from_pretrained.return_value = mock_tokenizer
        
        preprocessor = create_unimodal_preprocessor('dummy_path', max_text_length=256, model_type='glm47')
        
        assert_is_instance(preprocessor, UnimodalPreprocessor)
        mock_from_pretrained.assert_called_once_with('dummy_path', trust_remote_code=True)

    def apply_unimodal_preprocessing_to_model(self)():
        """Test applying unimodal preprocessing to a model."""
        # Create a mock model
        mock_model = Mock()
        
        # Create a mock preprocessor
        mock_preprocessor = Mock()
        
        # Apply preprocessing to the model
        result_model = apply_unimodal_preprocessing_to_model(mock_model, mock_preprocessor)
        
        # Check that the preprocessor was attached to the model
        assert_equal(result_model, mock_model)
        assert_equal(result_model.preprocessor, mock_preprocessor)
        assert_true(hasattr(result_model))

# TestModelIntegration

    """Test cases for integration with model plugins."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        # Create a temporary directory for model files
        temp_dir = tempfile.mkdtemp()
        
        # Create a minimal tokenizer file for testing
        tokenizer_path = os.path.join(temp_dir, 'tokenizer.json')
        # In a real scenario, we would create a proper tokenizer file
        # For now, we'll just use a dummy one
        with open(tokenizer_path, 'w') as f:
            f.write('{}')

    def cleanup_helper():
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(temp_dir)

    @unittest.skip("Skipping integration test that requires real model files")
    def integration_with_real_model(self)():
        """Test integration with a real model (requires actual model files)."""
        # This test would require actual model files to be downloaded
        # For now, we'll skip it
        pass

if __name__ == '__main__':
    # Run the tests
    run_tests(test_functions)