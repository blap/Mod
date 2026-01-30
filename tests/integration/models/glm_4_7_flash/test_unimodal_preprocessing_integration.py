"""
Integration Tests for Unimodal Preprocessing Pipeline

This module contains integration tests for the unimodal preprocessing pipeline
implementation with the actual model plugins.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import os
import torch
from transformers import AutoTokenizer

from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.common.unimodal_preprocessing import (
    TextPreprocessor,
    UnimodalPreprocessor,
    create_unimodal_preprocessor
)

# TestUnimodalPreprocessingIntegration

    """Integration tests for unimodal preprocessing with model plugins."""

    def glm_4_7_plugin_has_unimodal_preprocessor_attribute(self)():
        """Test GLM-4-7 plugin has unimodal preprocessor attribute."""
        plugin = GLM_4_7_Plugin()

        # Verify that the plugin has the unimodal preprocessor attribute
        assert_true(hasattr(plugin))
        # Initially should be None
        assert_is_none(plugin._unimodal_preprocessor)

    def qwen3_4b_instruct_plugin_has_unimodal_preprocessor_attribute(self)():
        """Test Qwen3-4b-instruct plugin has unimodal preprocessor attribute."""
        plugin = Qwen3_4B_Instruct_2507_Plugin()

        # Verify that the plugin has the unimodal preprocessor attribute
        assert_true(hasattr(plugin))
        # Initially should be None
        assertIsNone(plugin._unimodal_preprocessor)

    def qwen3_coder_30b_plugin_has_unimodal_preprocessor_attribute(self)():
        """Test Qwen3-coder-30b plugin has unimodal preprocessor attribute."""
        plugin = Qwen3_Coder_30B_Plugin()

        # Verify that the plugin has the unimodal preprocessor attribute
        assert_true(hasattr(plugin))
        # Initially should be None
        assertIsNone(plugin._unimodal_preprocessor)

    def tokenize_with_unimodal_preprocessor_glm47(self)():
        """Test tokenize method with unimodal preprocessor for GLM-4-7."""
        plugin = GLM_4_7_Plugin()
        
        # Mock the config and model
        plugin._config = Mock()
        plugin._config.model_path = 'dummy_path'
        plugin._config.max_text_length = 512
        plugin._model = Mock()
        plugin._model.get_tokenizer = Mock(return_value=Mock())
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        plugin._model.get_tokenizer.return_value = mock_tokenizer
        
        # Mock the unimodal preprocessor
        mock_unimodal_preprocessor = Mock(spec=UnimodalPreprocessor)
        mock_unimodal_preprocessor.preprocess = Mock(return_value={
            'input_ids': torch.tensor([[10, 20, 30, 40, 50]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        })
        plugin._unimodal_preprocessor = mock_unimodal_preprocessor
        
        # Test tokenize method
        result = plugin.tokenize("Hello world!")
        
        # Verify that the unimodal preprocessor was used
        mock_unimodal_preprocessor.preprocess.assert_called_once_with(
            "Hello world!",
            return_tensors="pt",
            model_type='glm47'
        )
        
        # Verify the result
        assert_equal(result['input_ids'][0][0].item(), 10)

    def tokenize_with_unimodal_preprocessor_qwen3_4b(self)():
        """Test tokenize method with unimodal preprocessor for Qwen3-4b."""
        plugin = Qwen3_4B_Instruct_2507_Plugin()
        
        # Mock the config and model
        plugin._config = Mock()
        plugin._config.model_path = 'dummy_path'
        plugin._config.max_text_length = 1024
        plugin._model = Mock()
        plugin._model.get_tokenizer = Mock(return_value=Mock())
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        plugin._model.get_tokenizer.return_value = mock_tokenizer
        
        # Mock the unimodal preprocessor
        mock_unimodal_preprocessor = Mock(spec=UnimodalPreprocessor)
        mock_unimodal_preprocessor.preprocess = Mock(return_value={
            'input_ids': torch.tensor([[10, 20, 30, 40, 50]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        })
        plugin._unimodal_preprocessor = mock_unimodal_preprocessor
        
        # Test tokenize method
        result = plugin.tokenize("Hello world!")
        
        # Verify that the unimodal preprocessor was used
        mock_unimodal_preprocessor.preprocess.assert_called_once_with(
            "Hello world!",
            return_tensors="pt",
            model_type='qwen3_4b'
        )
        
        # Verify the result
        assert_equal(result['input_ids'][0][0].item(), 10)

    def tokenize_with_unimodal_preprocessor_qwen3_coder(self)():
        """Test tokenize method with unimodal preprocessor for Qwen3-coder."""
        plugin = Qwen3_Coder_30B_Plugin()
        
        # Mock the config and model
        plugin._config = Mock()
        plugin._config.model_path = 'dummy_path'
        plugin._config.max_text_length = 2048
        plugin._model = Mock()
        plugin._model.get_tokenizer = Mock(return_value=Mock())
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        plugin._model.get_tokenizer.return_value = mock_tokenizer
        
        # Mock the unimodal preprocessor
        mock_unimodal_preprocessor = Mock(spec=UnimodalPreprocessor)
        mock_unimodal_preprocessor.preprocess = Mock(return_value={
            'input_ids': torch.tensor([[10, 20, 30, 40, 50]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        })
        plugin._unimodal_preprocessor = mock_unimodal_preprocessor
        
        # Test tokenize method
        result = plugin.tokenize("Hello world!")
        
        # Verify that the unimodal preprocessor was used
        mock_unimodal_preprocessor.preprocess.assert_called_once_with(
            "Hello world!",
            return_tensors="pt",
            model_type='qwen3_coder'
        )
        
        # Verify the result
        assert_equal(result['input_ids'][0][0].item(), 10)

    def tokenize_without_unimodal_preprocessor(self)():
        """Test tokenize method without unimodal preprocessor falls back to regular tokenizer."""
        plugin = GLM_4_7_Plugin()
        
        # Mock the config and model
        plugin._config = Mock()
        plugin._config.model_path = 'dummy_path'
        plugin._config.max_text_length = 512
        plugin._model = Mock()
        plugin._model.get_tokenizer = Mock(return_value=Mock())
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        plugin._model.get_tokenizer.return_value = mock_tokenizer
        
        # Set unimodal preprocessor to None
        plugin._unimodal_preprocessor = None
        
        # Test tokenize method
        result = plugin.tokenize("Hello world!", max_length=128)
        
        # Verify that the regular tokenizer was used
        mock_tokenizer.assert_called_once_with(
            "Hello world!",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

# TestUnimodalPreprocessingPerformance

    """Performance tests for unimodal preprocessing."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        # Create a mock tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
        tokenizer.pad_token = tokenizer.eos_token
        preprocessor = UnimodalPreprocessor(tokenizer, max_text_length=512)

    def preprocessing_performance_stats(self)():
        """Test that performance statistics are correctly tracked."""
        # Get initial stats
        initial_stats = preprocessor.get_performance_stats()
        initial_calls = initial_stats['num_preprocessing_calls']
        
        # Perform some preprocessing operations
        for i in range(5):
            text = f"This is test text {i}." * 20  # Make it reasonably long
            preprocessor.preprocess(text)
        
        # Get updated stats
        updated_stats = preprocessor.get_performance_stats()
        
        # Verify that the call count increased
        assert_equal(updated_stats['num_preprocessing_calls'], initial_calls + 5)
        
        # Verify that average time is calculated (though it might be 0 for fast operations)
        assertGreaterEqual(updated_stats['average_preprocessing_time'], 0.0)
        assertGreaterEqual(updated_stats['total_preprocessing_time'], 0.0)

    def batch_preprocessing_performance(self)():
        """Test batch preprocessing performance."""
        # Get initial stats
        initial_stats = preprocessor.get_performance_stats()
        initial_calls = initial_stats['num_preprocessing_calls']
        
        # Perform batch preprocessing
        texts = [f"This is test text {i}." * 10 for i in range(10)]
        preprocessor.batch_preprocess(texts)
        
        # Get updated stats
        updated_stats = preprocessor.get_performance_stats()
        
        # Verify that the call count increased
        assert_equal(updated_stats['num_preprocessing_calls'], initial_calls + 1)

if __name__ == '__main__':
    # Run the tests
    run_tests(test_functions)