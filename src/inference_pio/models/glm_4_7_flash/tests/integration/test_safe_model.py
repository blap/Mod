"""
Tests for GLM-4.7 Safe Model - Validating Unimodal Operation

These tests ensure that the GLM-4.7 model correctly operates in unimodal (text-only) mode
and properly rejects any multimodal inputs that would be inappropriate for this model.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import numpy as np
from src.inference_pio.models.glm_4_7.safe_model import GLM47SafeModel
from src.inference_pio.models.glm_4_7.config import GLM47Config

# TestGLM47SafeModel

    """Test suite for GLM-4.7 Safe Model."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = GLM47Config()
        # Ensure the model is configured as text-only
        config.supported_modalities = ["text"]

    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoTokenizer.from_pretrained')
    def initialization_with_text_only_config(self, mock_tokenizer, mock_model)():
        """Test that the model initializes correctly with text-only configuration."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        
        # This should not raise an exception
        model = GLM47SafeModel(config)
        
        assert_is_not_none(model)
        assert_equal(model.config)

    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoTokenizer.from_pretrained')
    def validation_of_text_inputs(self, mock_tokenizer, mock_model)():
        """Test that text inputs are validated correctly."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        
        model = GLM47SafeModel(config)
        
        # Create a mock text input (should be valid)
        text_input = torch.randint(0, 1000, (1, 10))  # Batch of 1, sequence length 10
        
        # This should return True (valid unimodal input)
        is_valid = model._validate_unimodal_operation(text_input)
        assert_true(is_valid)

    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoTokenizer.from_pretrained')
    def validation_rejects_vision_inputs(self)():
        """Test that vision inputs are rejected."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        
        model = GLM47SafeModel(config)
        
        # Create a mock vision input (4D tensor, which should be rejected)
        vision_input = torch.randn(1, 3, 224, 224)  # Batch of 1, 3 channels, 224x224 image
        
        # This should return False (invalid multimodal input)
        is_valid = model._validate_unimodal_operation(vision_input)
        assert_false(is_valid)

    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoTokenizer.from_pretrained')
    def validation_rejects_multimodal_dict_inputs(self)():
        """Test that multimodal dictionary inputs are rejected."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        
        model = GLM47SafeModel(config)
        
        # Create a mock multimodal input dictionary (should be rejected)
        multimodal_input = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'pixel_values': torch.randn(1, 3, 224, 224)  # Vision component
        }
        
        # This should return False (invalid multimodal input)
        is_valid = model._validate_unimodal_operation(multimodal_input)
        assert_false(is_valid)

    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoTokenizer.from_pretrained')
    def validation_accepts_text_dict_inputs(self)():
        """Test that text-only dictionary inputs are accepted."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        
        model = GLM47SafeModel(config)
        
        # Create a mock text-only input dictionary (should be accepted)
        text_input = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10)
        }
        
        # This should return True (valid unimodal input)
        is_valid = model._validate_unimodal_operation(text_input)
        assert_true(is_valid)

    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoTokenizer.from_pretrained')
    def model_rejects_multimodal_config(self)():
        """Test that the model rejects configurations with multimodal support."""
        # Modify config to include vision modality
        bad_config = GLM47Config()
        bad_config.supported_modalities = ["text", "vision"]  # This should cause an error
        
        # Attempting to create the model with multimodal config should raise an error
        with assert_raises(ValueError):
            GLM47SafeModel(bad_config)

    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoTokenizer.from_pretrained')
    def forward_pass_with_valid_input(self, mock_tokenizer, mock_model)():
        """Test forward pass with valid text input."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.return_value = torch.randn(1, 10, 512)  # Mock output
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        
        model = GLM47SafeModel(config)
        
        # Valid text input
        input_ids = torch.randint(0, 1000, (1, 10))
        
        # This should work without raising an exception
        output = model(input_ids=input_ids)
        
        # Verify the mock was called
        assert_is_not_none(output)

    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.glm_4_7.safe_model.AutoTokenizer.from_pretrained')
    def forward_pass_rejects_multimodal_input(self)():
        """Test that forward pass rejects multimodal input."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_tokenizer_instance = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftext|>"
        
        model = GLM47SafeModel(config)
        
        # Invalid multimodal input
        multimodal_input = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'pixel_values': torch.randn(1, 3, 224, 224)
        }
        
        # This should raise a ValueError
        with assert_raises(ValueError):
            model(**multimodal_input)

if __name__ == '__main__':
    run_tests(test_functions)