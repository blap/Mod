"""
Test suite for NAS-enabled models.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ...models.glm_4_7.config import GLM47Config
from ...models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from ...models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from ...models.qwen3_vl_2b.config import Qwen3VL2BConfig
from ...models.glm_4_7.model import GLM47Model
from ...models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from ...models.qwen3_coder_30b.model import Qwen3Coder30BModel
from ...models.qwen3_vl_2b.model import Qwen3VL2BModel

# TestNASModels

    """Test cases for NAS-enabled models."""
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def glm47_model_with_nas(self, mock_tokenizer, mock_model)():
        """Test GLM-4.7 model with NAS enabled."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.config = Mock()
        mock_model_instance.config.hidden_size = 512
        mock_model_instance.config.num_attention_heads = 8
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Create config with NAS enabled
        config = GLM47Config(
            model_path="test/path",
            enable_continuous_nas=True,
            nas_strategy="combined_adaptive",
            nas_min_depth_ratio=0.3,
            nas_max_depth_ratio=1.0,
            nas_min_width_ratio=0.3,
            nas_max_width_ratio=1.0,
            nas_latency_target_ms=100.0,
            nas_memory_budget_mb=2048.0,
            nas_accuracy_tradeoff_factor=0.7,
            nas_adaptation_frequency=10
        )
        
        # Create the model
        model = GLM47Model(config)
        
        # Check that NAS components are initialized
        assert_is_not_none(model._nas_controller)
        assertIsNotNone(model._model_adapter)
        
        # Test forward pass with NAS
        input_tensor = torch.randn(2)
        result = model.forward(input_tensor)
        
        # The model should handle NAS internally
        assert_is_not_none(result)
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def qwen3_4b_instruct_model_with_nas(self)():
        """Test Qwen3-4B-Instruct model with NAS enabled."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.config = Mock()
        mock_model_instance.config.hidden_size = 512
        mock_model_instance.config.num_attention_heads = 8
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Create config with NAS enabled
        config = Qwen34BInstruct2507Config(
            model_path="test/path",
            enable_continuous_nas=True,
            nas_strategy="combined_adaptive",
            nas_min_depth_ratio=0.3,
            nas_max_depth_ratio=1.0,
            nas_min_width_ratio=0.3,
            nas_max_width_ratio=1.0,
            nas_latency_target_ms=100.0,
            nas_memory_budget_mb=2048.0,
            nas_accuracy_tradeoff_factor=0.7,
            nas_adaptation_frequency=10
        )
        
        # Create the model
        model = Qwen34BInstruct2507Model(config)
        
        # Check that NAS components are initialized
        assert_is_not_none(model._nas_controller)
        assertIsNotNone(model._model_adapter)
        
        # Test forward pass with NAS
        input_tensor = torch.randn(2)
        result = model.forward(input_tensor)
        
        # The model should handle NAS internally
        assert_is_not_none(result)
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def qwen3_coder_model_with_nas(self)():
        """Test Qwen3-Coder model with NAS enabled."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.config = Mock()
        mock_model_instance.config.hidden_size = 512
        mock_model_instance.config.num_attention_heads = 8
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Create config with NAS enabled
        config = Qwen3Coder30BConfig(
            model_path="test/path",
            enable_continuous_nas=True,
            nas_strategy="combined_adaptive",
            nas_min_depth_ratio=0.3,
            nas_max_depth_ratio=1.0,
            nas_min_width_ratio=0.3,
            nas_max_width_ratio=1.0,
            nas_latency_target_ms=100.0,
            nas_memory_budget_mb=2048.0,
            nas_accuracy_tradeoff_factor=0.7,
            nas_adaptation_frequency=10
        )
        
        # Create the model
        model = Qwen3Coder30BModel(config)
        
        # Check that NAS components are initialized
        assert_is_not_none(model._nas_controller)
        assertIsNotNone(model._model_adapter)
        
        # Test forward pass with NAS
        input_tensor = torch.randn(2)
        result = model.forward(input_tensor)
        
        # The model should handle NAS internally
        assert_is_not_none(result)
    
    @patch('transformers.AutoModelForVision2Seq.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoImageProcessor.from_pretrained')
    def qwen3_vl_model_with_nas(self)():
        """Test Qwen3-VL model with NAS enabled."""
        # Mock the model, tokenizer, and image processor
        mock_model_instance = Mock()
        mock_model_instance.config = Mock()
        mock_model_instance.config.hidden_size = 512
        mock_model_instance.config.num_attention_heads = 8
        mock_model.return_value = mock_model_instance

        mock_tokenizer_instance = Mock()
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_image_proc_instance = Mock()
        mock_image_proc.return_value = mock_image_proc_instance

        # Create config with NAS enabled - need to set NAS params after creation
        config = Qwen3VL2BConfig()
        config.enable_continuous_nas = True
        config.nas_strategy = "combined_adaptive"
        config.nas_min_depth_ratio = 0.3
        config.nas_max_depth_ratio = 1.0
        config.nas_min_width_ratio = 0.3
        config.nas_max_width_ratio = 1.0
        config.nas_latency_target_ms = 100.0
        config.nas_memory_budget_mb = 2048.0
        config.nas_accuracy_tradeoff_factor = 0.7
        config.nas_adaptation_frequency = 10

        # Create the model
        model = Qwen3VL2BModel(config)

        # Check that NAS components are initialized
        assert_is_not_none(model._nas_controller)
        assertIsNotNone(model._model_adapter)

        # Test forward pass with NAS
        input_tensor = torch.randn(2)
        pixel_values = torch.randn(2, 3, 224, 224)
        result = model.forward(input_tensor, pixel_values=pixel_values)

        # The model should handle NAS internally
        assert_is_not_none(result)

if __name__ == '__main__':
    run_tests(test_functions)