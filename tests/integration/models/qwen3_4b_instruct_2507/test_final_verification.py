"""
Final verification test for async unimodal processing integration with models.
"""

import asyncio
import torch
import unittest
from unittest.mock import Mock, patch
import tempfile
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from src.inference_pio.common.async_unimodal_processing import AsyncUnimodalManager
from src.inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig


class TestAsyncUnimodalIntegration(unittest.TestCase):
    """Integration tests for async unimodal processing with models."""
    
    @patch('src.inference_pio.models.glm_4_7_flash.model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.glm_4_7_flash.model.AutoTokenizer.from_pretrained')
    def test_glm47_async_integration(self, mock_tokenizer, mock_model):
        """Test async integration with GLM-4-7 model."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.gradient_checkpointing_enable = Mock()
        mock_model_instance.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        mock_model_instance.return_value = torch.tensor([[0.1, 0.2, 0.3]])
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftoken|>"
        mock_tokenizer_instance.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer_instance.decode = Mock(return_value="test output")
        
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Create config with async processing enabled
        config = GLM47Config(
            model_path="fake/path",
            gradient_checkpointing=False,
            use_cache=True,
            torch_dtype="float32",
            device_map="cpu",
            low_cpu_mem_usage=True,
            max_memory=None,
            enable_async_unimodal_processing=True
        )
        
        # Create the model instance
        from src.inference_pio.models.glm_4_7_flash.model import GLM47FlashModel
        model = GLM47Model(config)
        
        # Verify that async manager was initialized
        self.assertIsNotNone(model._async_manager)
        self.assertIsInstance(model._async_manager, AsyncUnimodalManager)
        
        # Test async processing method
        result = model.process_async("Test GLM-4-7 async processing")
        
        # Verify the result
        self.assertIsNotNone(result)
        
        # Test getting async stats
        stats = model.get_async_stats()
        self.assertIn('initialized', stats)
        
        # Cleanup
        model.cleanup()
        print("✓ GLM-4-7 async integration test passed!")
    
    @patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoTokenizer.from_pretrained')
    def test_qwen3_4b_async_integration(self, mock_tokenizer, mock_model):
        """Test async integration with Qwen3-4b-instruct-2507 model."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.gradient_checkpointing_enable = Mock()
        mock_model_instance.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        mock_model_instance.return_value = torch.tensor([[0.1, 0.2, 0.3]])
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftoken|>"
        mock_tokenizer_instance.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer_instance.decode = Mock(return_value="test output")
        
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Create config with async processing enabled
        config = Qwen34BInstruct2507Config(
            model_path="fake/path",
            gradient_checkpointing=False,
            use_cache=True,
            torch_dtype="float32",
            device_map="cpu",
            low_cpu_mem_usage=True,
            max_memory=None,
            enable_async_unimodal_processing=True
        )
        
        # Create the model instance
        from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
        model = Qwen34BInstruct2507Model(config)
        
        # Verify that async manager was initialized
        self.assertIsNotNone(model._async_manager)
        self.assertIsInstance(model._async_manager, AsyncUnimodalManager)
        
        # Test async processing method
        result = model.process_async("Test Qwen3-4b async processing")
        
        # Verify the result
        self.assertIsNotNone(result)
        
        # Test getting async stats
        stats = model.get_async_stats()
        self.assertIn('initialized', stats)
        
        # Cleanup
        model.cleanup()
        print("✓ Qwen3-4b async integration test passed!")
    
    @patch('src.inference_pio.models.qwen3_coder_30b.model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.qwen3_coder_30b.model.AutoTokenizer.from_pretrained')
    def test_qwen3_coder_async_integration(self, mock_tokenizer, mock_model):
        """Test async integration with Qwen3-coder-30b model."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.gradient_checkpointing_enable = Mock()
        mock_model_instance.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
        mock_model_instance.return_value = torch.tensor([[0.1, 0.2, 0.3]])
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<|endoftoken|>"
        mock_tokenizer_instance.encode = Mock(return_value=[1, 2, 3])
        mock_tokenizer_instance.decode = Mock(return_value="test output")
        
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Create config with async processing enabled
        config = Qwen3Coder30BConfig(
            model_path="fake/path",
            gradient_checkpointing=False,
            use_cache=True,
            torch_dtype="float32",
            device_map="cpu",
            low_cpu_mem_usage=True,
            max_memory=None,
            enable_async_unimodal_processing=True
        )
        
        # Create the model instance
        from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
        model = Qwen3Coder30BModel(config)
        
        # Verify that async manager was initialized
        self.assertIsNotNone(model._async_manager)
        self.assertIsInstance(model._async_manager, AsyncUnimodalManager)
        
        # Test async processing method
        result = model.process_async("Test Qwen3-coder async processing")
        
        # Verify the result
        self.assertIsNotNone(result)
        
        # Test getting async stats
        stats = model.get_async_stats()
        self.assertIn('initialized', stats)
        
        # Cleanup
        model.cleanup()
        print("✓ Qwen3-coder async integration test passed!")


def run_final_verification():
    """Run final verification tests."""
    print("Running final verification tests for async unimodal processing...\n")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTest(unittest.makeSuite(TestAsyncUnimodalIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nVerification tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 All verification tests passed! Async unimodal processing is working correctly.")
        return True
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} tests failed.")
        return False


if __name__ == "__main__":
    success = run_final_verification()
    exit(0 if success else 1)