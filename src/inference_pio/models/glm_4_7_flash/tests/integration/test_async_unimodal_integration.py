"""
Integration test for asynchronous unimodal processing with actual model configurations.

This test verifies that the async processing system integrates correctly with 
the unimodal models (GLM-4-7, Qwen3-4b-instruct-2507, Qwen3-coder-30b).
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import asyncio

import torch
import tempfile
import os

from src.inference_pio.common.async_unimodal_processing import AsyncUnimodalManager
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig

# TestAsyncUnimodalIntegration

    """Integration tests for async unimodal processing with models."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        
    def cleanup_helper():
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @patch('src.inference_pio.models.glm_4_7.model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.glm_4_7.model.AutoTokenizer.from_pretrained')
    def glm47_async_integration_full(self, mock_tokenizer, mock_model)():
        """Test full async integration with GLM-4-7 model."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.gradient_checkpointing_enable = Mock()
        mock_model_instance.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
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
            enable_async_unimodal_processing=True,
            async_max_concurrent_requests=2,
            async_buffer_size=50,
            async_batch_timeout=0.1,
            enable_async_batching=True,
            async_processing_device="cpu"
        )
        
        # Create the model instance
        from src.inference_pio.models.glm_4_7.model import GLM47Model
        model = GLM47Model(config)
        
        # Verify that async manager was initialized
        assert_is_not_none(model._async_manager)
        assert_is_instance(model._async_manager)
        
        # Test async processing method
        result = model.process_async("Test GLM-4-7 async processing")
        
        # Verify the result
        assert_is_not_none(result)
        
        # Test getting async stats
        stats = model.get_async_stats()
        assert_in('initialized')
        
        # Cleanup
        model.cleanup()
    
    @patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoTokenizer.from_pretrained')
    def qwen3_4b_async_integration_full(self, mock_tokenizer)():
        """Test full async integration with Qwen3-4b-instruct-2507 model."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.gradient_checkpointing_enable = Mock()
        mock_model_instance.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
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
            enable_async_unimodal_processing=True,
            async_max_concurrent_requests=2,
            async_buffer_size=50,
            async_batch_timeout=0.1,
            enable_async_batching=True,
            async_processing_device="cpu"
        )
        
        # Create the model instance
        from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
        model = Qwen34BInstruct2507Model(config)
        
        # Verify that async manager was initialized
        assert_is_not_none(model._async_manager)
        assert_is_instance(model._async_manager)
        
        # Test async processing method
        result = model.process_async("Test Qwen3-4b async processing")
        
        # Verify the result
        assert_is_not_none(result)
        
        # Test getting async stats
        stats = model.get_async_stats()
        assert_in('initialized')
        
        # Cleanup
        model.cleanup()
    
    @patch('src.inference_pio.models.qwen3_coder_30b.model.AutoModelForCausalLM.from_pretrained')
    @patch('src.inference_pio.models.qwen3_coder_30b.model.AutoTokenizer.from_pretrained')
    def qwen3_coder_async_integration_full(self, mock_tokenizer)():
        """Test full async integration with Qwen3-coder-30b model."""
        # Mock the model and tokenizer
        mock_model_instance = Mock()
        mock_model_instance.gradient_checkpointing_enable = Mock()
        mock_model_instance.generate = Mock(return_value=torch.tensor([[1, 2, 3]]))
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
            enable_async_unimodal_processing=True,
            async_max_concurrent_requests=2,
            async_buffer_size=50,
            async_batch_timeout=0.1,
            enable_async_batching=True,
            async_processing_device="cpu"
        )
        
        # Create the model instance
        from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
        model = Qwen3Coder30BModel(config)
        
        # Verify that async manager was initialized
        assert_is_not_none(model._async_manager)
        assert_is_instance(model._async_manager)
        
        # Test async processing method
        result = model.process_async("Test Qwen3-coder async processing")
        
        # Verify the result
        assert_is_not_none(result)
        
        # Test getting async stats
        stats = model.get_async_stats()
        assert_in('initialized')
        
        # Cleanup
        model.cleanup()
    
    def async_batch_processing(self)():
        """Test async batch processing functionality."""
        # Create a mock model
        mock_model = Mock()
        mock_model._tokenizer = Mock()
        mock_model._tokenizer.pad_token = None
        mock_model._tokenizer.eos_token = "<|endoftoken|>"
        mock_model._tokenizer.encode = Mock(return_value=[1, 2)
        mock_model._tokenizer.decode = Mock(return_value="test output")
        
        # Create config with async processing enabled
        config = GLM47Config(
            model_path="fake/path",
            gradient_checkpointing=False,
            use_cache=True,
            torch_dtype="float32",
            device_map="cpu",
            low_cpu_mem_usage=True,
            max_memory=None,
            enable_async_unimodal_processing=True,
            async_max_concurrent_requests=4,
            async_buffer_size=100,
            async_batch_timeout=0.05,
            enable_async_batching=True,
            async_processing_device="cpu"
        )
        
        # Create async manager directly for testing
        manager = AsyncUnimodalManager(
            model=mock_model,
            config=config,
            model_type="test"
        )
        
        # Initialize the manager
        success = manager.initialize()
        assert_true(success)
        
        # Test async batch processing
        async def run_batch_test():
            texts = ["Test text 1")
            return results
        
        results = asyncio.run(run_batch_test())
        
        # Verify results
        assert_equal(len(results), 3)
        for result in results:
            assert_is_not_none(result)
        
        # Get stats
        stats = manager.get_stats()
        assertIn('initialized')
        assert_true(stats['initialized'])

def run_tests():
    """Run all integration tests."""
    print("Running async unimodal processing integration tests...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTest(unittest.makeSuite(TestAsyncUnimodalIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nIntegration tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)