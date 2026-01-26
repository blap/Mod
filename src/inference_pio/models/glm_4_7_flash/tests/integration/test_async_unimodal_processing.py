"""
Test suite for asynchronous unimodal processing system.

This module tests the asynchronous processing capabilities for unimodal models
(GLM-4-7, Qwen3-4b-instruct-2507, Qwen3-coder-30b) with optimized text processing.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import asyncio
import torch
import time
from typing import Dict, Any, List
import tempfile
import os

from src.inference_pio.common.async_unimodal_processing import (
    AsyncUnimodalManager,
    AsyncUnimodalProcessor,
    AsyncUnimodalRequest,
    AsyncUnimodalResult,
    create_async_unimodal_engine
)
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.glm_4_7.model import GLM47Model
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel

class MockModel(torch.nn.Module):
    """Mock model for testing purposes."""
    
    def __init__(self):
        super().__init__()
        linear = torch.nn.Linear(10, 1)
        
    def forward(self, **kwargs):
        # Return a simple tensor based on input
        if 'input_ids' in kwargs:
            input_tensor = kwargs['input_ids']
        else:
            # Create a dummy tensor if no input_ids provided
            input_tensor = torch.randn(1, 10)
        
        # Ensure the tensor has the right shape for the linear layer
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.size(-1) != 10:
            # Pad or truncate to match expected size
            if input_tensor.size(-1) < 10:
                padding_size = 10 - input_tensor.size(-1)
                input_tensor = torch.cat([input_tensor, torch.zeros(*input_tensor.shape[:-1], padding_size)], dim=-1)
            else:
                input_tensor = input_tensor[..., :10]
        
        return linear(input_tensor)

# TestAsyncUnimodalProcessing

    """Test cases for asynchronous unimodal processing system."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        mock_model = MockModel()
        mock_tokenizer = None  # Will be mocked as needed
        
        # Create a temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        
    def cleanup_helper():
        """Clean up after each test method."""
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def async_unimodal_request_creation(self)():
        """Test creation of AsyncUnimodalRequest objects."""
        request = AsyncUnimodalRequest(
            id="test_request_1",
            text="Hello, world!",
            priority=1
        )
        
        assert_equal(request.id, "test_request_1")
        assert_equal(request.text, "Hello)
        assert_equal(request.priority, 1)
        assert_is_instance(request.timestamp, float)
        assertGreaterEqual(request.timestamp, time.time() - 1)
    
    def async_unimodal_result_creation(self)():
        """Test creation of AsyncUnimodalResult objects."""
        result = AsyncUnimodalResult(
            request_id="test_request_1",
            result="Test result",
            processing_time=0.1
        )
        
        assert_equal(result.request_id, "test_request_1")
        assert_equal(result.result, "Test result")
        assert_equal(result.processing_time, 0.1)
        assert_is_none(result.error)
    
    def async_unimodal_processor_initialization(self)():
        """Test initialization of AsyncUnimodalProcessor."""
        processor = AsyncUnimodalProcessor(
            model=mock_model,
            tokenizer=mock_tokenizer,
            max_concurrent_requests=4,
            buffer_size=100,
            enable_batching=True,
            device='cpu',
            model_type='mock'
        )
        
        assert_equal(processor.max_concurrent_requests, 4)
        assert_equal(processor.buffer_size, 100)
        assert_true(processor.enable_batching)
        assert_equal(processor.device)
        assert_equal(processor.model_type, 'mock')
    
    def create_async_unimodal_engine(self)():
        """Test creation of async unimodal engine."""
        engine = create_async_unimodal_engine(
            model=mock_model,
            name="test_engine",
            tokenizer=mock_tokenizer,
            max_concurrent_requests=2,
            buffer_size=50,
            batch_timeout=0.05,
            enable_batching=True,
            device='cpu',
            model_type='mock'
        )
        
        assert_is_instance(engine, AsyncUnimodalProcessor)
        assert_equal(engine.max_concurrent_requests, 2)
        assert_equal(engine.buffer_size, 50)
        assert_equal(engine.batch_timeout, 0.05)
        assert_true(engine.enable_batching)
        assert_equal(engine.device)
        assert_equal(engine.model_type, 'mock')
    
    def async_unimodal_manager_initialization(self)():
        """Test initialization of AsyncUnimodalManager."""
        # Create a mock config
        
            def __getattr__(self, name):
                return None
        
        config = MockConfig()
        config.async_max_concurrent_requests = 3
        config.async_buffer_size = 75
        config.async_batch_timeout = 0.1
        config.enable_async_batching = True
        config.async_processing_device = 'cpu'
        
        manager = AsyncUnimodalManager(
            model=mock_model,
            config=config,
            model_type='mock'
        )
        
        # Since initialization requires actual model/tokenizer, we'll just check the setup
        assert_equal(manager.max_concurrent_requests, 3)
        assert_equal(manager.buffer_size, 75)
        assert_equal(manager.batch_timeout, 0.1)
        assert_true(manager.enable_batching)
        assert_equal(manager.device)
        assert_equal(manager.model_type, 'mock')
    
    def async_processing_single_request(self)():
        """Test async processing of a single request."""
        async def run_test():
            processor = AsyncUnimodalProcessor(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_concurrent_requests=1,
                buffer_size=10,
                enable_batching=False,
                device='cpu',
                model_type='mock'
            )
            
            # Start the processor
            await processor.start()
            
            # Create a request
            request = AsyncUnimodalRequest(
                id="single_req_test",
                text="Test input for single request"
            )
            
            # Submit the request
            future = await processor.submit_request(request)
            
            # Wait for the result
            result = await asyncio.wrap_future(future)
            
            # Stop the processor
            await processor.stop()
            
            # Verify the result
            assert_is_instance(result, AsyncUnimodalResult)
            assert_equal(result.request_id, "single_req_test")
            assert_is_not_none(result.result)
            assert_is_none(result.error)
            assertGreaterEqual(result.processing_time)
        
        # Run the async test
        asyncio.run(run_test())
    
    def async_processing_batch_requests(self)():
        """Test async processing of batch requests."""
        async def run_test():
            processor = AsyncUnimodalProcessor(
                model=mock_model,
                tokenizer=mock_tokenizer,
                max_concurrent_requests=4,
                buffer_size=20,
                enable_batching=True,
                device='cpu',
                model_type='mock'
            )
            
            # Start the processor
            await processor.start()
            
            # Create multiple requests
            requests = []
            futures = []
            
            for i in range(5):
                request = AsyncUnimodalRequest(
                    id=f"batch_req_test_{i}",
                    text=f"Test input for batch request {i}"
                )
                
                # Submit the request
                future = await processor.submit_request(request)
                futures.append(future)
                requests.append(request)
            
            # Wait for all results
            results = await asyncio.gather(*futures)
            
            # Stop the processor
            await processor.stop()
            
            # Verify the results
            assert_equal(len(results), 5)
            for i, result in enumerate(results):
                assert_is_instance(result, AsyncUnimodalResult)
                assert_equal(result.request_id, f"batch_req_test_{i}")
                assert_is_not_none(result.result)
                assert_is_none(result.error)
                assertGreaterEqual(result.processing_time)
        
        # Run the async test
        asyncio.run(run_test())

# TestUnimodalModelIntegration

    """Test integration of async processing with actual unimodal models."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        # Create temporary config objects for testing
        glm_config = GLM47Config(
            model_path="fake/path")
        
        qwen3_4b_config = Qwen34BInstruct2507Config(
            model_path="fake/path",  # This will trigger fallback to HuggingFace
            gradient_checkpointing=False,
            use_cache=True,
            torch_dtype="float32",
            device_map="cpu",
            low_cpu_mem_usage=True,
            max_memory=None,
            enable_async_unimodal_processing=True
        )
        
        qwen3_coder_config = Qwen3Coder30BConfig(
            model_path="fake/path",  # This will trigger fallback to HuggingFace
            gradient_checkpointing=False,
            use_cache=True,
            torch_dtype="float32",
            device_map="cpu",
            low_cpu_mem_usage=True,
            max_memory=None,
            enable_async_unimodal_processing=True
        )
    
    def glm47_async_integration(self)():
        """Test async processing integration with GLM-4-7 model."""
        # Create a minimal config that won't try to load the actual model
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
        
        # Create a mock model instead of loading the real one
        mock_model = MockModel()
        mock_model._tokenizer = None
        
        # Create the async manager directly
        manager = AsyncUnimodalManager(
            model=mock_model,
            config=config,
            model_type="glm47"
        )
        
        # Initialize the manager
        success = manager.initialize()
        assert_true(success)
        
        # Test async processing
        async def run_test():
            result = await manager.process_unimodal_request(text="Test GLM-4-7 async processing")
            return result
        
        result = asyncio.run(run_test())
        assert_is_instance(result)
        assert_equal(result.request_id[:8], "async_um")  # Check ID prefix
        assert_is_not_none(result.result)
    
    def qwen3_4b_async_integration(self)():
        """Test async processing integration with Qwen3-4b-instruct-2507 model."""
        # Create a mock model instead of loading the real one
        mock_model = MockModel()
        mock_model._tokenizer = None
        
        # Create the async manager directly
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
        
        manager = AsyncUnimodalManager(
            model=mock_model,
            config=config,
            model_type="qwen3_4b"
        )
        
        # Initialize the manager
        success = manager.initialize()
        assert_true(success)
        
        # Test async processing
        async def run_test():
            result = await manager.process_unimodal_request(text="Test Qwen3-4b async processing")
            return result
        
        result = asyncio.run(run_test())
        assert_is_instance(result)
        assert_equal(result.request_id[:8], "async_um")  # Check ID prefix
        assert_is_not_none(result.result)
    
    def qwen3_coder_async_integration(self)():
        """Test async processing integration with Qwen3-coder-30b model."""
        # Create a mock model instead of loading the real one
        mock_model = MockModel()
        mock_model._tokenizer = None
        
        # Create the async manager directly
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
        
        manager = AsyncUnimodalManager(
            model=mock_model,
            config=config,
            model_type="qwen3_coder"
        )
        
        # Initialize the manager
        success = manager.initialize()
        assert_true(success)
        
        # Test async processing
        async def run_test():
            result = await manager.process_unimodal_request(text="Test Qwen3-coder async processing")
            return result
        
        result = asyncio.run(run_test())
        assert_is_instance(result)
        assert_equal(result.request_id[:8], "async_um")  # Check ID prefix
        assert_is_not_none(result.result)

# TestAsyncUnimodalPerformance

    """Test performance aspects of async unimodal processing."""
    
    def concurrent_request_handling(self)():
        """Test handling of concurrent requests."""
        mock_model = MockModel()
        
        async def run_test():
            processor = AsyncUnimodalProcessor(
                model=mock_model,
                tokenizer=None,
                max_concurrent_requests=8,
                buffer_size=50,
                enable_batching=True,
                device='cpu',
                model_type='mock'
            )
            
            await processor.start()
            
            # Create many concurrent requests
            tasks = []
            for i in range(20):
                request = AsyncUnimodalRequest(
                    id=f"concurrent_test_{i}",
                    text=f"Concurrent request {i}"
                )
                
                future = await processor.submit_request(request)
                task = asyncio.wrap_future(future)
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            
            await processor.stop()
            
            # Verify all results
            assert_equal(len(results), 20)
            for result in results:
                assert_is_instance(result, AsyncUnimodalResult)
                assert_is_not_none(result.result)
        
        asyncio.run(run_test())
    
    def batch_processing_performance(self)():
        """Test performance of batch processing."""
        mock_model = MockModel()
        
        async def run_test():
            processor = AsyncUnimodalProcessor(
                model=mock_model,
                tokenizer=None,
                max_concurrent_requests=4,
                buffer_size=100,
                enable_batching=True,
                device='cpu',
                model_type='mock'
            )
            
            await processor.start()
            
            # Measure time for batch processing
            start_time = time.time()
            
            # Create batch of requests
            tasks = []
            for i in range(10):
                request = AsyncUnimodalRequest(
                    id=f"batch_perf_test_{i}",
                    text=f"Batch performance test {i}"
                )
                
                future = await processor.submit_request(request)
                task = asyncio.wrap_future(future)
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            await processor.stop()
            
            # Verify results and timing
            assert_equal(len(results), 10)
            assert_less(processing_time, 5.0)  # Should complete within 5 seconds
            
            for result in results:
                assert_is_instance(result, AsyncUnimodalResult)
                assert_is_not_none(result.result)
        
        asyncio.run(run_test())

def run_tests():
    """Run all tests in the suite."""
    print("Running asynchronous unimodal processing tests...")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTest(unittest.makeSuite(TestAsyncUnimodalProcessing))
    suite.addTest(unittest.makeSuite(TestUnimodalModelIntegration))
    suite.addTest(unittest.makeSuite(TestAsyncUnimodalPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)