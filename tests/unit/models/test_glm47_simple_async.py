"""
Simple test for the async unimodal processing implementation.
"""

import asyncio
import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from src.inference_pio.common.async_unimodal_processing import (
    AsyncUnimodalManager,
    AsyncUnimodalProcessor,
    AsyncUnimodalRequest,
    AsyncUnimodalResult,
    create_async_unimodal_engine
)

# Import the real model implementation
from real_model_for_testing import RealGLM47Model


class MockModel(RealGLM47Model):
    """Real model for testing purposes."""

    def __init__(self):
        # Initialize with smaller parameters for testing
        super().__init__(
            hidden_size=256,
            num_attention_heads=4,
            num_hidden_layers=2,
            intermediate_size=512,
            vocab_size=1000
        )


def test_async_unimodal_processor():
    """Test the async unimodal processor with a mock model."""
    print("Testing AsyncUnimodalProcessor...")
    
    async def run_test():
        # Create a mock model
        mock_model = MockModel()
        
        # Create processor without a tokenizer (should use fallback)
        processor = AsyncUnimodalProcessor(
            model=mock_model,
            tokenizer=None,  # This should trigger fallback
            max_concurrent_requests=2,
            buffer_size=10,
            enable_batching=False,
            device='cpu',
            model_type='mock'
        )
        
        # Start the processor
        await processor.start()
        
        # Create a request
        request = AsyncUnimodalRequest(
            id="test_request_1",
            text="Hello, async world!"
        )
        
        # Submit the request
        future = await processor.submit_request(request)
        
        # Wait for the result
        result = await asyncio.wrap_future(future)
        
        # Stop the processor
        await processor.stop()
        
        print(f"Result: {result}")
        print(f"Result type: {type(result)}")
        print(f"Request ID: {result.request_id}")
        print(f"Has result: {result.result is not None}")
        print(f"Processing time: {result.processing_time}")
        
        return result
    
    result = asyncio.run(run_test())
    assert result is not None
    assert result.request_id == "test_request_1"
    assert result.result is not None
    print("✓ AsyncUnimodalProcessor test passed!")


def test_async_unimodal_manager():
    """Test the async unimodal manager with a mock model."""
    print("\nTesting AsyncUnimodalManager...")
    
    # Create a mock config
    class MockConfig:
        def __getattr__(self, name):
            # Return default values for async processing settings
            defaults = {
                'async_max_concurrent_requests': 2,
                'async_buffer_size': 10,
                'async_batch_timeout': 0.1,
                'enable_async_batching': False,
                'async_processing_device': 'cpu'
            }
            return defaults.get(name, None)
    
    config = MockConfig()
    
    # Create a mock model
    mock_model = MockModel()
    
    # Create manager
    manager = AsyncUnimodalManager(
        model=mock_model,
        config=config,
        model_type='mock'
    )
    
    # Initialize the manager
    success = manager.initialize()
    print(f"Initialization success: {success}")
    
    if success:
        # Test async processing
        async def run_test():
            result = await manager.process_unimodal_request(text="Test async manager processing")
            return result
        
        result = asyncio.run(run_test())
        print(f"Result: {result}")
        print(f"Result type: {type(result)}")
        print(f"Has result: {result.result is not None}")
        print(f"Processing time: {result.processing_time}")
        
        assert result is not None
        assert result.result is not None
        print("✓ AsyncUnimodalManager test passed!")


def test_create_async_engine():
    """Test creating an async unimodal engine."""
    print("\nTesting create_async_unimodal_engine...")
    
    # Create a mock model
    mock_model = MockModel()
    
    # Create engine
    engine = create_async_unimodal_engine(
        model=mock_model,
        name="test_engine",
        tokenizer=None,  # Should use fallback
        max_concurrent_requests=2,
        buffer_size=10,
        batch_timeout=0.05,
        enable_batching=False,
        device='cpu',
        model_type='mock'
    )
    
    print(f"Engine type: {type(engine)}")
    assert engine is not None
    assert hasattr(engine, 'start')
    assert hasattr(engine, 'submit_request')
    print("✓ create_async_unimodal_engine test passed!")


if __name__ == "__main__":
    print("Running simple async unimodal processing tests...\n")
    
    try:
        test_async_unimodal_processor()
        test_async_unimodal_manager()
        test_create_async_engine()
        
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)