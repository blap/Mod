"""
Quick verification test for async unimodal processing.
"""

import asyncio
import torch
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from src.inference_pio.common.async_unimodal_processing import (
    AsyncUnimodalProcessor,
    AsyncUnimodalRequest
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


async def quick_test():
    """Quick test to verify the async processing works."""
    print("Running quick async test...")
    
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
        id="quick_test",
        text="Hello, async world!"
    )
    
    # Submit the request
    future = await processor.submit_request(request)
    
    # Wait for the result
    result = await asyncio.wrap_future(future)
    
    # Stop the processor
    await processor.stop()
    
    print(f"SUCCESS: Got result with ID {result.request_id}")
    print(f"Result tensor shape: {result.result.shape if hasattr(result.result, 'shape') else 'N/A'}")
    print(f"Processing time: {result.processing_time:.4f}s")
    
    return result


if __name__ == "__main__":
    result = asyncio.run(quick_test())
    print("Async unimodal processing is working correctly!")