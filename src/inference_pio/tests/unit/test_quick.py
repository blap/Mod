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


class MockModel(torch.nn.Module):
    """Mock model for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
        
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
        
        return self.linear(input_tensor)


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