"""
Simple test to verify the streaming computation system works correctly
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
from src.inference_pio.common.streaming_computation import StreamingComputationEngine, StreamRequest

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

def test_simple_streaming():
    model = SimpleModel()
    engine = StreamingComputationEngine(
        model=model,
        max_concurrent_requests=2,
        buffer_size=10,
        batch_timeout=0.1,
        enable_batching=False  # Disable batching for simplicity
    )

    engine.start()

    # Create a request
    request = StreamRequest(id="test_req", data=torch.randn(1, 10))
    future = engine.submit_request(request)

    # Get result
    result = future.result(timeout=10.0)
    print(f"Result: {result.result}")
    print(f"Processing time: {result.processing_time}")

    engine.stop()
    print("Test passed!")

if __name__ == "__main__":
    test_simple_streaming()