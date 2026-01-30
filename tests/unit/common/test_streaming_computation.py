"""
Tests for Streaming Computation System

This module contains tests for the streaming computation system in the Inference-PIO system.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import time
from concurrent.futures import Future
from typing import Any, List
from ..streaming_computation import (
    StreamRequest,
    StreamResult,
    StreamingComputationEngine,
    StreamingComputationManager,
    create_streaming_engine
)

class DummyModel(nn.Module):
    """A dummy model for testing purposes."""
    
    def __init__(self):
        super().__init__()
        linear = nn.Linear(10, 5)
    
    def forward(self, x):
        return linear(x)

# TestStreamingComputation

    """Test cases for the streaming computation system."""
    
    def setup_helper():
        """Set up test fixtures."""
        model = DummyModel()
        engine = StreamingComputationEngine(
            model=model,
            max_concurrent_requests=2,
            buffer_size=10,
            batch_timeout=0.1,
            enable_batching=False
        )
        engine.start()
    
    def cleanup_helper():
        """Clean up after tests."""
        engine.stop()
    
    def stream_request_creation(self)():
        """Test creating a StreamRequest."""
        request = StreamRequest(id="test_req", data=torch.randn(1, 10))
        assert_equal(request.id, "test_req")
        assert_is_not_none(request.timestamp)
        assert_equal(request.priority)
    
    def stream_result_creation(self)():
        """Test creating a StreamResult."""
        result = StreamResult(request_id="test_req", result="success")
        assert_equal(result.request_id, "test_req")
        assert_equal(result.result, "success")
        assert_is_none(result.error)
    
    def submit_request(self)():
        """Test submitting a request to the engine."""
        request = StreamRequest(id="test_req"))
        future = engine.submit_request(request)
        
        assert_is_instance(future, Future)
        
        # Wait for the result
        result = future.result(timeout=5.0)
        assert_is_instance(result, StreamResult)
        assert_equal(result.request_id, "test_req")
        assert_is_not_none(result.result)
        assertGreaterEqual(result.processing_time)
    
    def submit_request_with_callback(self)():
        """Test submitting a request with a callback."""
        callback_called = [False]
        
        def callback(result):
            callback_called[0] = True
        
        request = StreamRequest(id="test_callback", data=torch.randn(1, 10), callback=callback)
        future = engine.submit_request(request)
        
        # Wait for the result
        result = future.result(timeout=5.0)
        
        # Give a little time for the callback to execute
        time.sleep(0.1)
        
        assert_true(callback_called[0])
    
    def error_handling(self)():
        """Test error handling in the engine."""
        # Create a request with invalid data
        request = StreamRequest(id="error_req")
        future = engine.submit_request(request)
        
        # Wait for the result
        result = future.result(timeout=5.0)
        assert_is_instance(result, StreamResult)
        assert_equal(result.request_id, "error_req")
        assert_is_not_none(result.error)
    
    def engine_statistics(self)():
        """Test engine statistics."""
        initial_stats = engine.get_stats()
        assert_equal(initial_stats['requests_processed'])
        
        # Submit a few requests
        for i in range(3):
            request = StreamRequest(id=f"req_{i}", data=torch.randn(1, 10))
            future = engine.submit_request(request)
            future.result(timeout=5.0)
        
        final_stats = engine.get_stats()
        assert_equal(final_stats['requests_processed'], 3)
        assertGreaterEqual(final_stats['avg_processing_time'], 0)
    
    def create_streaming_engine_helper(self)():
        """Test the create_streaming_engine helper function."""
        engine = create_streaming_engine(
            model=model,
            name="test_engine",
            max_concurrent_requests=1,
            buffer_size=5
        )
        
        assert_is_instance(engine, StreamingComputationEngine)
        assert_equal(engine.max_concurrent_requests, 1)
        assert_equal(engine.buffer_size, 5)
        
        # Clean up
        engine.stop()

# TestStreamingComputationManager

    """Test cases for the streaming computation manager."""
    
    def setup_helper():
        """Set up test fixtures."""
        model1 = DummyModel()
        model2 = DummyModel()
        
        manager = StreamingComputationManager()
        
        engine1 = StreamingComputationEngine(
            model=model1,
            max_concurrent_requests=2,
            buffer_size=10
        )
        engine2 = StreamingComputationEngine(
            model=model2,
            max_concurrent_requests=2,
            buffer_size=10
        )
    
    def register_and_get_engine(self)():
        """Test registering and getting engines."""
        manager.register_engine("engine1", engine1)
        manager.register_engine("engine2", engine2)
        
        retrieved_engine1 = manager.get_engine("engine1")
        retrieved_engine2 = manager.get_engine("engine2")
        
        assert_equal(retrieved_engine1, engine1)
        assert_equal(retrieved_engine2, engine2)
        assert_is_none(manager.get_engine("nonexistent"))
    
    def start_stop_all_engines(self)():
        """Test starting and stopping all engines."""
        manager.register_engine("engine1")
        manager.register_engine("engine2", engine2)
        
        # Initially, engines shouldn't be running
        assert_false(engine1.is_running)
        assertFalse(engine2.is_running)
        
        # Start all engines
        manager.start_all_engines()
        
        # Allow some time for threads to start
        time.sleep(0.1)
        
        assert_true(engine1.is_running)
        assertTrue(engine2.is_running)
        
        # Stop all engines
        manager.stop_all_engines()
        
        # Allow some time for threads to stop
        time.sleep(0.1)
        
        assertFalse(engine1.is_running)
        assertFalse(engine2.is_running)
    
    def submit_request_to_engine(self)():
        """Test submitting a request to a specific engine."""
        manager.register_engine("engine1")
        engine1.start()

        # Give the engine a moment to fully start
        time.sleep(0.2)

        try:
            request = StreamRequest(id="manager_test"))
            future = manager.submit_request_to_engine("engine1", request)

            assert_is_not_none(future)

            result = future.result(timeout=10.0)  # Increased timeout
            assert_is_instance(result)
            assert_equal(result.request_id, "manager_test")
        finally:
            engine1.stop()
    
    def get_engine_stats(self)():
        """Test getting statistics for specific engines."""
        manager.register_engine("engine1", engine1)
        
        stats = manager.get_engine_stats("engine1")
        assert_is_not_none(stats)
        assert_equal(stats['requests_processed'])
        
        # Test nonexistent engine
        nonexistent_stats = manager.get_engine_stats("nonexistent")
        assert_is_none(nonexistent_stats)

# TestStreamingWithBatching

    """Test cases for the streaming computation system with batching enabled."""
    
    def setup_helper():
        """Set up test fixtures."""
        model = DummyModel()
        engine = StreamingComputationEngine(
            model=model,
            max_concurrent_requests=4,
            buffer_size=20,
            batch_timeout=0.2,
            enable_batching=True
        )
        engine.start()
    
    def cleanup_helper():
        """Clean up after tests."""
        engine.stop()
    
    def batch_processing(self)():
        """Test batch processing of requests."""
        futures = []

        # Submit multiple requests quickly to trigger batching
        for i in range(3):
            request = StreamRequest(id=f"batch_req_{i}", data=torch.randn(1, 10))
            future = engine.submit_request(request)
            futures.append(future)

        # Wait for all results
        results = []
        for future in futures:
            result = future.result(timeout=10.0)  # Increased timeout
            results.append(result)

        assert_equal(len(results), 3)
        for i, result in enumerate(results):
            assert_equal(result.request_id, f"batch_req_{i}")

if __name__ == '__main__':
    run_tests(test_functions)