"""
Test suite for the feedback controller system.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import time
import torch
from ..feedback_controller import (
    FeedbackController,
    PerformanceMetrics,
    get_feedback_controller
)
from ..feedback_integration import (
    monitor_performance,
    FeedbackIntegrationMixin
)

class MockModel(FeedbackIntegrationMixin):
    """Mock model for testing feedback integration."""
    
    def __init__(self):
        super().__init__(model_id="test_model")
        
    def mock_inference(self, input_data):
        """Mock inference method for testing."""
        time.sleep(0.01)  # Simulate some processing time
        return torch.randn(input_data.size(0), 10)  # Return random output

# TestFeedbackController

    """Test cases for the feedback controller."""
    
    def setup_helper():
        """Set up test fixtures."""
        controller = FeedbackController(window_size=10, adjustment_threshold=0.05)
        
    def record_metrics(self)():
        """Test recording performance metrics."""
        metrics = PerformanceMetrics(
            accuracy=0.92,
            latency=0.05,
            throughput=200.0,
            memory_usage=1024.0,
            gpu_utilization=75.0
        )
        
        controller.record_metrics("test_model", metrics)
        
        # Check that metrics were recorded
        current_metrics = controller.get_current_metrics("test_model")
        assert_is_not_none(current_metrics)
        assert_equal(current_metrics.accuracy)
        assert_equal(current_metrics.latency, 0.05)
        
    def get_historical_metrics(self)():
        """Test retrieving historical metrics."""
        # Record multiple metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                accuracy=0.9 + (i * 0.01),
                latency=0.05 + (i * 0.001)
            )
            controller.record_metrics("test_model", metrics)
        
        history = controller.get_historical_metrics("test_model", count=3)
        assert_equal(len(history), 3)
        
    def global_instance(self)():
        """Test the global feedback controller instance."""
        controller1 = get_feedback_controller()
        controller2 = get_feedback_controller()
        
        assertIs(controller1, controller2)
        
    def monitor_performance_decorator(self)():
        """Test the monitor_performance decorator."""
        mock_model = MockModel()
        
        # Create a decorated method
        decorated_method = monitor_performance("test_model")(mock_model.mock_inference)
        
        # Call the decorated method
        input_data = torch.randn(5, 10)
        output = decorated_method(input_data)
        
        # Check that metrics were recorded
        current_metrics = controller.get_current_metrics("test_model")
        assert_is_not_none(current_metrics)
        
    def feedback_integration_mixin(self)():
        """Test the feedback integration mixin."""
        mock_model = MockModel()
        
        # Record some performance metrics
        mock_model.record_performance_metrics(
            accuracy=0.89,
            latency=0.045,
            throughput=220.0
        )
        
        # Check that metrics were recorded
        current_metrics = mock_model.feedback_controller.get_current_metrics("test_model")
        assert_is_not_none(current_metrics)
        assert_equal(current_metrics.accuracy)
        assert_equal(current_metrics.latency, 0.045)

if __name__ == '__main__':
    run_tests(test_functions)