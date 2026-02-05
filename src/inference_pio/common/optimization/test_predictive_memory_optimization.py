"""
Tests for the Predictive Memory Optimization System
"""

import unittest
import torch
import tempfile
import shutil
import os
from datetime import datetime
from src.inference_pio.common.optimization.predictive_memory_optimization import (
    MemoryAccessPredictor,
    PredictiveMemoryManager,
    PredictiveMemoryOptimization
)


class TestMemoryAccessPredictor(unittest.TestCase):
    """Test cases for the MemoryAccessPredictor class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.predictor = MemoryAccessPredictor(prediction_horizon=30, window_size=100)

    def test_record_access_and_extract_features(self):
        """Test recording access events and extracting features."""
        # Record some access events
        timestamp = 1000.0
        self.predictor.record_access("tensor1", timestamp, 1024, "read", "cpu")
        self.predictor.record_access("tensor1", timestamp + 1.0, 1024, "write", "cpu")
        self.predictor.record_access("tensor1", timestamp + 2.0, 1024, "compute", "cuda:0")

        # Extract features
        features = self.predictor.extract_features("tensor1")
        
        # Check that features are returned
        self.assertIsInstance(features, dict)
        self.assertIn('access_frequency', features)
        self.assertIn('access_interval_mean', features)
        self.assertIn('access_interval_std', features)
        self.assertIn('recent_access_trend', features)
        self.assertIn('size_normalized_frequency', features)
        self.assertIn('access_type_distribution', features)

    def test_predict_access_probability(self):
        """Test predicting access probability."""
        # Record some access events to establish a pattern
        base_time = 1000.0
        for i in range(5):
            self.predictor.record_access("frequent_tensor", base_time + i * 2.0, 1024, "read", "cpu")
        
        # Predict access probability
        prob = self.predictor.predict_access_probability("frequent_tensor", time_ahead=10.0)
        
        # Probability should be between 0 and 1
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_predict_access_probability_unknown_tensor(self):
        """Test predicting access probability for unknown tensor."""
        prob = self.predictor.predict_access_probability("unknown_tensor", time_ahead=10.0)
        
        # Should return 0.0 for unknown tensor
        self.assertEqual(prob, 0.0)


class TestPredictiveMemoryManager(unittest.TestCase):
    """Test cases for the PredictiveMemoryManager class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = {
            'prediction_horizon_seconds': 30,
            'access_history_window_size': 100,
            'memory_prediction_threshold': 0.9,
            'proactive_management_interval': 0.1,  # Short interval for testing
            'offload_directory': tempfile.mkdtemp()
        }
        self.manager = PredictiveMemoryManager(self.config)

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory
        shutil.rmtree(self.config['offload_directory'])

    def test_start_stop_monitoring(self):
        """Test starting and stopping the monitoring."""
        # Start monitoring
        result = self.manager.start_monitoring()
        self.assertTrue(result)
        self.assertTrue(self.manager.active)

        # Stop monitoring
        result = self.manager.stop_monitoring()
        self.assertTrue(result)
        self.assertFalse(self.manager.active)

    def test_record_tensor_access(self):
        """Test recording tensor access."""
        # Create a dummy tensor
        tensor = torch.randn(10, 10)
        
        # Record access
        self.manager.record_tensor_access("test_tensor", tensor, "read")
        
        # Check that the tensor is tracked
        location = self.manager.get_tensor_location("test_tensor")
        self.assertIsNotNone(location)

    def test_get_tensor_location(self):
        """Test getting tensor location."""
        # Initially, tensor location should be unknown
        location = self.manager.get_tensor_location("nonexistent_tensor")
        self.assertEqual(location, "unknown")


class TestPredictiveMemoryOptimization(unittest.TestCase):
    """Test cases for the PredictiveMemoryOptimization class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.config = {
            'enable_predictive_management': True,
            'prediction_horizon_seconds': 30,
            'access_history_window_size': 100,
            'memory_prediction_threshold': 0.9,
            'proactive_management_interval': 0.1,  # Short interval for testing
            'offload_directory': tempfile.mkdtemp()
        }
        self.optimization = PredictiveMemoryOptimization(self.config)

    def tearDown(self):
        """Clean up after each test method."""
        # Remove the temporary directory
        shutil.rmtree(self.config['offload_directory'])

    def test_start_stop_optimization(self):
        """Test starting and stopping the optimization."""
        # Start optimization
        result = self.optimization.start_optimization()
        self.assertTrue(result)

        # Stop optimization
        result = self.optimization.stop_optimization()
        self.assertTrue(result)

    def test_record_tensor_access(self):
        """Test recording tensor access."""
        # Create a dummy tensor
        tensor = torch.randn(5, 5)
        
        # Record access
        result = self.optimization.record_tensor_access("test_tensor", tensor, "read")
        self.assertTrue(result)

    def test_get_prediction_for_tensor(self):
        """Test getting prediction for a tensor."""
        # Initially, prediction should be 0.0 for unknown tensor
        prob = self.optimization.get_prediction_for_tensor("unknown_tensor")
        self.assertEqual(prob, 0.0)

    def test_get_memory_optimization_stats(self):
        """Test getting memory optimization stats."""
        stats = self.optimization.get_memory_optimization_stats()
        
        # Check that expected keys are present
        self.assertIn("enabled", stats)
        self.assertIn("active", stats)
        self.assertIn("tracked_tensors", stats)
        self.assertIn("access_history_size", stats)
        self.assertIn("prediction_horizon", stats)
        self.assertIn("proactive_interval", stats)
        self.assertIn("memory_threshold", stats)

    def test_disabled_optimization(self):
        """Test optimization when it's disabled."""
        # Create optimization with disabled config
        disabled_config = self.config.copy()
        disabled_config['enable_predictive_management'] = False
        disabled_opt = PredictiveMemoryOptimization(disabled_config)
        
        # All operations should succeed even when disabled
        self.assertTrue(disabled_opt.start_optimization())
        self.assertTrue(disabled_opt.record_tensor_access("test", torch.randn(2, 2)))
        self.assertEqual(disabled_opt.get_prediction_for_tensor("test"), 0.0)
        
        stats = disabled_opt.get_memory_optimization_stats()
        self.assertEqual(stats["enabled"], False)


if __name__ == '__main__':
    unittest.main()