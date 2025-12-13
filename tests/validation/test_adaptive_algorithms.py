"""
Tests for adaptive_algorithms.py module
"""
import unittest
import time
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from power_management import PowerConstraint, PowerState
from thermal_management import ThermalManager
from adaptive_algorithms import (
    AdaptiveParameters,
    AdaptationStrategy,
    AdaptiveController,
    LoadBalancer,
    AdaptiveModelWrapper
)


class TestAdaptiveParameters(unittest.TestCase):
    """Test AdaptiveParameters dataclass"""
    
    def test_default_values(self):
        """Test default values of AdaptiveParameters"""
        params = AdaptiveParameters()
        self.assertEqual(params.performance_factor, 1.0)
        self.assertEqual(params.batch_size_factor, 1.0)
        self.assertEqual(params.frequency_factor, 1.0)
        self.assertEqual(params.resource_allocation, 1.0)
        self.assertEqual(params.execution_delay, 0.0)
        
    def test_custom_values(self):
        """Test custom values of AdaptiveParameters"""
        params = AdaptiveParameters(
            performance_factor=0.5,
            batch_size_factor=0.8,
            frequency_factor=0.6,
            resource_allocation=0.7,
            execution_delay=0.1
        )
        self.assertEqual(params.performance_factor, 0.5)
        self.assertEqual(params.batch_size_factor, 0.8)
        self.assertEqual(params.frequency_factor, 0.6)
        self.assertEqual(params.resource_allocation, 0.7)
        self.assertEqual(params.execution_delay, 0.1)


class TestAdaptationStrategy(unittest.TestCase):
    """Test AdaptationStrategy enum"""
    
    def test_strategy_values(self):
        """Test that all strategy values are correct"""
        self.assertEqual(AdaptationStrategy.PERFORMANCE_FIRST.value, "performance_first")
        self.assertEqual(AdaptationStrategy.POWER_EFFICIENT.value, "power_efficient")
        self.assertEqual(AdaptationStrategy.THERMAL_AWARE.value, "thermal_aware")
        self.assertEqual(AdaptationStrategy.BALANCED.value, "balanced")


class TestAdaptiveController(unittest.TestCase):
    """Test AdaptiveController class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.constraints = PowerConstraint(
            max_cpu_power_watts=25.0,
            max_gpu_power_watts=75.0,
            max_cpu_temp_celsius=90.0,
            max_gpu_temp_celsius=85.0,
            max_cpu_usage_percent=90.0,
            max_gpu_usage_percent=85.0
        )
        self.controller = AdaptiveController(self.constraints)
        
    def test_initialization(self):
        """Test AdaptiveController initialization"""
        self.assertEqual(self.controller.constraints, self.constraints)
        self.assertEqual(self.controller.current_parameters, AdaptiveParameters())
        self.assertEqual(self.controller.adaptation_strategy, AdaptationStrategy.BALANCED)
        self.assertFalse(self.controller.is_active)
        self.assertEqual(len(self.controller.parameter_history), 0)
        self.assertEqual(self.controller.max_history_size, 100)
        
    def test_update_parameters_performance_first_strategy(self):
        """Test update_parameters with PERFORMANCE_FIRST strategy"""
        power_state = PowerState(
            cpu_usage_percent=95.0,
            gpu_usage_percent=90.0,
            cpu_temp_celsius=85.0,
            gpu_temp_celsius=80.0,
            cpu_power_watts=24.0,
            gpu_power_watts=70.0,
            timestamp=time.time()
        )
        
        self.controller.adaptation_strategy = AdaptationStrategy.PERFORMANCE_FIRST
        params = self.controller.update_parameters(power_state)
        
        # With performance_first strategy, parameters should not be reduced as much
        self.assertGreater(params.performance_factor, 0.5)
        self.assertGreater(params.batch_size_factor, 0.5)
        self.assertGreater(params.frequency_factor, 0.3)
        
    def test_update_parameters_power_efficient_strategy(self):
        """Test update_parameters with POWER_EFFICIENT strategy"""
        power_state = PowerState(
            cpu_usage_percent=95.0,
            gpu_usage_percent=90.0,
            cpu_temp_celsius=85.0,
            gpu_temp_celsius=80.0,
            cpu_power_watts=24.0,
            gpu_power_watts=70.0,
            timestamp=time.time()
        )
        
        self.controller.adaptation_strategy = AdaptationStrategy.POWER_EFFICIENT
        params = self.controller.update_parameters(power_state)
        
        # With power_efficient strategy, parameters should be more reduced
        self.assertLess(params.performance_factor, 0.7)
        self.assertLess(params.batch_size_factor, 0.7)
        self.assertLess(params.frequency_factor, 0.7)
        
    def test_update_parameters_thermal_aware_strategy(self):
        """Test update_parameters with THERMAL_AWARE strategy"""
        power_state = PowerState(
            cpu_usage_percent=50.0,
            gpu_usage_percent=50.0,
            cpu_temp_celsius=85.0,  # High temperature
            gpu_temp_celsius=80.0,
            cpu_power_watts=15.0,
            gpu_power_watts=40.0,
            timestamp=time.time()
        )
        
        self.controller.adaptation_strategy = AdaptationStrategy.THERMAL_AWARE
        params = self.controller.update_parameters(power_state)
        
        # With thermal_aware strategy, parameters should be adjusted based on temperature
        self.assertLess(params.performance_factor, 0.8)
        self.assertLess(params.batch_size_factor, 0.8)
        self.assertLess(params.frequency_factor, 0.8)
        
        # Check if execution delay is added for high thermal stress
        if power_state.cpu_temp_celsius / self.constraints.max_cpu_temp_celsius > 0.9:
            self.assertGreater(params.execution_delay, 0.0)
            
    def test_get_historical_trend(self):
        """Test getting historical trend"""
        # Add some history
        for i in range(10):
            timestamp = time.time() - (10 - i)
            params = AdaptiveParameters(
                performance_factor=0.5 + (i * 0.05),
                batch_size_factor=0.6 + (i * 0.04),
                frequency_factor=0.4 + (i * 0.06)
            )
            self.controller._add_to_history(timestamp, params)
        
        trend = self.controller.get_historical_trend()
        self.assertIsNotNone(trend)
        self.assertIn('performance_trend', trend)
        self.assertIn('batch_trend', trend)
        self.assertIn('frequency_trend', trend)
        self.assertIn('avg_performance', trend)
        self.assertIn('avg_batch', trend)
        self.assertIn('avg_frequency', trend)
        
    def test_adapt_model_behavior(self):
        """Test adapt_model_behavior method"""
        def mock_model_func(*args, **kwargs):
            return f"Result with args: {args}, kwargs: {kwargs}"

        # Set some parameters that would modify behavior
        self.controller.current_parameters = AdaptiveParameters(
            execution_delay=0.01,  # Small delay for testing
            batch_size_factor=0.5
        )

        result = self.controller.adapt_model_behavior(
            mock_model_func,
            "input1",
            batch_size=10,
            other_param="value"
        )

        # Check that batch size was adjusted in kwargs
        expected_batch_size = 5  # 10 * 0.5
        self.assertIn(f"'batch_size': {expected_batch_size}", result)
        
    def test_get_adaptation_summary(self):
        """Test getting adaptation summary"""
        summary = self.controller.get_adaptation_summary()
        
        self.assertIn("current_parameters", summary)
        self.assertIn("strategy", summary)
        self.assertIn("active", summary)
        self.assertIn("history_size", summary)
        self.assertIn("trend", summary)
        
        # Check structure of current_parameters
        current_params = summary["current_parameters"]
        self.assertIn("performance_factor", current_params)
        self.assertIn("batch_size_factor", current_params)
        self.assertIn("frequency_factor", current_params)
        self.assertIn("resource_allocation", current_params)
        self.assertIn("execution_delay", current_params)


class TestLoadBalancer(unittest.TestCase):
    """Test LoadBalancer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.constraints = PowerConstraint()
        self.load_balancer = LoadBalancer(self.constraints)
        
    def test_distribute_load_severe_constraint(self):
        """Test load distribution with severe constraints"""
        power_state = PowerState(
            cpu_usage_percent=95.0,
            gpu_usage_percent=95.0,
            cpu_temp_celsius=88.0,
            gpu_temp_celsius=83.0,
            cpu_power_watts=24.0,
            gpu_power_watts=74.0,
            timestamp=time.time()
        )
        
        workloads = [("work1", lambda x: x), ("work2", lambda x: x)]
        distribution = self.load_balancer.distribute_load(workloads, power_state)
        
        # With severe constraints, distribution factors should be low
        for factor in distribution.values():
            self.assertLess(factor, 0.3)
            
    def test_distribute_load_mild_constraint(self):
        """Test load distribution with mild constraints"""
        power_state = PowerState(
            cpu_usage_percent=65.0,
            gpu_usage_percent=60.0,
            cpu_temp_celsius=60.0,
            gpu_temp_celsius=55.0,
            cpu_power_watts=15.0,
            gpu_power_watts=40.0,
            timestamp=time.time()
        )
        
        workloads = [("work1", lambda x: x), ("work2", lambda x: x)]
        distribution = self.load_balancer.distribute_load(workloads, power_state)
        
        # With mild constraints, distribution factors should be higher
        for factor in distribution.values():
            self.assertGreater(factor, 0.2)
            
    def test_execute_workloads(self):
        """Test executing workloads"""
        def workload_func(factor):
            return f"Processed with factor {factor}"
        
        workloads = [("work1", workload_func), ("work2", workload_func)]
        power_state = PowerState(
            cpu_usage_percent=50.0,
            gpu_usage_percent=45.0,
            cpu_temp_celsius=60.0,
            gpu_temp_celsius=55.0,
            cpu_power_watts=12.0,
            gpu_power_watts=35.0,
            timestamp=time.time()
        )
        
        results = self.load_balancer.execute_workloads(workloads, power_state)
        
        self.assertEqual(len(results), 2)
        for workload_id, result in results.items():
            self.assertIn(workload_id, ["work1", "work2"])
            self.assertIn("result", result)
            self.assertIn("factor", result)
            self.assertIn("status", result)
            self.assertEqual(result["status"], "success")


class TestAdaptiveModelWrapper(unittest.TestCase):
    """Test AdaptiveModelWrapper class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.constraints = PowerConstraint()
        self.model = Mock()
        self.adaptive_model = AdaptiveModelWrapper(self.model, self.constraints)
        
    def test_predict(self):
        """Test adaptive prediction"""
        power_state = PowerState(
            cpu_usage_percent=60.0,
            gpu_usage_percent=55.0,
            cpu_temp_celsius=65.0,
            gpu_temp_celsius=60.0,
            cpu_power_watts=15.0,
            gpu_power_watts=40.0,
            timestamp=time.time()
        )
        
        input_data = [1, 2, 3, 4, 5]
        result = self.adaptive_model.predict(input_data, power_state, batch_size=32)
        
        # Check that result has expected structure
        self.assertIn("prediction", result)
        self.assertIn("parameters_used", result)
        self.assertIn("performance_factor", result["parameters_used"])
        self.assertIn("batch_size_factor", result["parameters_used"])
        self.assertIn("frequency_factor", result["parameters_used"])
        
    def test_fit(self):
        """Test adaptive training"""
        power_state = PowerState(
            cpu_usage_percent=70.0,
            gpu_usage_percent=65.0,
            cpu_temp_celsius=70.0,
            gpu_temp_celsius=65.0,
            cpu_power_watts=18.0,
            gpu_power_watts=50.0,
            timestamp=time.time()
        )
        
        training_data = [[1, 2], [3, 4], [5, 6]]
        result = self.adaptive_model.fit(training_data, power_state, epochs=10, batch_size=16)
        
        # Check that result has expected structure
        self.assertIn("final_loss", result)
        self.assertIn("epochs_trained", result)
        self.assertIn("parameters_used", result)
        self.assertIn("performance_factor", result["parameters_used"])
        self.assertIn("batch_size_factor", result["parameters_used"])
        self.assertIn("frequency_factor", result["parameters_used"])


if __name__ == "__main__":
    unittest.main()