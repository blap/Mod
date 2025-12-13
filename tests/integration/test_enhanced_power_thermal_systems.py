"""
Comprehensive Tests for Enhanced Power and Thermal Management Systems

This module provides comprehensive tests for the enhanced power and thermal
management systems with error handling and validation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import sys
from typing import Dict, Any

# Import the enhanced systems
from enhanced_power_management import (
    EnhancedPowerAwareScheduler, PowerConstraint, PowerState, 
    PowerMode, TaskPriority, create_power_efficient_task
)
from enhanced_thermal_management import (
    EnhancedThermalManager, ThermalZone, CoolingDevice, 
    ThermalPolicy, ThermalAwareTask
)
from system_validation_utils import (
    SystemValidator, ErrorHandlingManager, ValidationResult, 
    ValidationError, PowerValidationError, ThermalValidationError, 
    ParameterValidationError, ResourceCleanupManager
)


class TestSystemValidationUtils(unittest.TestCase):
    """Test cases for system validation utilities"""
    
    def setUp(self):
        self.validator = SystemValidator()
        self.error_handler = ErrorHandlingManager()
    
    def test_validate_power_thresholds_valid(self):
        """Test that valid power thresholds pass validation"""
        result = self.validator.validate_power_thresholds(
            cpu_power=15.0, gpu_power=20.0,
            max_cpu_power=25.0, max_gpu_power=75.0
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.message, "Power thresholds are within acceptable limits")
    
    def test_validate_power_thresholds_invalid_types(self):
        """Test validation with invalid types"""
        result = self.validator.validate_power_thresholds(
            cpu_power="invalid", gpu_power=20.0,
            max_cpu_power=25.0, max_gpu_power=75.0
        )
        self.assertFalse(result.is_valid)
        self.assertIn("must be numeric", result.message)
    
    def test_validate_power_thresholds_negative_values(self):
        """Test validation with negative power values"""
        result = self.validator.validate_power_thresholds(
            cpu_power=-5.0, gpu_power=20.0,
            max_cpu_power=25.0, max_gpu_power=75.0
        )
        self.assertFalse(result.is_valid)
        self.assertIn("must be non-negative", result.message)
    
    def test_validate_power_thresholds_exceeding_limits(self):
        """Test validation with power exceeding limits"""
        result = self.validator.validate_power_thresholds(
            cpu_power=50.0, gpu_power=100.0,
            max_cpu_power=25.0, max_gpu_power=75.0
        )
        self.assertFalse(result.is_valid)
        self.assertIn("exceeds maximum", result.message)
    
    def test_validate_thermal_thresholds_valid(self):
        """Test that valid thermal thresholds pass validation"""
        result = self.validator.validate_thermal_thresholds(
            cpu_temp=70.0, gpu_temp=65.0,
            max_cpu_temp=90.0, max_gpu_temp=85.0
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.message, "Thermal thresholds are within acceptable limits")
    
    def test_validate_thermal_thresholds_invalid_types(self):
        """Test validation with invalid types"""
        result = self.validator.validate_thermal_thresholds(
            cpu_temp="invalid", gpu_temp=65.0,
            max_cpu_temp=90.0, max_gpu_temp=85.0
        )
        self.assertFalse(result.is_valid)
        self.assertIn("must be numeric", result.message)
    
    def test_validate_thermal_thresholds_out_of_range(self):
        """Test validation with temperatures out of reasonable range"""
        result = self.validator.validate_thermal_thresholds(
            cpu_temp=300.0, gpu_temp=65.0,
            max_cpu_temp=90.0, max_gpu_temp=85.0
        )
        self.assertFalse(result.is_valid)
        self.assertIn("outside reasonable range", result.message)
    
    def test_validate_thermal_thresholds_exceeding_limits(self):
        """Test validation with temperatures exceeding limits"""
        result = self.validator.validate_thermal_thresholds(
            cpu_temp=95.0, gpu_temp=65.0,
            max_cpu_temp=90.0, max_gpu_temp=85.0
        )
        self.assertFalse(result.is_valid)
        self.assertIn("exceeds maximum", result.message)
    
    def test_validate_parameter_range_valid(self):
        """Test that valid parameter ranges pass validation"""
        result = self.validator.validate_parameter_range(
            value=5.0, min_val=1.0, max_val=10.0, param_name="test_param"
        )
        self.assertTrue(result.is_valid)
        self.assertIn("is within valid range", result.message)
    
    def test_validate_parameter_range_invalid_types(self):
        """Test validation with invalid parameter types"""
        result = self.validator.validate_parameter_range(
            value="invalid", min_val=1.0, max_val=10.0, param_name="test_param"
        )
        self.assertFalse(result.is_valid)
        self.assertIn("must be numeric", result.message)
    
    def test_validate_parameter_range_below_minimum(self):
        """Test validation with parameter below minimum"""
        result = self.validator.validate_parameter_range(
            value=0.5, min_val=1.0, max_val=10.0, param_name="test_param"
        )
        self.assertFalse(result.is_valid)
        self.assertIn("is below minimum", result.message)
    
    def test_validate_parameter_range_above_maximum(self):
        """Test validation with parameter above maximum"""
        result = self.validator.validate_parameter_range(
            value=15.0, min_val=1.0, max_val=10.0, param_name="test_param"
        )
        self.assertFalse(result.is_valid)
        self.assertIn("exceeds maximum", result.message)


class TestEnhancedPowerAwareScheduler(unittest.TestCase):
    """Test cases for enhanced power-aware scheduler"""
    
    def setUp(self):
        self.constraints = PowerConstraint(
            max_cpu_power_watts=25.0,
            max_gpu_power_watts=75.0,
            max_cpu_temp_celsius=90.0,
            max_gpu_temp_celsius=85.0
        )
        self.scheduler = EnhancedPowerAwareScheduler(self.constraints)
    
    def test_initialization_with_valid_constraints(self):
        """Test initialization with valid constraints"""
        self.assertEqual(self.scheduler.constraints.max_cpu_power_watts, 25.0)
        self.assertEqual(self.scheduler.constraints.max_gpu_power_watts, 75.0)
        self.assertEqual(self.scheduler.power_mode, PowerMode.BALANCED)
    
    def test_initialization_with_invalid_constraints(self):
        """Test initialization with invalid constraints raises error"""
        invalid_constraints = PowerConstraint(max_cpu_power_watts=-5.0)  # Invalid negative value
        with self.assertRaises(ParameterValidationError):
            EnhancedPowerAwareScheduler(invalid_constraints)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.cpu_freq')
    def test_get_system_power_state(self, mock_freq, mock_cpu_percent):
        """Test getting system power state with mocked hardware"""
        mock_cpu_percent.return_value = 50.0
        mock_freq.return_value = Mock()
        mock_freq.return_value.current = 2400.0
        mock_freq.return_value.max = 4200.0

        with patch.object(self.scheduler, '_get_gpu_info', return_value=None):
            state = self.scheduler.get_system_power_state()

        self.assertIsInstance(state, PowerState)
        self.assertGreaterEqual(state.cpu_usage_percent, 0)
        self.assertGreaterEqual(state.cpu_temp_celsius, 0)
        self.assertGreaterEqual(state.cpu_power_watts, 0)
    
    def test_add_task_with_valid_parameters(self):
        """Test adding a task with valid parameters"""
        def sample_task():
            return "task completed"
        
        self.scheduler.add_task(sample_task, priority=5, power_requirements=0.5)
        
        status = self.scheduler.get_task_queue_status()
        self.assertEqual(status['pending_tasks'], 1)
    
    def test_add_task_with_invalid_priority(self):
        """Test adding a task with invalid priority raises error"""
        def sample_task():
            return "task completed"
        
        with self.assertRaises(ParameterValidationError):
            self.scheduler.add_task(sample_task, priority=15, power_requirements=0.5)  # Priority > 10
    
    def test_add_task_with_invalid_power_requirements(self):
        """Test adding a task with invalid power requirements raises error"""
        def sample_task():
            return "task completed"
        
        with self.assertRaises(ParameterValidationError):
            self.scheduler.add_task(sample_task, priority=5, power_requirements=1.5)  # Power req > 1.0
    
    def test_should_execute_task_normal_conditions(self):
        """Test task execution decision under normal conditions"""
        task = {
            'id': 1,
            'function': lambda: None,
            'priority': 5,
            'power_requirements': 0.5,
            'created_at': time.time()
        }
        
        # Set normal power state
        self.scheduler.power_state = PowerState(
            cpu_usage_percent=50.0,
            gpu_usage_percent=40.0,
            cpu_temp_celsius=60.0,
            gpu_temp_celsius=50.0,
            cpu_power_watts=10.0,
            gpu_power_watts=20.0
        )
        
        result = self.scheduler.should_execute_task(task)
        self.assertTrue(result)
    
    def test_should_execute_task_high_temperature(self):
        """Test task execution decision under high temperature"""
        task = {
            'id': 1,
            'function': lambda: None,
            'priority': 3,  # Low priority
            'power_requirements': 0.5,
            'created_at': time.time()
        }
        
        # Set high temperature state
        self.scheduler.power_state = PowerState(
            cpu_usage_percent=50.0,
            gpu_usage_percent=40.0,
            cpu_temp_celsius=85.0,  # High temperature
            gpu_temp_celsius=50.0,
            cpu_power_watts=10.0,
            gpu_power_watts=20.0
        )
        
        result = self.scheduler.should_execute_task(task)
        self.assertFalse(result)  # Should not execute due to high temperature
    
    def test_should_execute_task_thermal_management_mode(self):
        """Test task execution decision in thermal management mode"""
        high_priority_task = {
            'id': 1,
            'function': lambda: None,
            'priority': 8,  # High priority
            'power_requirements': 0.5,
            'created_at': time.time()
        }
        
        low_priority_task = {
            'id': 2,
            'function': lambda: None,
            'priority': 2,  # Low priority
            'power_requirements': 0.5,
            'created_at': time.time()
        }
        
        # Set thermal management mode
        self.scheduler.power_mode = PowerMode.THERMAL_MANAGEMENT
        
        # High priority task should execute
        result_high = self.scheduler.should_execute_task(high_priority_task)
        self.assertTrue(result_high)
        
        # Low priority task should not execute
        result_low = self.scheduler.should_execute_task(low_priority_task)
        self.assertFalse(result_low)
    
    def test_set_power_mode_with_valid_mode(self):
        """Test setting power mode with valid mode"""
        self.scheduler.set_power_mode(PowerMode.PERFORMANCE)
        self.assertEqual(self.scheduler.power_mode, PowerMode.PERFORMANCE)
    
    def test_set_power_mode_with_invalid_mode(self):
        """Test setting power mode with invalid mode raises error"""
        with self.assertRaises(ParameterValidationError):
            self.scheduler.set_power_mode("invalid_mode")
    
    def test_start_monitoring_with_valid_interval(self):
        """Test starting monitoring with valid interval"""
        self.scheduler.start_monitoring(interval=0.5)
        self.assertTrue(self.scheduler.is_monitoring)
        self.scheduler.stop_monitoring()
    
    def test_start_monitoring_with_invalid_interval(self):
        """Test starting monitoring with invalid interval raises error"""
        with self.assertRaises(ParameterValidationError):
            self.scheduler.start_monitoring(interval=0.05)  # Too short interval


class TestEnhancedThermalManager(unittest.TestCase):
    """Test cases for enhanced thermal manager"""
    
    def setUp(self):
        self.constraints = PowerConstraint(
            max_cpu_power_watts=25.0,
            max_gpu_power_watts=75.0,
            max_cpu_temp_celsius=90.0,
            max_gpu_temp_celsius=85.0
        )
        self.thermal_manager = EnhancedThermalManager(self.constraints)
    
    def test_initialization_with_valid_constraints(self):
        """Test initialization with valid constraints"""
        self.assertEqual(self.thermal_manager.constraints.max_cpu_temp_celsius, 90.0)
        self.assertEqual(self.thermal_manager.constraints.max_gpu_temp_celsius, 85.0)
        self.assertEqual(self.thermal_manager.policy, ThermalPolicy.HYBRID)
    
    def test_initialization_with_invalid_constraints(self):
        """Test initialization with invalid constraints raises error"""
        invalid_constraints = PowerConstraint(max_cpu_temp_celsius=-10.0)  # Invalid negative value
        with self.assertRaises(ParameterValidationError):
            EnhancedThermalManager(invalid_constraints)
    
    def test_get_thermal_state(self):
        """Test getting thermal state"""
        zones = self.thermal_manager.get_thermal_state()
        self.assertIsInstance(zones, list)
        self.assertGreaterEqual(len(zones), 1)  # At least CPU zone
        
        for zone in zones:
            self.assertIsInstance(zone, ThermalZone)
            self.assertIsInstance(zone.name, str)
            self.assertIsInstance(zone.current_temp, (int, float))
    
    def test_get_cooling_state(self):
        """Test getting cooling state"""
        devices = self.thermal_manager.get_cooling_state()
        self.assertIsInstance(devices, list)
        
        for device in devices:
            self.assertIsInstance(device, CoolingDevice)
            self.assertIsInstance(device.name, str)
            self.assertIsInstance(device.current_state, int)
    
    def test_register_callback_with_valid_callback(self):
        """Test registering a valid callback"""
        def sample_callback(event_type: str, value: float):
            pass
        
        self.thermal_manager.register_callback(sample_callback)
        self.assertEqual(len(self.thermal_manager.callbacks), 1)
    
    def test_register_callback_with_invalid_callback(self):
        """Test registering an invalid callback raises error"""
        with self.assertRaises(ParameterValidationError):
            self.thermal_manager.register_callback("not_a_function")
    
    def test_adjust_cooling_with_valid_parameters(self):
        """Test adjusting cooling with valid parameters"""
        # This should not raise an exception
        self.thermal_manager.adjust_cooling("CPU", 60.0)
    
    def test_adjust_cooling_with_invalid_temperature(self):
        """Test adjusting cooling with invalid temperature raises error"""
        with self.assertRaises(ParameterValidationError):
            self.thermal_manager.adjust_cooling("CPU", -100.0)  # Too low temperature
    
    def test_adjust_cooling_with_invalid_zone_name(self):
        """Test adjusting cooling with invalid zone name raises error"""
        with self.assertRaises(ParameterValidationError):
            self.thermal_manager.adjust_cooling(123, 60.0)  # Invalid type for zone name
    
    def test_set_cooling_level_with_valid_parameters(self):
        """Test setting cooling level with valid parameters"""
        # This should not raise an exception
        self.thermal_manager._set_cooling_level("CPU Fan", 75)
    
    def test_set_cooling_level_with_invalid_level(self):
        """Test setting cooling level with invalid level raises error"""
        with self.assertRaises(ParameterValidationError):
            self.thermal_manager._set_cooling_level("CPU Fan", 150)  # Too high level
    
    def test_reduce_performance_with_valid_factor(self):
        """Test reducing performance with valid factor"""
        # This should not raise an exception
        self.thermal_manager._reduce_performance(0.2)
    
    def test_reduce_performance_with_invalid_factor(self):
        """Test reducing performance with invalid factor raises error"""
        with self.assertRaises(ParameterValidationError):
            self.thermal_manager._reduce_performance(1.5)  # Too high factor
    
    def test_start_management_with_valid_interval(self):
        """Test starting thermal management with valid interval"""
        self.thermal_manager.start_management(monitoring_interval=0.5)
        self.assertTrue(self.thermal_manager.is_active)
        self.thermal_manager.stop_management()
    
    def test_start_management_with_invalid_interval(self):
        """Test starting thermal management with invalid interval raises error"""
        with self.assertRaises(ParameterValidationError):
            self.thermal_manager.start_management(monitoring_interval=0.05)  # Too short interval
    
    def test_set_policy_with_valid_policy(self):
        """Test setting thermal policy with valid policy"""
        self.thermal_manager.set_policy(ThermalPolicy.ACTIVE)
        self.assertEqual(self.thermal_manager.policy, ThermalPolicy.ACTIVE)
    
    def test_set_policy_with_invalid_policy(self):
        """Test setting thermal policy with invalid policy raises error"""
        with self.assertRaises(ParameterValidationError):
            self.thermal_manager.set_policy("invalid_policy")


class TestThermalAwareTask(unittest.TestCase):
    """Test cases for thermal-aware task"""
    
    def test_initialization_with_valid_parameters(self):
        """Test initialization with valid parameters"""
        task = ThermalAwareTask("test_task", base_power=0.5)
        self.assertEqual(task.name, "test_task")
        self.assertEqual(task.base_power, 0.5)
    
    def test_initialization_with_invalid_name(self):
        """Test initialization with invalid name raises error"""
        with self.assertRaises(ParameterValidationError):
            ThermalAwareTask("", base_power=0.5)  # Empty name
    
    def test_initialization_with_invalid_base_power(self):
        """Test initialization with invalid base power raises error"""
        with self.assertRaises(ParameterValidationError):
            ThermalAwareTask("test_task", base_power=-1.0)  # Negative power
    
    def test_attach_thermal_manager_with_valid_manager(self):
        """Test attaching valid thermal manager"""
        task = ThermalAwareTask("test_task", base_power=0.5)
        thermal_manager = Mock(spec=EnhancedThermalManager)
        
        task.attach_thermal_manager(thermal_manager)
        self.assertEqual(task.thermal_manager, thermal_manager)
    
    def test_attach_thermal_manager_with_invalid_manager(self):
        """Test attaching invalid thermal manager raises error"""
        task = ThermalAwareTask("test_task", base_power=0.5)
        
        with self.assertRaises(ParameterValidationError):
            task.attach_thermal_manager("not_a_thermal_manager")


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling functionality"""
    
    def setUp(self):
        self.error_handler = ErrorHandlingManager()
    
    def test_handle_hardware_access_error(self):
        """Test handling hardware access errors"""
        exception = RuntimeError("Hardware not available")
        error_info = self.error_handler.handle_hardware_access_error(
            "test_operation", "test_hardware", exception
        )
        
        self.assertEqual(error_info["operation"], "test_operation")
        self.assertEqual(error_info["hardware_type"], "test_hardware")
        self.assertEqual(error_info["error_type"], "RuntimeError")
        self.assertIn("Hardware not available", error_info["error_message"])
    
    def test_handle_power_operation_error(self):
        """Test handling power operation errors"""
        exception = ValueError("Invalid power value")
        error_info = self.error_handler.handle_power_operation_error(
            "test_operation", {"power": 100}, exception
        )
        
        self.assertEqual(error_info["operation"], "test_operation")
        self.assertEqual(error_info["error_type"], "ValueError")
        self.assertIn("Invalid power value", error_info["error_message"])
    
    def test_handle_thermal_operation_error(self):
        """Test handling thermal operation errors"""
        exception = RuntimeError("Temperature sensor failed")
        error_info = self.error_handler.handle_thermal_operation_error(
            "test_operation", {"temp": 80}, exception
        )
        
        self.assertEqual(error_info["operation"], "test_operation")
        self.assertEqual(error_info["error_type"], "RuntimeError")
        self.assertIn("Temperature sensor failed", error_info["error_message"])


class TestResourceCleanup(unittest.TestCase):
    """Test cases for resource cleanup functionality"""
    
    def setUp(self):
        self.cleanup_manager = ResourceCleanupManager()
    
    def test_register_and_execute_cleanup_functions(self):
        """Test registering and executing cleanup functions"""
        cleanup_called = [False]
        
        def sample_cleanup():
            cleanup_called[0] = True
        
        self.cleanup_manager.register_cleanup_function(sample_cleanup)
        self.cleanup_manager.cleanup_all_resources()
        
        self.assertTrue(cleanup_called[0])
        self.assertEqual(len(self.cleanup_manager.cleanup_functions), 0)
    
    def test_execute_cleanup_functions_with_error(self):
        """Test executing cleanup functions when one raises an error"""
        cleanup1_called = [False]
        cleanup2_called = [False]
        
        def sample_cleanup1():
            cleanup1_called[0] = True
        
        def sample_cleanup2():
            raise RuntimeError("Cleanup error")
        
        def sample_cleanup3():
            cleanup2_called[0] = True
        
        self.cleanup_manager.register_cleanup_function(sample_cleanup1)
        self.cleanup_manager.register_cleanup_function(sample_cleanup2)
        self.cleanup_manager.register_cleanup_function(sample_cleanup3)
        
        # This should not raise an exception despite the error in cleanup2
        self.cleanup_manager.cleanup_all_resources()
        
        # Cleanup 1 and 3 should still be called despite error in 2
        self.assertTrue(cleanup1_called[0])
        self.assertTrue(cleanup2_called[0])
        self.assertEqual(len(self.cleanup_manager.cleanup_functions), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the enhanced power and thermal management systems"""
    
    def setUp(self):
        self.constraints = PowerConstraint(
            max_cpu_power_watts=25.0,
            max_gpu_power_watts=75.0,
            max_cpu_temp_celsius=90.0,
            max_gpu_temp_celsius=85.0
        )
    
    def test_power_and_thermal_system_integration(self):
        """Test integration between power and thermal management systems"""
        scheduler = EnhancedPowerAwareScheduler(self.constraints)
        thermal_manager = EnhancedThermalManager(self.constraints)
        
        # Register thermal callback to adjust power scheduling based on thermal state
        def thermal_callback(event_type: str, temperature: float):
            if event_type == "critical_temp":
                scheduler.set_power_mode(PowerMode.THERMAL_MANAGEMENT)
            elif event_type == "high_temp":
                if scheduler.get_power_mode() != PowerMode.THERMAL_MANAGEMENT:
                    scheduler.set_power_mode(PowerMode.POWER_SAVE)
            elif event_type == "safe_temp":
                if scheduler.get_power_mode() == PowerMode.THERMAL_MANAGEMENT:
                    scheduler.set_power_mode(PowerMode.BALANCED)
        
        thermal_manager.register_callback(thermal_callback)
        
        # Add a sample task
        def sample_task():
            return "task completed"
        
        scheduler.add_task(sample_task, priority=5, power_requirements=0.3)
        
        # Start both systems
        scheduler.start_monitoring(interval=0.1)
        thermal_manager.start_management(monitoring_interval=0.1)
        
        # Let them run briefly
        time.sleep(0.2)
        
        # Stop both systems
        scheduler.stop_monitoring()
        thermal_manager.stop_management()
        
        # Verify both systems were active
        self.assertFalse(scheduler.is_monitoring)
        self.assertFalse(thermal_manager.is_active)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)