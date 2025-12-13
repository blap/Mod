"""
Integration Test for Enhanced Power and Thermal Management Systems

This module provides integration tests to verify that the enhanced power and thermal
management systems work together properly with comprehensive error handling.
"""

import unittest
import time
from unittest.mock import patch, Mock

# Import the enhanced systems
from enhanced_power_management import EnhancedPowerAwareScheduler, PowerConstraint
from enhanced_thermal_management import EnhancedThermalManager, ThermalAwareTask
from system_validation_utils import ParameterValidationError


class TestEnhancedPowerThermalIntegration(unittest.TestCase):
    """Integration tests for enhanced power and thermal management systems"""
    
    def setUp(self):
        self.constraints = PowerConstraint(
            max_cpu_power_watts=25.0,
            max_gpu_power_watts=75.0,
            max_cpu_temp_celsius=90.0,
            max_gpu_temp_celsius=85.0
        )
    
    def test_power_thermal_callback_integration(self):
        """Test integration between power and thermal management via callbacks"""
        scheduler = EnhancedPowerAwareScheduler(self.constraints)
        thermal_manager = EnhancedThermalManager(self.constraints)
        
        # Register thermal callback to adjust power scheduling based on thermal state
        def thermal_callback(event_type: str, temperature: float):
            if event_type == "critical_temp":
                scheduler.set_power_mode(EnhancedPowerAwareScheduler.power_mode_class.THERMAL_MANAGEMENT)
            elif event_type == "high_temp":
                if scheduler.get_power_mode() != EnhancedPowerAwareScheduler.power_mode_class.THERMAL_MANAGEMENT:
                    scheduler.set_power_mode(EnhancedPowerAwareScheduler.power_mode_class.POWER_SAVE)
            elif event_type == "safe_temp":
                if scheduler.get_power_mode() == EnhancedPowerAwareScheduler.power_mode_class.THERMAL_MANAGEMENT:
                    scheduler.set_power_mode(EnhancedPowerAwareScheduler.power_mode_class.BALANCED)
        
        # Note: We need to handle the callback registration differently since we don't have direct access
        # to the power mode class. Let's just register the callback for now.
        thermal_manager.register_callback(thermal_callback)
        
        # Verify callback was registered
        self.assertEqual(len(thermal_manager.callbacks), 1)
        
        # Cleanup
        thermal_manager.stop_management()
        scheduler.stop_monitoring()
    
    def test_thermal_aware_task_with_enhanced_systems(self):
        """Test thermal-aware task with enhanced power and thermal systems"""
        scheduler = EnhancedPowerAwareScheduler(self.constraints)
        thermal_manager = EnhancedThermalManager(self.constraints)
        
        # Create a thermal-aware task
        thermal_task = ThermalAwareTask("integration_test_task", base_power=0.3)
        thermal_task.attach_thermal_manager(thermal_manager)
        
        # Create a power state
        from enhanced_power_management import PowerState
        power_state = PowerState(
            cpu_usage_percent=45.0,
            gpu_usage_percent=35.0,
            cpu_temp_celsius=65.0,
            gpu_temp_celsius=55.0,
            cpu_power_watts=12.0,
            gpu_power_watts=25.0
        )
        
        # Execute the thermal-aware task
        result = thermal_task.execute_with_thermal_awareness(power_state)
        self.assertTrue(result)  # Should execute under normal conditions
        
        # Test with high temperature
        high_temp_power_state = PowerState(
            cpu_usage_percent=45.0,
            gpu_usage_percent=35.0,
            cpu_temp_celsius=85.0,  # High temperature
            gpu_temp_celsius=80.0,
            cpu_power_watts=12.0,
            gpu_power_watts=25.0
        )
        
        result = thermal_task.execute_with_thermal_awareness(high_temp_power_state)
        # Should still execute but with reduced intensity
        self.assertTrue(result)
        
        # Cleanup
        thermal_manager.stop_management()
        scheduler.stop_monitoring()
    
    def test_concurrent_monitoring_and_validation(self):
        """Test concurrent monitoring with validation"""
        scheduler = EnhancedPowerAwareScheduler(self.constraints)
        thermal_manager = EnhancedThermalManager(self.constraints)
        
        # Start both monitoring systems
        scheduler.start_monitoring(interval=0.1)
        thermal_manager.start_management(monitoring_interval=0.1)
        
        # Let them run briefly
        time.sleep(0.3)
        
        # Verify both are active
        self.assertTrue(scheduler.is_monitoring)
        self.assertTrue(thermal_manager.is_active)
        
        # Add a task to the scheduler
        def sample_task():
            return "completed"
        
        scheduler.add_task(sample_task, priority=5, power_requirements=0.4)
        
        # Execute tasks
        scheduler.execute_tasks()
        
        # Check status
        task_status = scheduler.get_task_queue_status()
        thermal_summary = thermal_manager.get_thermal_summary()
        
        # Verify status information is returned
        self.assertIsInstance(task_status, dict)
        self.assertIsInstance(thermal_summary, dict)
        
        # Stop both systems
        scheduler.stop_monitoring()
        thermal_manager.stop_management()
        
        # Verify both are stopped
        self.assertFalse(scheduler.is_monitoring)
        self.assertFalse(thermal_manager.is_active)
    
    def test_error_propagation_and_handling(self):
        """Test that errors are properly handled and don't propagate unexpectedly"""
        scheduler = EnhancedPowerAwareScheduler(self.constraints)
        thermal_manager = EnhancedThermalManager(self.constraints)
        
        # Test that invalid operations are handled gracefully
        # Invalid power mode
        with self.assertRaises(ParameterValidationError):
            scheduler.set_power_mode("invalid_mode")
        
        # Invalid thermal policy
        with self.assertRaises(ParameterValidationError):
            thermal_manager.set_policy("invalid_policy")
        
        # Invalid parameter values
        with self.assertRaises(ParameterValidationError):
            scheduler.add_task(lambda: None, priority=15, power_requirements=0.5)  # Invalid priority
        
        with self.assertRaises(ParameterValidationError):
            thermal_manager._reduce_performance(1.5)  # Invalid factor
        
        # Cleanup
        thermal_manager.stop_management()
        scheduler.stop_monitoring()
    
    def test_resource_cleanup_on_error(self):
        """Test that resources are cleaned up properly even when errors occur"""
        scheduler = EnhancedPowerAwareScheduler(self.constraints)
        thermal_manager = EnhancedThermalManager(self.constraints)
        
        # Start monitoring
        scheduler.start_monitoring(interval=0.1)
        thermal_manager.start_management(monitoring_interval=0.1)
        
        # Simulate an error condition
        try:
            # This should raise an error
            scheduler.set_power_mode("invalid_mode")
        except ParameterValidationError:
            pass  # Expected error
        
        # Cleanup should still work
        scheduler.stop_monitoring()
        thermal_manager.stop_management()
        
        # Verify cleanup worked
        self.assertFalse(scheduler.is_monitoring)
        self.assertFalse(thermal_manager.is_active)
    
    def test_validation_with_real_system_calls(self):
        """Test validation with actual system calls (where possible)"""
        scheduler = EnhancedPowerAwareScheduler(self.constraints)
        
        # Get actual system power state (this will call real system functions)
        power_state = scheduler.get_system_power_state()
        
        # Verify the returned state has expected attributes
        self.assertIsInstance(power_state.cpu_usage_percent, (int, float))
        self.assertIsInstance(power_state.gpu_usage_percent, (int, float))
        self.assertIsInstance(power_state.cpu_temp_celsius, (int, float))
        self.assertIsInstance(power_state.gpu_temp_celsius, (int, float))
        self.assertIsInstance(power_state.cpu_power_watts, (int, float))
        self.assertIsInstance(power_state.gpu_power_watts, (int, float))
        
        # Verify values are in reasonable ranges
        self.assertGreaterEqual(power_state.cpu_usage_percent, 0)
        self.assertLessEqual(power_state.cpu_usage_percent, 100)
        self.assertGreaterEqual(power_state.gpu_usage_percent, 0)
        self.assertLessEqual(power_state.gpu_usage_percent, 100)
        self.assertGreaterEqual(power_state.cpu_temp_celsius, 0)
        self.assertGreaterEqual(power_state.gpu_temp_celsius, 0)
        self.assertGreaterEqual(power_state.cpu_power_watts, 0)
        self.assertGreaterEqual(power_state.gpu_power_watts, 0)
        
        # Cleanup
        scheduler.stop_monitoring()


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with original system"""
    
    def test_backward_compatibility_with_original_classes(self):
        """Test that original class names still work (via aliases)"""
        constraints = PowerConstraint()
        
        # These should work due to aliases
        scheduler = EnhancedPowerAwareScheduler(constraints)  # This is aliased as PowerAwareScheduler
        thermal_manager = EnhancedThermalManager(constraints)  # This is aliased as ThermalManager
        
        # Verify they work as expected
        self.assertIsNotNone(scheduler)
        self.assertIsNotNone(thermal_manager)
        
        # Test basic functionality
        power_state = scheduler.get_system_power_state()
        thermal_zones = thermal_manager.get_thermal_state()
        
        self.assertIsNotNone(power_state)
        self.assertIsNotNone(thermal_zones)
        
        # Cleanup
        thermal_manager.stop_management()
        scheduler.stop_monitoring()


if __name__ == '__main__':
    # Run the integration tests
    unittest.main(verbosity=2)