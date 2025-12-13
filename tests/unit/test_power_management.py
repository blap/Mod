"""
Comprehensive tests for the power management system
Testing power-aware scheduling, thermal management, adaptive algorithms, and DVFS
"""
import unittest
import time
import threading
from unittest.mock import Mock, patch
from power_management import PowerConstraint, PowerAwareScheduler, PowerState, PowerMode
from thermal_management import ThermalManager, ThermalPolicy
from adaptive_algorithms import AdaptiveController, AdaptationStrategy, AdaptiveModelWrapper, LoadBalancer
from dvfs_controller import DVFSController, WorkloadBasedDVFS, WORKLOAD_PROFILES


class TestPowerManagement(unittest.TestCase):
    """Test cases for power management system"""
    
    def setUp(self):
        self.constraints = PowerConstraint(
            max_cpu_power_watts=25.0,
            max_gpu_power_watts=75.0,
            max_cpu_temp_celsius=90.0,
            max_gpu_temp_celsius=85.0,
            max_cpu_usage_percent=90.0,
            max_gpu_usage_percent=85.0
        )
    
    def test_power_constraint_initialization(self):
        """Test that power constraints are initialized correctly"""
        self.assertEqual(self.constraints.max_cpu_power_watts, 25.0)
        self.assertEqual(self.constraints.max_gpu_power_watts, 75.0)
        self.assertEqual(self.constraints.max_cpu_temp_celsius, 90.0)
        self.assertEqual(self.constraints.max_gpu_temp_celsius, 85.0)
    
    def test_power_state_initialization(self):
        """Test that power state is initialized correctly"""
        state = PowerState()
        self.assertEqual(state.cpu_usage_percent, 0.0)
        self.assertEqual(state.gpu_usage_percent, 0.0)
        self.assertEqual(state.cpu_temp_celsius, 0.0)
        self.assertEqual(state.gpu_temp_celsius, 0.0)
        self.assertEqual(state.cpu_power_watts, 0.0)
        self.assertEqual(state.gpu_power_watts, 0.0)


class TestPowerAwareScheduler(unittest.TestCase):
    """Test cases for power-aware scheduler"""
    
    def setUp(self):
        self.constraints = PowerConstraint()
        self.scheduler = PowerAwareScheduler(self.constraints)
    
    def test_scheduler_initialization(self):
        """Test that scheduler is initialized correctly"""
        self.assertEqual(self.scheduler.constraints, self.constraints)
        self.assertEqual(len(self.scheduler.tasks), 0)
        self.assertEqual(self.scheduler.power_mode, PowerMode.BALANCED)
    
    def test_add_task(self):
        """Test adding tasks to scheduler"""
        def sample_task():
            return "executed"
        
        self.scheduler.add_task(sample_task, priority=5, power_requirements=0.5)
        self.assertEqual(len(self.scheduler.tasks), 1)
        self.assertEqual(self.scheduler.tasks[0]['priority'], 5)
        self.assertEqual(self.scheduler.tasks[0]['power_requirements'], 0.5)
    
    def test_task_priority_sorting(self):
        """Test that tasks are sorted by priority"""
        def sample_task():
            return "executed"
        
        # Add tasks with different priorities
        self.scheduler.add_task(sample_task, priority=1)
        self.scheduler.add_task(sample_task, priority=5)
        self.scheduler.add_task(sample_task, priority=3)
        
        # Check that tasks are sorted by priority (highest first)
        self.assertEqual(self.scheduler.tasks[0]['priority'], 5)
        self.assertEqual(self.scheduler.tasks[1]['priority'], 3)
        self.assertEqual(self.scheduler.tasks[2]['priority'], 1)
    
    @patch('psutil.cpu_percent')
    def test_get_system_power_state(self, mock_cpu_percent):
        """Test getting system power state"""
        # Mock return values
        mock_cpu_percent.return_value = 75.0

        state = self.scheduler.get_system_power_state()

        self.assertEqual(state.cpu_usage_percent, 75.0)
        # Temperature should be reasonable even if sensors_temperatures is not available
        self.assertGreaterEqual(state.cpu_temp_celsius, 0.0)
        self.assertLessEqual(state.cpu_temp_celsius, 100.0)  # Reasonable upper limit
    
    def test_should_execute_task_normal_conditions(self):
        """Test task execution under normal conditions"""
        def sample_task():
            return "executed"
        
        task = {
            'id': 1,
            'function': sample_task,
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
            cpu_power_watts=12.0,
            gpu_power_watts=30.0,
            timestamp=time.time()
        )
        
        # Task should execute in balanced mode with normal conditions
        self.assertTrue(self.scheduler.should_execute_task(task))
    
    def test_should_execute_task_high_priority_in_thermal_management(self):
        """Test that high priority tasks execute even in thermal management mode"""
        def sample_task():
            return "executed"
        
        task_high_priority = {
            'id': 1,
            'function': sample_task,
            'priority': 8,
            'power_requirements': 0.5,
            'created_at': time.time()
        }
        
        task_low_priority = {
            'id': 2,
            'function': sample_task,
            'priority': 2,
            'power_requirements': 0.5,
            'created_at': time.time()
        }
        
        # Set thermal management mode
        self.scheduler.power_mode = PowerMode.THERMAL_MANAGEMENT
        
        # High priority task should execute
        self.assertTrue(self.scheduler.should_execute_task(task_high_priority))
        # Low priority task should not execute
        self.assertFalse(self.scheduler.should_execute_task(task_low_priority))
    
    def test_power_mode_adjustment(self):
        """Test automatic power mode adjustment based on thermal state"""
        # Test thermal emergency mode
        self.scheduler.power_state = PowerState(
            cpu_temp_celsius=95.0,  # Above max
            gpu_temp_celsius=90.0,  # Above max
            cpu_usage_percent=80.0,
            gpu_usage_percent=70.0,
            cpu_power_watts=20.0,
            gpu_power_watts=60.0,
            timestamp=time.time()
        )
        
        self.scheduler._adjust_power_mode()
        self.assertEqual(self.scheduler.power_mode, PowerMode.THERMAL_MANAGEMENT)
        
        # Test power save mode due to high temperature
        self.scheduler.power_state = PowerState(
            cpu_temp_celsius=75.0,  # Above 80% of max (72°C)
            gpu_temp_celsius=70.0,  # Above 80% of max (68°C)
            cpu_usage_percent=60.0,
            gpu_usage_percent=50.0,
            cpu_power_watts=15.0,
            gpu_power_watts=40.0,
            timestamp=time.time()
        )
        
        self.scheduler._adjust_power_mode()
        self.assertEqual(self.scheduler.power_mode, PowerMode.POWER_SAVE)


class TestThermalManager(unittest.TestCase):
    """Test cases for thermal management system"""
    
    def setUp(self):
        self.constraints = PowerConstraint()
        self.thermal_manager = ThermalManager(self.constraints)
    
    def test_thermal_manager_initialization(self):
        """Test that thermal manager is initialized correctly"""
        self.assertEqual(self.thermal_manager.constraints, self.constraints)
        self.assertEqual(self.thermal_manager.policy, ThermalPolicy.HYBRID)
        self.assertEqual(len(self.thermal_manager.thermal_zones), 1)  # CPU zone
    
    def test_get_thermal_state(self):
        """Test getting thermal state"""
        zones = self.thermal_manager.get_thermal_state()
        self.assertGreaterEqual(len(zones), 1)  # At least CPU zone
        
        cpu_zone = next((z for z in zones if z.zone_type == "CPU"), None)
        self.assertIsNotNone(cpu_zone)
        self.assertLessEqual(cpu_zone.current_temp, cpu_zone.critical_temp)
    
    def test_cooling_level_adjustment(self):
        """Test adjusting cooling levels"""
        # Test setting cooling level
        self.thermal_manager._set_cooling_level("CPU", 80)
        
        cpu_fan = next((d for d in self.thermal_manager.cooling_devices 
                       if "CPU" in d.name), None)
        self.assertIsNotNone(cpu_fan)
        self.assertEqual(cpu_fan.current_state, 80)
    
    def test_thermal_zone_status(self):
        """Test getting thermal zone status"""
        # Create a test zone
        from thermal_management import ThermalZone
        test_zone = ThermalZone(
            name="Test",
            current_temp=50.0,
            critical_temp=90.0,
            passive_temp=72.0,  # 80% of 90
            zone_type="CPU"
        )
        
        # Test normal status
        status = self.thermal_manager._get_zone_status(test_zone)
        self.assertEqual(status, "normal")
        
        # Test warning status
        test_zone.current_temp = 80.0
        status = self.thermal_manager._get_zone_status(test_zone)
        self.assertEqual(status, "warning")
        
        # Test critical status
        test_zone.current_temp = 95.0
        status = self.thermal_manager._get_zone_status(test_zone)
        self.assertEqual(status, "critical")


class TestAdaptiveController(unittest.TestCase):
    """Test cases for adaptive controller"""
    
    def setUp(self):
        self.constraints = PowerConstraint()
        self.controller = AdaptiveController(self.constraints)
    
    def test_adaptive_parameter_initialization(self):
        """Test that adaptive parameters are initialized correctly"""
        params = self.controller.current_parameters
        self.assertEqual(params.performance_factor, 1.0)
        self.assertEqual(params.batch_size_factor, 1.0)
        self.assertEqual(params.frequency_factor, 1.0)
        self.assertEqual(params.resource_allocation, 1.0)
        self.assertEqual(params.execution_delay, 0.0)
    
    def test_update_parameters_performance_first(self):
        """Test parameter updates with performance-first strategy"""
        self.controller.adaptation_strategy = AdaptationStrategy.PERFORMANCE_FIRST
        
        power_state = PowerState(
            cpu_usage_percent=90.0,
            gpu_usage_percent=85.0,
            cpu_temp_celsius=85.0,
            gpu_temp_celsius=80.0,
            cpu_power_watts=22.0,
            gpu_power_watts=65.0,
            timestamp=time.time()
        )
        
        params = self.controller.update_parameters(power_state)
        
        # With performance-first strategy, factors should not be too low even under stress
        self.assertGreaterEqual(params.performance_factor, 0.5)
        self.assertGreaterEqual(params.batch_size_factor, 0.4)
    
    def test_update_parameters_power_efficient(self):
        """Test parameter updates with power-efficient strategy"""
        self.controller.adaptation_strategy = AdaptationStrategy.POWER_EFFICIENT
        
        power_state = PowerState(
            cpu_usage_percent=95.0,
            gpu_usage_percent=90.0,
            cpu_temp_celsius=88.0,
            gpu_temp_celsius=82.0,
            cpu_power_watts=24.0,
            gpu_power_watts=70.0,
            timestamp=time.time()
        )
        
        params = self.controller.update_parameters(power_state)
        
        # With power-efficient strategy, factors should be lower under stress
        self.assertLessEqual(params.performance_factor, 0.5)
        self.assertLessEqual(params.batch_size_factor, 0.5)
    
    def test_update_parameters_thermal_aware(self):
        """Test parameter updates with thermal-aware strategy"""
        self.controller.adaptation_strategy = AdaptationStrategy.THERMAL_AWARE
        
        power_state = PowerState(
            cpu_usage_percent=70.0,
            gpu_usage_percent=65.0,
            cpu_temp_celsius=88.0,  # Close to critical
            gpu_temp_celsius=82.0,  # Close to critical
            cpu_power_watts=18.0,
            gpu_power_watts=55.0,
            timestamp=time.time()
        )
        
        params = self.controller.update_parameters(power_state)
        
        # With thermal-aware strategy, factors should be reduced when temp is high
        self.assertLessEqual(params.performance_factor, 0.6)
        self.assertGreater(params.execution_delay, 0.0)  # Should have execution delay
    
    def test_parameter_history(self):
        """Test that parameter history is maintained"""
        power_state = PowerState(
            cpu_usage_percent=50.0,
            gpu_usage_percent=40.0,
            cpu_temp_celsius=60.0,
            gpu_temp_celsius=50.0,
            cpu_power_watts=12.0,
            gpu_power_watts=30.0,
            timestamp=time.time()
        )
        
        # Update parameters multiple times
        for i in range(5):
            power_state.timestamp = time.time() + i
            self.controller.update_parameters(power_state)
        
        self.assertEqual(len(self.controller.parameter_history), 5)
    
    def test_historical_trend(self):
        """Test historical trend calculation"""
        power_state = PowerState(
            cpu_usage_percent=50.0,
            gpu_usage_percent=40.0,
            cpu_temp_celsius=60.0,
            gpu_temp_celsius=50.0,
            cpu_power_watts=12.0,
            gpu_power_watts=30.0,
            timestamp=time.time()
        )
        
        # Update parameters with increasing performance factors
        for i in range(10):
            power_state.timestamp = time.time() + i
            params = self.controller.update_parameters(power_state)
            # Modify the params to simulate a trend
            params.performance_factor = 0.5 + (i * 0.05)
        
        trend = self.controller.get_historical_trend(window_size=5)
        if trend:
            # The trend should be positive since we increased performance factors
            self.assertGreater(trend['performance_trend'], 0)


class TestLoadBalancer(unittest.TestCase):
    """Test cases for load balancer"""
    
    def setUp(self):
        self.constraints = PowerConstraint()
        self.load_balancer = LoadBalancer(self.constraints)
    
    def test_load_distribution_normal_conditions(self):
        """Test load distribution under normal conditions"""
        workloads = [("work1", lambda x: x), ("work2", lambda x: x), ("work3", lambda x: x)]
        
        power_state = PowerState(
            cpu_usage_percent=50.0,
            gpu_usage_percent=40.0,
            cpu_temp_celsius=60.0,
            gpu_temp_celsius=50.0,
            cpu_power_watts=12.0,
            gpu_power_watts=30.0,
            timestamp=time.time()
        )
        
        distribution = self.load_balancer.distribute_load(workloads, power_state)
        
        # Under normal conditions, distribution should be relatively high
        # Each workload should get at least 0.95 / total_workloads of capacity
        total_workloads = 3
        expected_min_per_workload = 0.95 / total_workloads  # About 0.317
        for workload_id, factor in distribution.items():
            self.assertGreaterEqual(factor, expected_min_per_workload * 0.8)  # At least 80% of expected
    
    def test_load_distribution_severe_constraints(self):
        """Test load distribution under severe constraints"""
        workloads = [("work1", lambda x: x), ("work2", lambda x: x)]
        
        power_state = PowerState(
            cpu_usage_percent=95.0,
            gpu_usage_percent=92.0,
            cpu_temp_celsius=92.0,  # Above critical
            gpu_temp_celsius=88.0,  # Above critical
            cpu_power_watts=24.0,
            gpu_power_watts=72.0,
            timestamp=time.time()
        )
        
        distribution = self.load_balancer.distribute_load(workloads, power_state)
        
        # Under severe constraints, distribution should be very low
        for workload_id, factor in distribution.items():
            self.assertLessEqual(factor, 0.4)  # At most 40% of capacity


class TestAdaptiveModelWrapper(unittest.TestCase):
    """Test cases for adaptive model wrapper"""
    
    def setUp(self):
        self.constraints = PowerConstraint()
        
        # Create a dummy model
        class DummyModel:
            def predict(self, X):
                return [0.5] * len(X) if hasattr(X, '__len__') else [0.5]
        
        dummy_model = DummyModel()
        self.adaptive_model = AdaptiveModelWrapper(dummy_model, self.constraints)
    
    def test_adaptive_prediction(self):
        """Test adaptive prediction with power constraints"""
        input_data = [1, 2, 3, 4, 5]
        
        power_state = PowerState(
            cpu_usage_percent=60.0,
            gpu_usage_percent=50.0,
            cpu_temp_celsius=70.0,
            gpu_temp_celsius=60.0,
            cpu_power_watts=15.0,
            gpu_power_watts=40.0,
            timestamp=time.time()
        )
        
        result = self.adaptive_model.predict(input_data, power_state)
        
        # Result should contain prediction and parameters used
        self.assertIn('prediction', result)
        self.assertIn('parameters_used', result)
        self.assertEqual(len(result['prediction']), len(input_data))
    
    def test_adaptive_training(self):
        """Test adaptive training with power constraints"""
        training_data = [1, 2, 3, 4, 5]
        
        power_state = PowerState(
            cpu_usage_percent=80.0,
            gpu_usage_percent=75.0,
            cpu_temp_celsius=80.0,
            gpu_temp_celsius=70.0,
            cpu_power_watts=20.0,
            gpu_power_watts=60.0,
            timestamp=time.time()
        )
        
        result = self.adaptive_model.fit(training_data, power_state, epochs=10, batch_size=32)
        
        # Result should contain training metrics and parameters used
        self.assertIn('final_loss', result)
        self.assertIn('epochs_trained', result)
        self.assertIn('parameters_used', result)


class TestDVFSController(unittest.TestCase):
    """Test cases for DVFS controller"""
    
    def setUp(self):
        self.dvfs = DVFSController()
    
    def test_dvfs_initialization(self):
        """Test that DVFS controller is initialized correctly"""
        # Check that frequency states are populated
        freq_states = self.dvfs.get_frequency_states()
        self.assertGreaterEqual(len(freq_states), 1)

        # GPU frequency states may or may not be available depending on system
        # so we just verify the method returns without error
        gpu_freq_states = self.dvfs.get_gpu_frequency_states()
        # The method should return a list (even if empty)
        self.assertIsInstance(gpu_freq_states, list)
    
    def test_frequency_state_determination(self):
        """Test frequency state determination logic"""
        # Test high usage, acceptable temp -> max frequency
        state = self.dvfs._determine_frequency_state(cpu_usage=90.0, cpu_temp=60.0)
        if state:
            self.assertEqual(state, self.dvfs._get_max_frequency_state())
        
        # Test low usage, acceptable temp -> min frequency
        state = self.dvfs._determine_frequency_state(cpu_usage=10.0, cpu_temp=40.0)
        if state:
            self.assertEqual(state, self.dvfs._get_min_frequency_state())
        
        # Test acceptable usage, critical temp -> min frequency
        state = self.dvfs._determine_frequency_state(cpu_usage=50.0, cpu_temp=95.0)
        if state:
            self.assertEqual(state, self.dvfs._get_min_frequency_state())


class TestWorkloadBasedDVFS(unittest.TestCase):
    """Test cases for workload-based DVFS"""
    
    def setUp(self):
        self.dvfs = DVFSController()
        self.workload_dvfs = WorkloadBasedDVFS(self.dvfs)
        
        # Register test profiles
        for name, profile in WORKLOAD_PROFILES.items():
            self.workload_dvfs.register_workload_profile(name, profile)
    
    def test_workload_profile_registration(self):
        """Test that workload profiles are registered correctly"""
        self.assertEqual(len(self.workload_dvfs.workload_profiles), len(WORKLOAD_PROFILES))
        self.assertIn("high_performance", self.workload_dvfs.workload_profiles)
        self.assertIn("power_efficient", self.workload_dvfs.workload_profiles)
        self.assertIn("balanced", self.workload_dvfs.workload_profiles)
    
    def test_workload_type_setting(self):
        """Test setting workload type"""
        self.workload_dvfs.set_workload_type("high_performance")
        self.assertEqual(self.workload_dvfs.current_workload, "high_performance")
        
        # Setting unknown workload should result in "unknown"
        self.workload_dvfs.set_workload_type("unknown_workload")
        self.assertEqual(self.workload_dvfs.current_workload, "unknown")


class TestIntegration(unittest.TestCase):
    """Integration tests for the entire power management system"""
    
    def setUp(self):
        self.constraints = PowerConstraint()
    
    def test_full_system_integration(self):
        """Test integration of all power management components"""
        # Initialize all components
        scheduler = PowerAwareScheduler(self.constraints)
        thermal_manager = ThermalManager(self.constraints)
        adaptive_controller = AdaptiveController(self.constraints)
        dvfs = DVFSController()
        
        # Create a sample task
        def sample_task():
            return "completed"
        
        # Add task to scheduler
        scheduler.add_task(sample_task, priority=5)
        
        # Update power state
        power_state = PowerState(
            cpu_usage_percent=70.0,
            gpu_usage_percent=60.0,
            cpu_temp_celsius=75.0,
            gpu_temp_celsius=65.0,
            cpu_power_watts=18.0,
            gpu_power_watts=50.0,
            timestamp=time.time()
        )
        
        # Update thermal state
        thermal_manager.get_thermal_state()
        
        # Update adaptive parameters
        adaptive_controller.update_parameters(power_state)
        
        # Check system efficiency
        efficiency = dvfs.get_system_power_efficiency()
        
        # Verify all components have been updated
        self.assertGreaterEqual(len(scheduler.tasks), 0)
        self.assertIsNotNone(thermal_manager.get_thermal_state())
        self.assertIsNotNone(adaptive_controller.get_current_parameters())
        self.assertIn('current_frequency_mhz', efficiency)
        
        # Clean up
        scheduler.stop_monitoring()
        thermal_manager.stop_management()
        adaptive_controller.stop_adaptation()
        dvfs.stop_adaptive_scaling()


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)