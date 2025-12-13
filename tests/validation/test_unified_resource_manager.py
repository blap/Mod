"""
Comprehensive tests for the Unified Resource Manager

This module contains comprehensive tests for the unified resource management system,
including tests for memory, CPU, and thermal resource management, context managers,
error handling, and retry logic.
"""

import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import numpy as np

import sys
import os
# Add dev_tools to the Python path to import the moved modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dev_tools', 'utils'))

from unified_resource_manager import (
    UnifiedResourceManager, ResourceType, ResourceState, ResourceInfo,
    MemoryResourceManager, CPUResourceManager, ThermalResourceManager,
    ResourceContextManager, resource_context, get_global_resource_manager,
    init_global_resource_manager, GlobalResourceManager
)
from advanced_memory_pooling_system import AdvancedMemoryPoolingSystem, TensorType
from advanced_memory_management_vl import VisionLanguageMemoryOptimizer
from enhanced_thermal_management import EnhancedThermalManager
from enhanced_power_management import PowerConstraint


class TestMemoryResourceManager(unittest.TestCase):
    """Tests for MemoryResourceManager"""
    
    def setUp(self):
        self.memory_pooling_system = Mock()
        self.memory_optimizer = Mock()
        self.memory_manager = MemoryResourceManager(
            self.memory_pooling_system, self.memory_optimizer
        )
    
    def test_allocate_tensor_with_pooling_system(self):
        """Test tensor allocation with memory pooling system"""
        # Mock the pooling system allocation
        mock_block = Mock()
        self.memory_pooling_system.allocate.return_value = mock_block
        
        result = self.memory_manager.allocate_tensor(TensorType.KV_CACHE, 1024, "test_tensor")
        
        self.assertEqual(result, mock_block)
        self.memory_pooling_system.allocate.assert_called_once_with(
            TensorType.KV_CACHE, 1024, "test_tensor"
        )
        # Check that resource was registered
        resource = self.memory_manager.get_resource("test_tensor")
        self.assertIsNotNone(resource)
        self.assertEqual(resource.resource_type, ResourceType.MEMORY)
    
    def test_allocate_tensor_with_optimizer(self):
        """Test tensor allocation with memory optimizer"""
        # Set pooling system to None to use optimizer
        self.memory_manager.memory_pooling_system = None
        mock_tensor = np.zeros(256, dtype=np.float32)
        self.memory_optimizer.allocate_tensor_memory.return_value = mock_tensor

        result = self.memory_manager.allocate_tensor(TensorType.IMAGE_FEATURES, 1024, "test_tensor")

        # For numpy arrays, we should check if they're the same object or have same content
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, mock_tensor)
    
    def test_deallocate_tensor(self):
        """Test tensor deallocation"""
        # First allocate a tensor to register it
        mock_block = Mock()
        self.memory_pooling_system.allocate.return_value = mock_block
        self.memory_manager.allocate_tensor(TensorType.KV_CACHE, 1024, "test_tensor")
        
        # Mock the deallocation
        self.memory_pooling_system.deallocate.return_value = True
        
        result = self.memory_manager.deallocate_tensor(TensorType.KV_CACHE, "test_tensor")
        
        self.assertTrue(result)
        self.memory_pooling_system.deallocate.assert_called_once_with(
            TensorType.KV_CACHE, "test_tensor"
        )
        # Check that resource was unregistered
        resource = self.memory_manager.get_resource("test_tensor")
        self.assertIsNone(resource)
    
    def test_check_memory_pressure(self):
        """Test memory pressure detection"""
        with patch('psutil.virtual_memory') as mock_vm:
            # Mock high memory usage
            mock_vm.return_value.percent = 90.0
            self.assertTrue(self.memory_manager.check_memory_pressure())
            
            # Mock low memory usage
            mock_vm.return_value.percent = 30.0
            self.assertFalse(self.memory_manager.check_memory_pressure())


class TestCPUResourceManager(unittest.TestCase):
    """Tests for CPUResourceManager"""
    
    def setUp(self):
        self.cpu_manager = CPUResourceManager()
    
    def test_get_cpu_usage(self):
        """Test CPU usage retrieval"""
        with patch('psutil.cpu_percent') as mock_cpu_percent:
            mock_cpu_percent.return_value = 75.0
            usage = self.cpu_manager.get_cpu_usage()
            self.assertEqual(usage, 75.0)
    
    def test_check_cpu_pressure(self):
        """Test CPU pressure detection"""
        with patch('psutil.cpu_percent') as mock_cpu_percent:
            # Mock high CPU usage
            mock_cpu_percent.return_value = 85.0
            self.assertTrue(self.cpu_manager.check_cpu_pressure())
            
            # Mock low CPU usage
            mock_cpu_percent.return_value = 50.0
            self.assertFalse(self.cpu_manager.check_cpu_pressure())
    
    def test_optimize_thread_pool(self):
        """Test thread pool optimization"""
        with patch('psutil.cpu_percent') as mock_cpu_percent:
            # High CPU pressure
            mock_cpu_percent.return_value = 90.0
            optimized_workers = self.cpu_manager.optimize_thread_pool(8)
            self.assertLess(optimized_workers, 8)
            
            # Low CPU pressure
            mock_cpu_percent.return_value = 30.0
            optimized_workers = self.cpu_manager.optimize_thread_pool(8)
            self.assertEqual(optimized_workers, 8)


class TestThermalResourceManager(unittest.TestCase):
    """Tests for ThermalResourceManager"""
    
    def setUp(self):
        self.thermal_manager_mock = Mock()
        self.thermal_resource_manager = ThermalResourceManager(self.thermal_manager_mock)
    
    def test_get_thermal_state(self):
        """Test getting thermal state"""
        mock_summary = {
            'zones': [{'name': 'CPU', 'current_temp': 70.0, 'critical_temp': 80.0, 'status': 'normal'}],
            'active': True
        }
        self.thermal_manager_mock.get_thermal_summary.return_value = mock_summary
        
        state = self.thermal_resource_manager.get_thermal_state()
        self.assertEqual(state, mock_summary)
    
    def test_check_thermal_pressure(self):
        """Test thermal pressure detection"""
        # Test with high temperature
        with patch.object(self.thermal_resource_manager, 'get_thermal_state') as mock_get_state:
            mock_get_state.return_value = {
                'zones': [{'current_temp': 80.0, 'critical_temp': 75.0}]
            }
            self.assertTrue(self.thermal_resource_manager.check_thermal_pressure())
            
            # Test with normal temperature
            mock_get_state.return_value = {
                'zones': [{'current_temp': 60.0, 'critical_temp': 75.0}]
            }
            self.assertFalse(self.thermal_resource_manager.check_thermal_pressure())


class TestUnifiedResourceManager(unittest.TestCase):
    """Tests for UnifiedResourceManager"""
    
    def setUp(self):
        self.memory_pooling_system = Mock()
        self.memory_optimizer = Mock()
        self.thermal_manager = Mock()
        self.resource_manager = UnifiedResourceManager(
            self.memory_pooling_system,
            self.memory_optimizer,
            self.thermal_manager
        )
    
    def test_get_system_health(self):
        """Test system health monitoring"""
        # Mock the individual managers
        with patch.object(self.resource_manager.memory_manager, 'get_memory_stats') as mock_mem_stats, \
             patch.object(self.resource_manager.cpu_manager, 'get_cpu_stats') as mock_cpu_stats, \
             patch.object(self.resource_manager.thermal_manager, 'get_thermal_state') as mock_thermal_state, \
             patch.object(self.resource_manager.memory_manager, 'check_memory_pressure') as mock_mem_pressure, \
             patch.object(self.resource_manager.cpu_manager, 'check_cpu_pressure') as mock_cpu_pressure, \
             patch.object(self.resource_manager.thermal_manager, 'check_thermal_pressure') as mock_thermal_pressure:
            
            mock_mem_stats.return_value = {'system': {'percent_used': 50.0}}
            mock_cpu_stats.return_value = {'cpu_percent': 40.0}
            mock_thermal_state.return_value = {'zones': []}
            mock_mem_pressure.return_value = False
            mock_cpu_pressure.return_value = False
            mock_thermal_pressure.return_value = False
            
            health = self.resource_manager.get_system_health()
            
            self.assertEqual(health['overall_health'], 'healthy')
            self.assertFalse(health['memory_pressure'])
            self.assertFalse(health['cpu_pressure'])
            self.assertFalse(health['thermal_pressure'])
    
    def test_allocate_resource_memory(self):
        """Test memory resource allocation"""
        mock_block = Mock()
        self.memory_pooling_system.allocate.return_value = mock_block
        
        result = self.resource_manager.allocate_resource(
            ResourceType.MEMORY, 1024, "test_tensor", tensor_type=TensorType.KV_CACHE
        )
        
        self.assertEqual(result, mock_block)
        self.memory_pooling_system.allocate.assert_called_once_with(
            TensorType.KV_CACHE, 1024, "test_tensor"
        )
    
    def test_allocate_resource_cpu(self):
        """Test CPU resource allocation"""
        # Mock system health as healthy
        with patch.object(self.resource_manager, 'get_system_health') as mock_health:
            mock_health.return_value = {'overall_health': 'healthy'}
            
            result = self.resource_manager.allocate_resource(
                ResourceType.CPU, 0, "test_cpu_task"
            )
            
            self.assertTrue(result)
            self.assertEqual(self.resource_manager.cpu_manager.active_threads, 1)
    
    def test_deallocate_resource_memory(self):
        """Test memory resource deallocation"""
        # First allocate a resource
        mock_block = Mock()
        self.memory_pooling_system.allocate.return_value = mock_block
        self.resource_manager.allocate_resource(
            ResourceType.MEMORY, 1024, "test_tensor", tensor_type=TensorType.KV_CACHE
        )
        
        # Mock deallocation
        self.memory_pooling_system.deallocate.return_value = True
        
        result = self.resource_manager.deallocate_resource(ResourceType.MEMORY, "test_tensor")
        
        self.assertTrue(result)
        self.memory_pooling_system.deallocate.assert_called_once_with(
            TensorType.KV_CACHE, "test_tensor"
        )
    
    def test_execute_with_resource_optimization(self):
        """Test execution with resource optimization"""
        def test_func():
            return "test_result"
        
        result = self.resource_manager.execute_with_resource_optimization(test_func)
        self.assertEqual(result, "test_result")


class TestResourceContextManager(unittest.TestCase):
    """Tests for ResourceContextManager"""
    
    def setUp(self):
        self.mock_resource_manager = Mock()
        self.context_manager = ResourceContextManager(
            self.mock_resource_manager, ResourceType.MEMORY, 1024, "test_resource"
        )
    
    def test_context_manager_success(self):
        """Test successful context manager usage"""
        mock_resource = Mock()
        self.mock_resource_manager.allocate_resource.return_value = mock_resource
        self.mock_resource_manager.deallocate_resource.return_value = True
        
        with self.context_manager as resource:
            self.assertEqual(resource, mock_resource)
        
        self.mock_resource_manager.allocate_resource.assert_called_once()
        self.mock_resource_manager.deallocate_resource.assert_called_once_with(
            ResourceType.MEMORY, "test_resource", tensor_type=TensorType.KV_CACHE
        )
    
    def test_context_manager_failure_on_allocation(self):
        """Test context manager when allocation fails"""
        self.mock_resource_manager.allocate_resource.return_value = None
        
        with self.assertRaises(RuntimeError):
            with self.context_manager:
                pass  # This should not be reached
    
    def test_context_manager_failure_on_deallocation(self):
        """Test context manager when deallocation fails"""
        mock_resource = Mock()
        self.mock_resource_manager.allocate_resource.return_value = mock_resource
        self.mock_resource_manager.deallocate_resource.return_value = False
        
        with self.context_manager as resource:
            self.assertEqual(resource, mock_resource)
        
        # Should still call deallocation even if it fails
        self.mock_resource_manager.deallocate_resource.assert_called_once()


class TestGlobalResourceManager(unittest.TestCase):
    """Tests for GlobalResourceManager singleton"""
    
    def setUp(self):
        # Reset singleton instance
        GlobalResourceManager._instance = None
    
    def test_singleton_pattern(self):
        """Test that GlobalResourceManager follows singleton pattern"""
        instance1 = GlobalResourceManager()
        instance2 = GlobalResourceManager()
        
        self.assertIs(instance1, instance2)
    
    def test_get_set_resource_manager(self):
        """Test getting and setting resource manager"""
        global_rm = GlobalResourceManager()
        mock_rm = Mock()
        
        global_rm.set_resource_manager(mock_rm)
        retrieved_rm = global_rm.get_resource_manager()
        
        self.assertEqual(retrieved_rm, mock_rm)
    
    def test_get_global_resource_manager(self):
        """Test the get_global_resource_manager function"""
        mock_rm = Mock()
        GlobalResourceManager().set_resource_manager(mock_rm)
        
        retrieved_rm = get_global_resource_manager()
        self.assertEqual(retrieved_rm, mock_rm)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_complete_integration(self):
        """Test complete integration with real components"""
        # Create real components (not mocks)
        memory_pooling_system = AdvancedMemoryPoolingSystem()
        memory_optimizer = VisionLanguageMemoryOptimizer()
        power_constraints = PowerConstraint()
        thermal_manager = EnhancedThermalManager(power_constraints)
        
        # Create unified resource manager
        resource_manager = UnifiedResourceManager(
            memory_pooling_system, memory_optimizer, thermal_manager
        )
        
        # Test allocation and deallocation
        with resource_context(resource_manager, ResourceType.MEMORY, 1024, "integration_test", 
                             tensor_type=TensorType.KV_CACHE) as resource:
            self.assertIsNotNone(resource)
        
        # Check that resource was properly cleaned up
        summary = resource_manager.get_resource_usage_summary()
        self.assertGreaterEqual(summary['metrics']['allocation_count'], 1)
        self.assertGreaterEqual(summary['metrics']['deallocation_count'], 1)
    
    def test_error_handling_in_context_manager(self):
        """Test error handling in context manager"""
        resource_manager = UnifiedResourceManager()
        
        # Mock to simulate an error during allocation
        with patch.object(resource_manager, 'allocate_resource') as mock_alloc:
            mock_alloc.side_effect = Exception("Allocation failed")
            
            with self.assertRaises(Exception):
                with resource_context(resource_manager, ResourceType.MEMORY, 1024, "error_test"):
                    pass
    
    def test_retry_logic(self):
        """Test retry logic in resource operations"""
        # Create a resource manager with mocked allocation that fails initially
        resource_manager = UnifiedResourceManager()

        # Set max retries to 2 for testing
        resource_manager.memory_manager.retry_manager.max_retries = 2

        # Create a mock function that tracks calls and fails initially
        call_count = 0
        def failing_function(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary failure")
            return Mock()

        # Test the retry manager directly
        result = resource_manager.memory_manager.retry_manager.execute_with_retry(failing_function)

        self.assertIsNotNone(result)
        self.assertEqual(call_count, 3)  # Called 3 times (2 failures + 1 success)


class TestPerformanceMetrics(unittest.TestCase):
    """Tests for performance metrics collection"""
    
    def test_metrics_collection(self):
        """Test that metrics are properly collected"""
        # Create a resource manager with mocked components to avoid real hardware access
        memory_pooling_system = Mock()
        memory_optimizer = Mock()
        thermal_manager = Mock()

        resource_manager = UnifiedResourceManager(
            memory_pooling_system, memory_optimizer, thermal_manager
        )

        # Set up the mocks to return appropriate values
        memory_pooling_system.allocate.return_value = Mock()  # Mock memory block
        memory_pooling_system.deallocate.return_value = True
        thermal_manager.get_thermal_summary.return_value = {'zones': [], 'active': False}

        # Perform some operations
        initial_metrics = resource_manager.metrics.copy()

        # Perform real allocation/deallocation with mocked dependencies
        resource_manager.allocate_resource(ResourceType.MEMORY, 1024, "metric_test",
                                         tensor_type=TensorType.KV_CACHE)
        resource_manager.deallocate_resource(ResourceType.MEMORY, "metric_test")

        # Check that metrics were updated
        final_metrics = resource_manager.metrics
        self.assertGreater(final_metrics['allocation_count'], initial_metrics['allocation_count'])
        self.assertGreater(final_metrics['deallocation_count'], initial_metrics['deallocation_count'])


if __name__ == '__main__':
    print("Running Unified Resource Manager tests...")
    unittest.main(verbosity=2)