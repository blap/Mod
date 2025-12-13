"""
Demonstration of the Unified Resource Manager for Qwen3-VL Model

This script demonstrates the unified resource management system that coordinates
between memory, CPU, and thermal resources for optimal performance.
"""

import time
import threading
import sys
import os
# Add dev_tools to the Python path to import the moved modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'dev_tools', 'utils'))

from unified_resource_manager import (
    UnifiedResourceManager, ResourceType, resource_context,
    init_global_resource_manager
)
from advanced_memory_pooling_system import AdvancedMemoryPoolingSystem, TensorType
from advanced_memory_management_vl import VisionLanguageMemoryOptimizer
from enhanced_thermal_management import EnhancedThermalManager
from enhanced_power_management import PowerConstraint


def demonstrate_basic_usage():
    """Demonstrate basic usage of the unified resource manager"""
    print("=== Unified Resource Manager Demo ===")
    print()
    
    # Create the necessary components
    memory_pooling_system = AdvancedMemoryPoolingSystem()
    memory_optimizer = VisionLanguageMemoryOptimizer()
    
    # Create power constraints for thermal management
    power_constraints = PowerConstraint(
        max_cpu_temp_celsius=85.0,
        max_gpu_temp_celsius=80.0,
        max_cpu_power_watts=45.0,
        max_gpu_power_watts=75.0
    )
    thermal_manager = EnhancedThermalManager(power_constraints)
    
    # Initialize unified resource manager
    resource_manager = init_global_resource_manager(
        memory_pooling_system,
        memory_optimizer,
        thermal_manager
    )
    
    print("1. System Health Check")
    health = resource_manager.get_system_health()
    print(f"   Overall Health: {health['overall_health']}")
    print(f"   Memory Pressure: {health['memory_pressure']}")
    print(f"   CPU Pressure: {health['cpu_pressure']}")
    print(f"   Thermal Pressure: {health['thermal_pressure']}")
    print()
    
    print("2. Resource Allocation with Context Manager")
    # Allocate memory using context manager (automatic cleanup)
    with resource_context(resource_manager, ResourceType.MEMORY, 1024*1024, "demo_tensor_1", 
                         tensor_type=TensorType.KV_CACHE) as tensor:
        print(f"   Allocated tensor resource: {type(tensor)}")
        print(f"   Tensor shape: {tensor.shape if hasattr(tensor, 'shape') else 'N/A'}")
    
    print("   Resource automatically deallocated via context manager")
    print()
    
    print("3. Manual Resource Management")
    # Manual allocation and deallocation
    resource = resource_manager.allocate_resource(
        ResourceType.MEMORY, 2*1024*1024, "demo_tensor_2", 
        tensor_type=TensorType.IMAGE_FEATURES
    )
    if resource:
        print(f"   Manually allocated resource: {type(resource)}")
        success = resource_manager.deallocate_resource(ResourceType.MEMORY, "demo_tensor_2")
        print(f"   Manual deallocation successful: {success}")
    print()
    
    print("4. Resource Usage Summary")
    summary = resource_manager.get_resource_usage_summary()
    for key, value in summary.items():
        if key != 'metrics':  # Print metrics separately
            print(f"   {key}: {value}")
    
    print("   Metrics:")
    for metric_key, metric_value in summary['metrics'].items():
        print(f"     {metric_key}: {metric_value}")
    print()


def demonstrate_error_handling():
    """Demonstrate error handling and recovery mechanisms"""
    print("=== Error Handling Demo ===")
    print()
    
    # Create resource manager with mocked components to simulate errors
    from unittest.mock import Mock
    
    memory_pooling_system = Mock()
    memory_pooling_system.allocate.return_value = None  # Simulate allocation failure
    memory_pooling_system.deallocate.return_value = True
    
    thermal_manager = Mock()
    thermal_manager.get_thermal_summary.return_value = {'zones': [], 'active': False}
    
    resource_manager = UnifiedResourceManager(
        memory_pooling_system=memory_pooling_system,
        thermal_manager=thermal_manager
    )
    
    print("1. Handling Allocation Failure")
    result = resource_manager.allocate_resource(
        ResourceType.MEMORY, 1024, "failing_tensor"
    )
    print(f"   Allocation result: {result}")
    print(f"   Error count after failed allocation: {resource_manager.metrics['error_count']}")
    print()
    
    print("2. Context Manager with Exception Handling")
    try:
        with resource_context(resource_manager, ResourceType.MEMORY, 1024, "error_tensor") as tensor:
            print("   This should not print due to allocation failure")
    except RuntimeError as e:
        print(f"   Context manager properly handled error: {e}")
    print()


def demonstrate_retry_mechanism():
    """Demonstrate retry logic for critical operations"""
    print("=== Retry Logic Demo ===")
    print()
    
    # Import from the moved location
    from unified_resource_manager import RetryManager
    
    retry_manager = RetryManager(max_retries=3, base_delay=0.1)
    
    # Simulate a function that fails initially but succeeds later
    call_count = 0
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            print(f"   Attempt {call_count}: Failed")
            raise Exception("Temporary failure")
        else:
            print(f"   Attempt {call_count}: Succeeded")
            return "Success!"
    
    try:
        result = retry_manager.execute_with_retry(flaky_function)
        print(f"   Final result: {result}")
        print(f"   Total attempts: {call_count}")
    except Exception as e:
        print(f"   Retry exhausted: {e}")
    print()


def demonstrate_resource_coordination():
    """Demonstrate coordination between different resource types"""
    print("=== Resource Coordination Demo ===")
    print()
    
    # Create a simple workload that uses multiple resource types
    def simulate_vision_language_workload(resource_manager, workload_id):
        print(f"   Starting workload {workload_id}")
        
        # Allocate memory for image processing
        image_resource = resource_manager.allocate_resource(
            ResourceType.MEMORY, 5*1024*1024, f"img_{workload_id}", 
            tensor_type=TensorType.IMAGE_FEATURES
        )
        
        if image_resource:
            print(f"   Allocated image processing memory for {workload_id}")
            
            # Simulate CPU-intensive processing
            result = resource_manager.execute_with_resource_optimization(
                lambda: sum(i*i for i in range(1000))
            )
            print(f"   Processed with optimization result: {result}")
            
            # Clean up
            resource_manager.deallocate_resource(ResourceType.MEMORY, f"img_{workload_id}")
            print(f"   Cleaned up resources for {workload_id}")
    
    # Create resource manager
    memory_pooling_system = AdvancedMemoryPoolingSystem()
    memory_optimizer = VisionLanguageMemoryOptimizer()
    
    power_constraints = PowerConstraint()
    thermal_manager = EnhancedThermalManager(power_constraints)
    
    resource_manager = UnifiedResourceManager(
        memory_pooling_system, memory_optimizer, thermal_manager
    )
    
    # Run multiple workloads concurrently to demonstrate coordination
    threads = []
    for i in range(3):
        thread = threading.Thread(
            target=simulate_vision_language_workload, 
            args=(resource_manager, f"workload_{i}")
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print(f"   Final system health: {resource_manager.get_system_health()['overall_health']}")
    print()


def main():
    """Main function to run all demonstrations"""
    print("Unified Resource Manager for Qwen3-VL - Comprehensive Demo")
    print("=" * 60)
    print()
    
    demonstrate_basic_usage()
    demonstrate_error_handling()
    demonstrate_retry_mechanism()
    demonstrate_resource_coordination()
    
    print("=" * 60)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()