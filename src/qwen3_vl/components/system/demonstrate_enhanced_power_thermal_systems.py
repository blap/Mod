"""
Demonstration of Enhanced Power and Thermal Management System

This script demonstrates the enhanced power and thermal management system
with comprehensive error handling and validation.
"""

import time
import logging
from typing import Dict, Any

# Import the enhanced systems
from enhanced_power_management import EnhancedPowerAwareScheduler, PowerConstraint, PowerMode
from enhanced_thermal_management import EnhancedThermalManager, ThermalPolicy
from system_validation_utils import SystemValidator, ErrorHandlingManager

# Create global instances for use in the demo
system_validator = SystemValidator()
system_error_handler = ErrorHandlingManager()


def setup_logging():
    """Setup logging for the demonstration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demonstrate_validation_utilities():
    """Demonstrate the validation utilities"""
    print("\n=== VALIDATION UTILITIES DEMONSTRATION ===")
    
    # Test power threshold validation
    print("\n1. Power Threshold Validation:")
    result = system_validator.validate_power_thresholds(
        cpu_power=15.0, gpu_power=20.0,
        max_cpu_power=25.0, max_gpu_power=75.0
    )
    print(f"   Valid: {result.is_valid}, Message: {result.message}")
    
    # Test thermal threshold validation
    print("\n2. Thermal Threshold Validation:")
    result = system_validator.validate_thermal_thresholds(
        cpu_temp=70.0, gpu_temp=65.0,
        max_cpu_temp=90.0, max_gpu_temp=85.0
    )
    print(f"   Valid: {result.is_valid}, Message: {result.message}")
    
    # Test parameter range validation
    print("\n3. Parameter Range Validation:")
    result = system_validator.validate_parameter_range(
        value=5.0, min_val=1.0, max_val=10.0, param_name="example_param"
    )
    print(f"   Valid: {result.is_valid}, Message: {result.message}")
    
    # Test system resources validation
    print("\n4. System Resources Validation:")
    result = system_validator.validate_system_resources()
    print(f"   Valid: {result.is_valid}, Message: {result.message}")


def demonstrate_power_management():
    """Demonstrate the enhanced power management system"""
    print("\n=== ENHANCED POWER MANAGEMENT DEMONSTRATION ===")
    
    # Create constraints
    constraints = PowerConstraint(
        max_cpu_power_watts=25.0,
        max_gpu_power_watts=75.0,
        max_cpu_temp_celsius=90.0,
        max_gpu_temp_celsius=85.0
    )
    
    # Create the enhanced scheduler
    scheduler = EnhancedPowerAwareScheduler(constraints)
    
    # Add some example tasks
    def example_task():
        print("   Executing example task...")
        time.sleep(0.5)
        return "Task completed"
    
    def cpu_intensive_task():
        print("   Executing CPU intensive task...")
        # Simulate CPU work
        sum(i * i for i in range(100000))
        return "CPU task completed"
    
    print("\n1. Adding tasks to scheduler:")
    scheduler.add_task(example_task, priority=5, power_requirements=0.3)
    scheduler.add_task(cpu_intensive_task, priority=8, power_requirements=0.7)
    scheduler.add_task(example_task, priority=3, power_requirements=0.2)
    
    # Get initial power state
    print("\n2. Getting initial power state:")
    initial_state = scheduler.get_system_power_state()
    print(f"   CPU Usage: {initial_state.cpu_usage_percent:.1f}%")
    print(f"   CPU Temp: {initial_state.cpu_temp_celsius:.1f}°C")
    print(f"   CPU Power: {initial_state.cpu_power_watts:.2f}W")
    print(f"   GPU Usage: {initial_state.gpu_usage_percent:.1f}%")
    print(f"   GPU Temp: {initial_state.gpu_temp_celsius:.1f}°C")
    print(f"   GPU Power: {initial_state.gpu_power_watts:.2f}W")
    
    # Execute tasks
    print("\n3. Executing tasks:")
    scheduler.execute_tasks()
    
    # Check task queue status
    print("\n4. Task queue status:")
    status = scheduler.get_task_queue_status()
    print(f"   Pending tasks: {status['pending_tasks']}")
    print(f"   Running tasks: {status['running_tasks']}")
    print(f"   Power mode: {status['power_mode']}")
    print(f"   Current state: CPU={status['current_state']['cpu_usage']:.1f}%, GPU={status['current_state']['gpu_usage']:.1f}%")
    
    # Demonstrate power mode changes
    print("\n5. Power mode demonstration:")
    print(f"   Current power mode: {scheduler.get_power_mode().value}")
    scheduler.set_power_mode(PowerMode.PERFORMANCE)
    print(f"   Set power mode to: {scheduler.get_power_mode().value}")
    
    # Get power model info
    print("\n6. Power model information:")
    model_info = scheduler.get_power_model_info()
    print(f"   CPU model available: {model_info['cpu_power_model_available']}")
    print(f"   GPU model available: {model_info['gpu_power_model_available']}")
    print(f"   Current total power: {model_info['total_power_w']:.2f}W")
    
    # Cleanup
    scheduler.stop_monitoring()


def demonstrate_thermal_management():
    """Demonstrate the enhanced thermal management system"""
    print("\n=== ENHANCED THERMAL MANAGEMENT DEMONSTRATION ===")
    
    # Create constraints
    constraints = PowerConstraint(
        max_cpu_power_watts=25.0,
        max_gpu_power_watts=75.0,
        max_cpu_temp_celsius=90.0,
        max_gpu_temp_celsius=85.0
    )
    
    # Create the enhanced thermal manager
    thermal_manager = EnhancedThermalManager(constraints)
    
    # Register a callback
    def thermal_callback(event_type: str, value: float):
        print(f"   Thermal event: {event_type} at {value}°C")
    
    print("\n1. Registering thermal callback:")
    thermal_manager.register_callback(thermal_callback)
    
    # Get initial thermal state
    print("\n2. Getting initial thermal state:")
    zones = thermal_manager.get_thermal_state()
    for zone in zones:
        print(f"   {zone.name}: {zone.current_temp:.1f}°C (critical: {zone.critical_temp}°C)")
    
    # Get cooling state
    print("\n3. Getting cooling state:")
    cooling_devices = thermal_manager.get_cooling_state()
    for device in cooling_devices:
        print(f"   {device.name}: {device.current_state}% (range: {device.min_state}-{device.max_state}%)")
    
    # Adjust cooling for a zone
    print("\n4. Adjusting cooling for CPU zone:")
    thermal_manager.adjust_cooling("CPU", 70.0)  # Normal temperature
    
    # Get thermal summary
    print("\n5. Thermal summary:")
    summary = thermal_manager.get_thermal_summary()
    print(f"   Active: {summary['active']}")
    print(f"   Policy: {summary['policy']}")
    print(f"   Zones: {len(summary['zones'])}")
    print(f"   Cooling devices: {len(summary['cooling_devices'])}")
    
    # Change thermal policy
    print("\n6. Changing thermal policy:")
    print(f"   Current policy: {thermal_manager.policy.value}")
    thermal_manager.set_policy(ThermalPolicy.ACTIVE)
    print(f"   New policy: {thermal_manager.policy.value}")
    
    # Cleanup
    thermal_manager.stop_management()


def demonstrate_error_handling():
    """Demonstrate the error handling capabilities"""
    print("\n=== ERROR HANDLING DEMONSTRATION ===")
    
    # Demonstrate parameter validation errors
    print("\n1. Parameter validation errors:")
    try:
        from enhanced_power_management import PowerAwareScheduler
        constraints = PowerConstraint(max_cpu_power_watts=-5.0)  # Invalid value
        scheduler = EnhancedPowerAwareScheduler(constraints)
    except Exception as e:
        print(f"   Caught expected validation error: {type(e).__name__}: {str(e)}")
    
    # Demonstrate thermal validation errors
    print("\n2. Thermal validation errors:")
    try:
        from enhanced_thermal_management import ThermalManager
        constraints = PowerConstraint(max_cpu_temp_celsius=-10.0)  # Invalid value
        thermal_manager = EnhancedThermalManager(constraints)
    except Exception as e:
        print(f"   Caught expected validation error: {type(e).__name__}: {str(e)}")
    
    # Demonstrate error handling for invalid operations
    print("\n3. Operation error handling:")
    try:
        constraints = PowerConstraint()
        scheduler = EnhancedPowerAwareScheduler(constraints)
        # Try to add a task with invalid parameters
        scheduler.add_task(lambda: None, priority=15, power_requirements=0.5)  # Invalid priority
    except Exception as e:
        print(f"   Caught expected validation error: {type(e).__name__}: {str(e)}")


def demonstrate_thermal_aware_task():
    """Demonstrate thermal-aware task execution"""
    print("\n=== THERMAL-AWARE TASK DEMONSTRATION ===")
    
    from enhanced_thermal_management import ThermalAwareTask, EnhancedThermalManager
    
    # Create thermal manager
    constraints = PowerConstraint()
    thermal_manager = EnhancedThermalManager(constraints)
    
    # Create a thermal-aware task
    thermal_task = ThermalAwareTask("example_task", base_power=0.5)
    
    # Attach thermal manager
    thermal_task.attach_thermal_manager(thermal_manager)
    
    # Create a power state for testing
    from enhanced_power_management import PowerState
    power_state = PowerState(
        cpu_usage_percent=40.0,
        gpu_usage_percent=30.0,
        cpu_temp_celsius=60.0,
        gpu_temp_celsius=50.0,
        cpu_power_watts=10.0,
        gpu_power_watts=15.0
    )
    
    print("\n1. Executing thermal-aware task under normal conditions:")
    result = thermal_task.execute_with_thermal_awareness(power_state)
    print(f"   Task execution result: {result}")
    
    # Stop thermal manager
    thermal_manager.stop_management()


def main():
    """Main demonstration function"""
    print("Enhanced Power and Thermal Management System Demonstration")
    print("=" * 60)
    
    setup_logging()
    
    try:
        demonstrate_validation_utilities()
        demonstrate_power_management()
        demonstrate_thermal_management()
        demonstrate_error_handling()
        demonstrate_thermal_aware_task()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("The enhanced power and thermal management system includes:")
        print("  - Comprehensive parameter validation")
        print("  - Robust error handling with meaningful messages")
        print("  - Detailed logging for error conditions and system events")
        print("  - Resource cleanup mechanisms for error conditions")
        print("  - Validation for power and thermal thresholds")
        print("  - Validation for frequency and power state transitions")
        print("  - Backward compatibility with existing code")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()