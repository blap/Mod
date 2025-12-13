#!/usr/bin/env python3
"""
Demonstration of the Power Estimation Models for Intel i5-10210U and NVIDIA SM61
This script demonstrates the accurate power estimation models and their integration
with the existing power management system.
"""

from power_estimation_models import IntelI5_10210UPowerModel, NVidiaSM61PowerModel, PowerProfiler, estimate_cpu_power, estimate_gpu_power
from power_management import PowerAwareScheduler, PowerConstraint
from integrated_power_management import PowerManagementFramework


def demonstrate_cpu_power_model():
    """Demonstrate the Intel i5-10210U power estimation model"""
    print("=== Intel i5-10210U Power Estimation Model ===")
    
    cpu_model = IntelI5_10210UPowerModel()
    
    # Test different utilization and frequency scenarios
    test_cases = [
        (0.0, 0.1, "Idle"),
        (0.3, 1.0, "Low Load"),
        (0.6, 1.5, "Medium Load"),
        (0.9, 2.0, "High Load"),
        (1.0, 2.625, "Max Load (Boost)"),
    ]
    
    for utilization, freq_ratio, label in test_cases:
        power = cpu_model.estimate_power(utilization, freq_ratio)
        print(f"{label:12} - Util: {utilization*100:3.0f}%, Freq Ratio: {freq_ratio:.2f}x -> Power: {power:.2f}W")
    
    print()


def demonstrate_gpu_power_model():
    """Demonstrate the NVIDIA SM61 power estimation model"""
    print("=== NVIDIA SM61 Power Estimation Model ===")
    
    gpu_model = NVidiaSM61PowerModel()
    
    # Test different utilization scenarios
    test_cases = [
        (0.0, 0.0, "Idle"),
        (0.2, 0.1, "Low Load"),
        (0.5, 0.4, "Medium Load"),
        (0.8, 0.7, "High Load"),
        (1.0, 1.0, "Max Load"),
    ]
    
    for utilization, mem_util, label in test_cases:
        power = gpu_model.estimate_power(utilization, mem_util)
        print(f"{label:12} - Compute: {utilization*100:3.0f}%, Memory: {mem_util*100:3.0f}% -> Power: {power:.2f}W")
    
    print()


def demonstrate_power_profiling():
    """Demonstrate power profiling utilities"""
    print("=== Power Profiling Utilities ===")
    
    profiler = PowerProfiler()
    
    # Get current power estimates
    current_cpu_power = profiler.get_current_cpu_power()
    current_gpu_power = profiler.get_current_gpu_power()
    
    print(f"Current estimated CPU power: {current_cpu_power:.2f}W")
    print(f"Current estimated GPU power: {current_gpu_power:.2f}W")
    print(f"Total estimated system power: {current_cpu_power + current_gpu_power:.2f}W")
    
    # Profile a sample workload
    def sample_workload():
        # Simulate some CPU work
        result = 0
        for i in range(100000):
            result += i * i
        return result
    
    print("\nProfiling a sample workload...")
    result, profile_data = profiler.profile_workload(sample_workload)
    
    print(f"Workload completed with result: {result}")
    print(f"Profile data: {profile_data}")
    
    print()


def demonstrate_high_level_functions():
    """Demonstrate high-level power estimation functions"""
    print("=== High-Level Power Estimation Functions ===")
    
    # CPU power estimation
    cpu_power = estimate_cpu_power(utilization=0.7, frequency_ratio=1.5)
    print(f"CPU Power (70% util, 1.5x freq): {cpu_power:.2f}W")
    
    # GPU power estimation
    gpu_power = estimate_gpu_power(utilization=0.6, memory_utilization=0.5)
    print(f"GPU Power (60% util, 50% mem): {gpu_power:.2f}W")
    
    print()


def demonstrate_integration_with_power_management():
    """Demonstrate integration with existing power management system"""
    print("=== Integration with Power Management System ===")
    
    # Create constraints for Intel i5-10210U + NVIDIA SM61
    constraints = PowerConstraint(
        max_cpu_power_watts=25.0,  # Intel i5-10210U TDP
        max_gpu_power_watts=25.0,  # NVIDIA SM61 mobile GPU
    )
    
    # Create scheduler with enhanced power models
    scheduler = PowerAwareScheduler(constraints)
    
    # Get current power state using enhanced models
    power_state = scheduler.get_system_power_state()
    
    print(f"Power State via Scheduler:")
    print(f"  CPU Usage: {power_state.cpu_usage_percent:.1f}%")
    print(f"  CPU Power: {power_state.cpu_power_watts:.2f}W")
    print(f"  GPU Usage: {power_state.gpu_usage_percent:.1f}%")
    print(f"  GPU Power: {power_state.gpu_power_watts:.2f}W")
    
    # Get power model info
    power_info = scheduler.get_power_model_info()
    print(f"Power Model Info: {power_info}")
    
    print()


def demonstrate_integrated_framework():
    """Demonstrate the integrated power management framework"""
    print("=== Integrated Power Management Framework ===")
    
    # Create and start the optimized framework
    framework = PowerManagementFramework()
    
    # Get power consumption estimate
    power_estimate = framework.get_power_consumption_estimate()
    print(f"Framework Power Estimate: {power_estimate}")
    
    # Get framework summary
    summary = framework.get_framework_summary()
    print(f"Framework Status: Active={summary['active']}")
    print(f"  Scheduler Mode: {summary['scheduler_status']['mode']}")
    print(f"  Pending Tasks: {summary['scheduler_status']['pending_tasks']}")
    print(f"  Running Tasks: {summary['scheduler_status']['running_tasks']}")
    
    print()


def main():
    """Main demonstration function"""
    print("Power Estimation Models for Intel i5-10210U and NVIDIA SM61")
    print("=========================================================")
    print()
    
    demonstrate_cpu_power_model()
    demonstrate_gpu_power_model()
    demonstrate_power_profiling()
    demonstrate_high_level_functions()
    demonstrate_integration_with_power_management()
    demonstrate_integrated_framework()
    
    print("All demonstrations completed successfully!")
    print()
    print("Key Features:")
    print("- Accurate power estimation models based on Intel i5-10210U and NVIDIA SM61 specifications")
    print("- Dynamic power calculation based on utilization and frequency/memory usage")
    print("- Integration with existing power management system")
    print("- Power profiling utilities for workload analysis")
    print("- Error handling and validation for robust operation")
    print("- Backward compatibility with existing code")


if __name__ == "__main__":
    main()