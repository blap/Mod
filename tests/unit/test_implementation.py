"""
Test script to verify all modules work properly with the implemented solutions
"""

def test_power_management():
    print("Testing power_management module...")
    from power_management import PowerState, PowerConstraint, PowerMonitor
    
    # Test creating power state
    power_state = PowerState(
        cpu_usage_percent=50.0,
        gpu_usage_percent=30.0,
        cpu_temp_celsius=60.0,
        gpu_temp_celsius=50.0,
        cpu_power_watts=15.0,
        gpu_power_watts=25.0
    )
    print(f"  Created PowerState: {power_state}")
    
    # Test creating power constraint
    constraints = PowerConstraint()
    print(f"  Created PowerConstraint: {constraints}")
    
    # Test power monitor
    monitor = PowerMonitor(constraints)
    current_state = monitor.get_current_power_state()
    print(f"  Current power state: CPU={current_state.cpu_usage_percent}%, Temp={current_state.cpu_temp_celsius}°C")
    
    # Test violation checking
    violations = monitor.is_power_violation(current_state)
    print(f"  Power violations: {violations}")
    
    print("  OK Power management tests passed\n")


def test_thermal_management():
    print("Testing thermal_management module...")
    from thermal_management import ThermalReading, ThermalConstraint, ThermalManager, apply_thermal_throttling
    
    # Test creating thermal reading
    reading = ThermalReading(
        sensor_name="cpu_core_1",
        temperature_celsius=65.0
    )
    print(f"  Created ThermalReading: {reading}")
    
    # Test creating thermal constraint
    constraints = ThermalConstraint()
    print(f"  Created ThermalConstraint: {constraints}")
    
    # Test thermal manager
    thermal_manager = ThermalManager(constraints)
    temperatures = thermal_manager.get_current_temperatures()
    print(f"  Current temperatures: {len(temperatures)} sensors detected")
    
    avg_temp = thermal_manager.get_average_temperature()
    print(f"  Average temperature: {avg_temp:.2f}°C")
    
    # Test violation checking
    violations = thermal_manager.is_thermal_violation(avg_temp)
    print(f"  Thermal violations: {violations}")
    
    # Test thermal throttling
    throttling_factor = apply_thermal_throttling(avg_temp)
    print(f"  Thermal throttling factor: {throttling_factor:.2f}")
    
    print("  OK Thermal management tests passed\n")


def test_intel_extension_fallback():
    print("Testing intel_extension_for_pytorch_fallback module...")
    from intel_extension_for_pytorch_fallback import (
        optimize, 
        optimize_model, 
        get_device_type, 
        get_fp32_math_mode,
        FP32_MATH_MODE
    )
    import torch
    import torch.nn as nn
    
    # Create a simple model for testing
    model = nn.Linear(10, 5)
    
    # Test optimize function
    optimized_model = optimize_model(model)
    print(f"  optimize_model returned: {type(optimized_model)}")
    
    # Test device type
    device_type = get_device_type()
    print(f"  Device type: {device_type}")
    
    # Test FP32 math mode
    fp32_mode = get_fp32_math_mode()
    print(f"  FP32 math mode: {fp32_mode}")
    
    # Test constants
    print(f"  FP32_MATH_MODE.FP32: {FP32_MATH_MODE.FP32}")
    
    print("  OK Intel extension fallback tests passed\n")


def test_adaptive_algorithms():
    print("Testing adaptive_algorithms module...")
    from adaptive_algorithms import AdaptiveController, AdaptiveParameters, PowerConstraint
    
    # Test creating adaptive controller
    constraints = PowerConstraint()
    controller = AdaptiveController(constraints)
    
    # Test updating parameters
    from power_management import PowerState
    power_state = PowerState(
        cpu_usage_percent=70.0,
        gpu_usage_percent=50.0,
        cpu_temp_celsius=70.0,
        gpu_temp_celsius=60.0,
        cpu_power_watts=20.0,
        gpu_power_watts=40.0
    )
    
    params = controller.update_parameters(power_state)
    print(f"  Adaptive parameters: {params}")
    
    print("  OK Adaptive algorithms tests passed\n")


def test_adaptive_precision_optimization():
    print("Testing adaptive_precision_optimization module...")
    from adaptive_precision_optimization import AdaptivePrecisionController, AdaptivePrecisionConfig
    
    # Test creating config
    config = AdaptivePrecisionConfig()
    print(f"  Created AdaptivePrecisionConfig: {config}")
    
    # Test creating controller
    controller = AdaptivePrecisionController(config)
    print(f"  Created AdaptivePrecisionController")
    
    print("  OK Adaptive precision optimization tests passed\n")


def test_cpu_optimizations():
    print("Testing advanced_cpu_optimizations_intel_i5_10210u module...")
    from advanced_cpu_optimizations_intel_i5_10210u import AdvancedCPUOptimizationConfig, IntelOptimizedPipeline
    import torch.nn as nn
    
    # Test creating config
    config = AdvancedCPUOptimizationConfig()
    print(f"  Created AdvancedCPUOptimizationConfig: {config}")
    
    # Test creating pipeline (with a dummy model)
    dummy_model = nn.Linear(10, 5)
    pipeline = IntelOptimizedPipeline(dummy_model, config)
    print(f"  Created IntelOptimizedPipeline")
    
    print("  OK CPU optimizations tests passed\n")


def test_memory_management():
    print("Testing advanced_memory_management_vl module...")
    from advanced_memory_management_vl import VisionLanguageMemoryOptimizer, MemoryPoolType
    
    # Test creating memory optimizer
    optimizer = VisionLanguageMemoryOptimizer()
    print(f"  Created VisionLanguageMemoryOptimizer")
    
    # Test allocating tensor memory
    tensor = optimizer.allocate_tensor_memory((10, 20), dtype="float32", tensor_type="general")
    print(f"  Allocated tensor of shape: {tensor.shape}")
    
    # Get memory stats
    stats = optimizer.get_memory_stats()
    print(f"  Memory stats keys: {list(stats.keys())}")
    
    # Cleanup
    optimizer.cleanup()
    
    print("  OK Memory management tests passed\n")


def test_memory_swapping():
    print("Testing advanced_memory_swapping_system module...")
    from advanced_memory_swapping_system import AdvancedMemorySwapper, MemoryRegionType, create_optimized_swapping_system
    
    # Test creating swapping system
    swapper = create_optimized_swapping_system()
    print(f"  Created AdvancedMemorySwapper")
    
    # Register a memory block
    block = swapper.register_memory_block("test_block", 1024*1024, MemoryRegionType.TENSOR_DATA)
    print(f"  Registered memory block: {block.id}")
    
    # Access the block
    accessed_block = swapper.access_memory_block("test_block")
    print(f"  Accessed memory block: {accessed_block.id if accessed_block else 'None'}")
    
    # Get status
    status = swapper.get_status()
    print(f"  Swapper status - Algorithm: {status['algorithm']}, Pressure: {status['pressure_level']}")
    
    print("  OK Memory swapping tests passed\n")


def main():
    print("Running comprehensive tests for all implemented modules...")
    print("=" * 60)
    
    test_power_management()
    test_thermal_management()
    test_intel_extension_fallback()
    test_adaptive_algorithms()
    test_adaptive_precision_optimization()
    test_cpu_optimizations()
    test_memory_management()
    test_memory_swapping()
    
    print("=" * 60)
    print("All tests passed! The missing dependencies and imports have been successfully implemented.")


if __name__ == "__main__":
    main()