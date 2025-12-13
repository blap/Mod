#!/usr/bin/env python3
"""
Test script to verify that the restructured code maintains proper functionality.
This script tests imports and basic functionality of moved modules.
"""

import sys
import os

# Add the src directory to the Python path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_memory_imports():
    """Test that memory management modules can be imported."""
    print("Testing memory management imports...")
    try:
        from qwen3_vl.components.memory.optimized_buddy_allocator import OptimizedBuddyAllocator, OptimizedMemoryPool
        print("+ Successfully imported OptimizedBuddyAllocator and OptimizedMemoryPool")

        from qwen3_vl.components.memory.optimized_locking_strategies import ReaderWriterLock, LockStriping
        print("+ Successfully imported locking strategies")

        # Test basic functionality
        allocator = OptimizedBuddyAllocator(total_size=1024*1024)  # 1MB
        print("+ Successfully created OptimizedBuddyAllocator instance")

        pool = OptimizedMemoryPool
        print("+ Successfully referenced OptimizedMemoryPool class")
        
        return True
    except Exception as e:
        print(f"✗ Error testing memory imports: {e}")
        return False

def test_cpu_optimization_imports():
    """Test that CPU optimization modules can be imported."""
    print("\nTesting CPU optimization imports...")
    try:
        from qwen3_vl.components.optimization.advanced_cpu_optimizations_intel_i5_10210u import (
            AdvancedCPUOptimizationConfig,
            IntelCPUOptimizedPreprocessor,
            IntelOptimizedPipeline
        )
        print("+ Successfully imported Intel optimization components")

        # Test basic functionality
        config = AdvancedCPUOptimizationConfig()
        print("+ Successfully created AdvancedCPUOptimizationConfig instance")

        return True
    except Exception as e:
        print(f"X Error testing CPU optimization imports: {e}")
        return False

def test_power_management_imports():
    """Test that power management modules can be imported."""
    print("\nTesting power management imports...")
    try:
        from qwen3_vl.components.system.power_management import PowerState, PowerConstraint, PowerMonitor
        print("+ Successfully imported power management components")

        # Test basic functionality
        constraint = PowerConstraint()
        monitor = PowerMonitor(constraint)
        print("+ Successfully created PowerConstraint and PowerMonitor instances")

        return True
    except Exception as e:
        print(f"X Error testing power management imports: {e}")
        return False

def test_adaptive_algorithms_imports():
    """Test that adaptive algorithms modules can be imported."""
    print("\nTesting adaptive algorithms imports...")
    try:
        from qwen3_vl.components.routing.adaptive_algorithms import (
            AdaptiveParameters,
            AdaptiveController,
            AdaptationStrategy
        )
        print("+ Successfully imported adaptive algorithms components")

        # Test basic functionality
        params = AdaptiveParameters()
        print("+ Successfully created AdaptiveParameters instance")

        return True
    except Exception as e:
        print(f"X Error testing adaptive algorithms imports: {e}")
        return False

def test_hardware_abstraction_imports():
    """Test that hardware abstraction modules can be imported."""
    print("\nTesting hardware abstraction imports...")
    try:
        from qwen3_vl.components.hardware.hardware_abstraction_layer import HardwareManager
        print("+ Successfully imported hardware abstraction components")

        # Test basic functionality
        hw_manager = HardwareManager()
        print("+ Successfully created HardwareManager instance")

        return True
    except Exception as e:
        print(f"X Error testing hardware abstraction imports: {e}")
        return False

def test_utils_imports():
    """Test that utility modules can be imported."""
    print("\nTesting utility imports...")
    try:
        from qwen3_vl.utils.centralized_metrics_collector import (
            CentralizedMetricsCollector,
            record_metric,
            record_timing
        )
        print("+ Successfully imported metrics collector components")

        from qwen3_vl.utils.timing_utilities import time_function, time_block
        print("+ Successfully imported timing utilities")

        # Test basic functionality
        collector = CentralizedMetricsCollector.get_instance()
        print("+ Successfully got CentralizedMetricsCollector instance")

        return True
    except Exception as e:
        print(f"X Error testing utility imports: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing restructured Qwen3-VL project...")
    print("="*50)
    
    results = []
    results.append(test_memory_imports())
    results.append(test_cpu_optimization_imports())
    results.append(test_power_management_imports())
    results.append(test_adaptive_algorithms_imports())
    results.append(test_hardware_abstraction_imports())
    results.append(test_utils_imports())
    
    print("\n" + "="*50)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("+ All tests passed! The restructuring was successful.")
        return 0
    else:
        print("X Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    exit(main())