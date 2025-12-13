"""
Test Runner for Qwen3-VL Memory Optimization Validation

This script runs all the validation tests for the Qwen3-VL memory optimizations.
"""

import unittest
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def run_all_tests():
    """Run all validation tests"""
    print("Running Qwen3-VL Memory Optimization Validation Tests")
    print("=" * 60)
    
    # Discover and run all tests
    loader = unittest.TestLoader()
    
    # Test suites
    test_files = [
        'comprehensive_memory_optimization_tests',
        'hardware_specific_validation_tests',
        'performance_benchmarking_suite'  # This is a standalone script
    ]
    
    # Run the comprehensive memory optimization tests
    print("\n1. Running Comprehensive Memory Optimization Tests...")
    try:
        from comprehensive_memory_optimization_tests import (
            TestRegressionValidation,
            TestPerformanceComparison,
            TestMemoryEfficiency,
            TestInferenceTimeImprovement,
            TestIntegrationValidation,
            TestHardwareSpecificValidation
        )
        
        regression_suite = loader.loadTestsFromTestCase(TestRegressionValidation)
        performance_suite = loader.loadTestsFromTestCase(TestPerformanceComparison)
        memory_suite = loader.loadTestsFromTestCase(TestMemoryEfficiency)
        inference_suite = loader.loadTestsFromTestCase(TestInferenceTimeImprovement)
        integration_suite = loader.loadTestsFromTestCase(TestIntegrationValidation)
        hardware_suite = loader.loadTestsFromTestCase(TestHardwareSpecificValidation)
        
        all_memory_tests = unittest.TestSuite([
            regression_suite,
            performance_suite,
            memory_suite,
            inference_suite,
            integration_suite,
            hardware_suite
        ])
        
        runner = unittest.TextTestRunner(verbosity=2)
        memory_result = runner.run(all_memory_tests)
        
        print(f"\nMemory Optimization Tests - Ran: {memory_result.testsRun}, "
              f"Failures: {len(memory_result.failures)}, "
              f"Errors: {len(memory_result.errors)}")
              
    except Exception as e:
        print(f"Error running memory optimization tests: {e}")
        import traceback
        traceback.print_exc()
    
    # Run hardware-specific validation tests
    print("\n2. Running Hardware-Specific Validation Tests...")
    try:
        from hardware_specific_validation_tests import (
            TestIntelI5_10210UOptimizations,
            TestNvidiaSM61Optimizations,
            TestNVMeSSDOptimizations,
            TestIntegratedHardwareOptimizations
        )
        
        intel_suite = loader.loadTestsFromTestCase(TestIntelI5_10210UOptimizations)
        nvidia_suite = loader.loadTestsFromTestCase(TestNvidiaSM61Optimizations)
        nvme_suite = loader.loadTestsFromTestCase(TestNVMeSSDOptimizations)
        integrated_suite = loader.loadTestsFromTestCase(TestIntegratedHardwareOptimizations)
        
        hardware_tests = unittest.TestSuite([
            intel_suite,
            nvidia_suite,
            nvme_suite,
            integrated_suite
        ])
        
        runner = unittest.TextTestRunner(verbosity=2)
        hardware_result = runner.run(hardware_tests)
        
        print(f"\nHardware Validation Tests - Ran: {hardware_result.testsRun}, "
              f"Failures: {len(hardware_result.failures)}, "
              f"Errors: {len(hardware_result.errors)}")
              
    except Exception as e:
        print(f"Error running hardware validation tests: {e}")
        import traceback
        traceback.print_exc()
    
    # Run performance benchmarking (this is a standalone script)
    print("\n3. Running Performance Benchmarking...")
    try:
        from performance_benchmarking_suite import run_all_benchmarks
        benchmark_results = run_all_benchmarks()
        print("Performance benchmarking completed successfully!")
    except Exception as e:
        print(f"Error running performance benchmarks: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("ALL VALIDATION TESTS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()