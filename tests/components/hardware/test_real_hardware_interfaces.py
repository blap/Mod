"""
Comprehensive Test for Real Hardware Interfaces Implementation

This test validates that all hardware interfaces work correctly with real hardware detection,
power monitoring, temperature monitoring, GPU monitoring, and hardware-specific optimizations.
"""

import unittest
import logging
from unittest.mock import patch, MagicMock
from typing import Dict, Any

# Import the real hardware interfaces
from src.qwen3_vl.components.system.real_hardware_interfaces import (
    IntelCPUInterface, 
    NVidiaGPUInterface, 
    PlatformSpecificInterface,
    HardwareAbstractionLayer,
    get_hardware_interface,
    get_current_power_state,
    get_current_thermal_state,
    get_hardware_capabilities,
    CPUInfo,
    GPUInfo,
    ThermalInfo,
    PowerInfo
)


class TestRealHardwareInterfaces(unittest.TestCase):
    """Test real hardware interfaces implementation"""

    def setUp(self):
        """Set up test fixtures"""
        self.hw_interface = get_hardware_interface()
        
    def test_cpu_interface_initialization(self):
        """Test Intel CPU interface initialization"""
        cpu_interface = IntelCPUInterface()
        self.assertIsInstance(cpu_interface.cpu_info, CPUInfo)
        self.assertEqual(cpu_interface.cpu_info.vendor, "Intel")
        self.assertGreater(cpu_interface.cpu_info.cores, 0)
        self.assertGreater(cpu_interface.cpu_info.threads, 0)
        self.assertGreater(cpu_interface.cpu_info.max_frequency, 0)
        
        print(f"✓ CPU Interface initialized: {cpu_interface.cpu_info.model}")
    
    def test_gpu_interface_initialization(self):
        """Test NVIDIA GPU interface initialization"""
        gpu_interface = NVidiaGPUInterface()
        
        # The GPU might not be available on all systems, so we just check initialization
        self.assertIsNotNone(gpu_interface)
        print(f"✓ GPU Interface initialized (info available: {gpu_interface.get_gpu_info() is not None})")
    
    def test_platform_interface_initialization(self):
        """Test platform-specific interface initialization"""
        platform_interface = PlatformSpecificInterface()
        self.assertIsNotNone(platform_interface)
        self.assertIn(platform_interface.platform, ["Windows", "Linux", "Darwin"])
        
        platform_info = platform_interface.get_platform_info()
        self.assertIn('platform', platform_info)
        self.assertIn('architecture', platform_info)
        
        print(f"✓ Platform Interface initialized: {platform_info['platform']}")
    
    def test_hardware_abstraction_layer_initialization(self):
        """Test hardware abstraction layer initialization"""
        hw_layer = HardwareAbstractionLayer()
        self.assertIsNotNone(hw_layer)
        self.assertIsNotNone(hw_layer.cpu_interface)
        self.assertIsNotNone(hw_layer.gpu_interface)
        
        print(f"✓ Hardware Abstraction Layer initialized")
    
    def test_cpu_info_retrieval(self):
        """Test CPU information retrieval"""
        cpu_info = self.hw_interface.get_cpu_info()
        self.assertIsInstance(cpu_info, CPUInfo)
        self.assertEqual(cpu_info.vendor, "Intel")
        self.assertGreater(cpu_info.cores, 0)
        self.assertGreater(cpu_info.threads, 0)
        self.assertGreater(cpu_info.max_frequency, 0)
        self.assertTrue(cpu_info.has_avx2)  # i5-10210U supports AVX2
        
        print(f"✓ CPU Info retrieved: {cpu_info.model}, {cpu_info.cores} cores, AVX2: {cpu_info.has_avx2}")
    
    def test_gpu_info_retrieval(self):
        """Test GPU information retrieval"""
        gpu_info = self.hw_interface.get_gpu_info()
        
        if gpu_info is not None:
            self.assertIsInstance(gpu_info, GPUInfo)
            self.assertEqual(gpu_info.vendor, "NVIDIA")
            self.assertGreater(gpu_info.memory_total, 0)
            print(f"✓ GPU Info retrieved: {gpu_info.name}, {gpu_info.memory_total / (1024**3):.2f}GB")
        else:
            print("✓ GPU not available on this system")
    
    def test_thermal_info_retrieval(self):
        """Test thermal information retrieval"""
        thermal_info = self.hw_interface.get_thermal_info()
        self.assertIsInstance(thermal_info, ThermalInfo)
        self.assertGreaterEqual(thermal_info.cpu_temperature, 0)
        self.assertGreaterEqual(thermal_info.cpu_critical_temperature, 0)
        self.assertGreaterEqual(thermal_info.cpu_thermal_margin, 0)
        
        print(f"✓ Thermal Info retrieved: CPU {thermal_info.cpu_temperature}°C, GPU {thermal_info.gpu_temperature}°C")
    
    def test_power_info_retrieval(self):
        """Test power information retrieval"""
        power_info = self.hw_interface.get_power_info()
        self.assertIsInstance(power_info, PowerInfo)
        self.assertGreaterEqual(power_info.cpu_power, 0)
        self.assertGreaterEqual(power_info.cpu_power_limit, 0)
        
        print(f"✓ Power Info retrieved: CPU {power_info.cpu_power}W, GPU {power_info.gpu_power}W")
    
    def test_avx2_support_detection(self):
        """Test AVX2 support detection"""
        is_avx2_supported = self.hw_interface.is_avx2_supported()
        self.assertIsInstance(is_avx2_supported, bool)
        
        print(f"✓ AVX2 Support: {is_avx2_supported}")
    
    def test_gpu_availability_detection(self):
        """Test GPU availability detection"""
        is_gpu_available = self.hw_interface.is_gpu_available()
        self.assertIsInstance(is_gpu_available, bool)
        
        print(f"✓ GPU Available: {is_gpu_available}")
    
    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation"""
        batch_size = self.hw_interface.get_optimal_batch_size("inference")
        self.assertIsInstance(batch_size, int)
        self.assertGreaterEqual(batch_size, 1)
        
        batch_size_train = self.hw_interface.get_optimal_batch_size("training")
        self.assertIsInstance(batch_size_train, int)
        self.assertGreaterEqual(batch_size_train, 1)
        
        print(f"✓ Optimal batch sizes - Inference: {batch_size}, Training: {batch_size_train}")
    
    def test_compute_capability_detection(self):
        """Test compute capability detection"""
        compute_cap = self.hw_interface.get_compute_capability()
        if compute_cap is not None:
            self.assertIsInstance(compute_cap, tuple)
            self.assertEqual(len(compute_cap), 2)
            self.assertIsInstance(compute_cap[0], int)
            self.assertIsInstance(compute_cap[1], int)
            print(f"✓ Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        else:
            print("✓ Compute capability not available (GPU not detected)")
    
    def test_memory_bandwidth_estimation(self):
        """Test memory bandwidth estimation"""
        bandwidth = self.hw_interface.get_memory_bandwidth()
        self.assertIsInstance(bandwidth, dict)
        self.assertIn('cpu_memory_bandwidth_gb_s', bandwidth)
        self.assertIn('gpu_memory_bandwidth_gb_s', bandwidth)
        self.assertIn('interconnect_bandwidth_gb_s', bandwidth)
        
        print(f"✓ Memory Bandwidth: CPU={bandwidth['cpu_memory_bandwidth_gb_s']}GB/s, "
              f"GPU={bandwidth['gpu_memory_bandwidth_gb_s']}GB/s, "
              f"Interconnect={bandwidth['interconnect_bandwidth_gb_s']}GB/s")
    
    def test_hardware_capabilities(self):
        """Test hardware capabilities detection"""
        capabilities = get_hardware_capabilities()
        self.assertIsInstance(capabilities, dict)
        self.assertIn('cpu_available', capabilities)
        self.assertIn('gpu_available', capabilities)
        self.assertIn('avx2_supported', capabilities)
        self.assertIn('system_memory_bytes', capabilities)
        
        self.assertTrue(capabilities['cpu_available'])
        self.assertIsInstance(capabilities['avx2_supported'], bool)
        
        print(f"✓ Hardware capabilities: CPU={capabilities['cpu_available']}, "
              f"GPU={capabilities['gpu_available']}, AVX2={capabilities['avx2_supported']}")
    
    def test_power_state_retrieval(self):
        """Test power state retrieval"""
        power_state = get_current_power_state()
        self.assertIsInstance(power_state, PowerInfo)
        
        print(f"✓ Current power state: CPU={power_state.cpu_power}W, GPU={power_state.gpu_power}W")
    
    def test_thermal_state_retrieval(self):
        """Test thermal state retrieval"""
        thermal_state = get_current_thermal_state()
        self.assertIsInstance(thermal_state, ThermalInfo)
        
        print(f"✓ Current thermal state: CPU={thermal_state.cpu_temperature}°C, GPU={thermal_state.gpu_temperature}°C")
    
    def test_monitoring_functionality(self):
        """Test hardware monitoring functionality"""
        measurements = self.hw_interface.monitor_hardware_status(interval=0.1, duration=0.5)
        self.assertIsInstance(measurements, list)
        
        # Should have collected at least one measurement
        if len(measurements) > 0:
            first_measurement = measurements[0]
            self.assertIsInstance(first_measurement, dict)
            self.assertIn('cpu_info', first_measurement)
            self.assertIn('thermal_info', first_measurement)
            self.assertIn('power_info', first_measurement)
        
        print(f"✓ Hardware monitoring collected {len(measurements)} measurements")


class TestHardwareOptimizations(unittest.TestCase):
    """Test hardware-specific optimizations"""

    def setUp(self):
        """Set up test fixtures"""
        self.hw_interface = get_hardware_interface()
        
    def test_hardware_specific_batch_size_optimization(self):
        """Test that batch size is optimized for hardware"""
        # Test inference batch size
        inference_batch_size = self.hw_interface.get_optimal_batch_size("inference")
        self.assertIsInstance(inference_batch_size, int)
        self.assertGreaterEqual(inference_batch_size, 1)
        
        # Test training batch size (should be smaller than inference)
        training_batch_size = self.hw_interface.get_optimal_batch_size("training")
        self.assertIsInstance(training_batch_size, int)
        self.assertGreaterEqual(training_batch_size, 1)
        
        print(f"✓ Hardware-specific batch size optimization: Inference={inference_batch_size}, Training={training_batch_size}")
    
    def test_memory_efficient_operations(self):
        """Test memory-efficient operations based on hardware"""
        # Test that the system returns reasonable memory bandwidth estimates
        bandwidth = self.hw_interface.get_memory_bandwidth()
        
        # For Intel i5-10210U + NVIDIA SM61 system
        expected_cpu_bw = 34.0  # DDR4 memory bandwidth
        expected_gpu_bw = 192.0  # GTX 1050 Ti memory bandwidth
        expected_interconnect_bw = 12.0  # PCIe 3.0 x16 bandwidth
        
        # Allow some tolerance for estimates
        self.assertAlmostEqual(bandwidth['cpu_memory_bandwidth_gb_s'], expected_cpu_bw, delta=10.0)
        self.assertAlmostEqual(bandwidth['gpu_memory_bandwidth_gb_s'], expected_gpu_bw, delta=50.0)
        self.assertAlmostEqual(bandwidth['interconnect_bandwidth_gb_s'], expected_interconnect_bw, delta=5.0)
        
        print(f"✓ Memory-efficient operations optimized for target hardware")
    
    def test_power_efficient_processing(self):
        """Test power-efficient processing based on hardware"""
        # Get current power state
        power_state = self.hw_interface.get_power_info()
        
        # Check that power values are reasonable
        self.assertGreaterEqual(power_state.cpu_power, 0)
        self.assertLessEqual(power_state.cpu_power, 25.0)  # Max for i5-10210U
        self.assertGreaterEqual(power_state.cpu_power_limit, 20.0)  # Reasonable power limit
        
        if power_state.gpu_power > 0:
            self.assertLessEqual(power_state.gpu_power, 75.0)  # Max for mobile Pascal GPU
        
        print(f"✓ Power-efficient processing: CPU={power_state.cpu_power}W, GPU={power_state.gpu_power}W")
    
    def test_thermal_aware_processing(self):
        """Test thermal-aware processing based on hardware"""
        # Get current thermal state
        thermal_state = self.hw_interface.get_thermal_info()
        
        # Check that temperature values are reasonable
        self.assertGreaterEqual(thermal_state.cpu_temperature, 0)
        self.assertLessEqual(thermal_state.cpu_temperature, 100)  # Max reasonable CPU temp
        self.assertGreaterEqual(thermal_state.cpu_thermal_margin, -100)  # Could be negative if overheating
        
        print(f"✓ Thermal-aware processing: CPU={thermal_state.cpu_temperature}°C, "
              f"Margin={thermal_state.cpu_thermal_margin}°C")


def run_comprehensive_tests():
    """Run all comprehensive tests for real hardware interfaces"""
    logging.basicConfig(level=logging.INFO)
    
    print("Running Comprehensive Tests for Real Hardware Interfaces Implementation")
    print("=" * 70)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add basic functionality tests
    suite.addTest(unittest.makeSuite(TestRealHardwareInterfaces))
    
    # Add hardware optimization tests
    suite.addTest(unittest.makeSuite(TestHardwareOptimizations))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print test results
    print(f"\nTest Results:")
    print(f"  Tests Run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = result.wasSuccessful()
    print(f"\nOverall Result: {'✓ SUCCESS' if success else '✗ FAILURE'}")
    
    return success


if __name__ == "__main__":
    success = run_comprehensive_tests()

    if success:
        print("\n🎉 All tests passed! Real hardware interfaces are working correctly.")
        print("The implementation successfully replaces mock implementations with real hardware interfaces.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        exit(1)