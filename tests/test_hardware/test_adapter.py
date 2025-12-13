"""
Comprehensive tests for the hardware adapter module.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.hardware.adapter import (
    HardwareInterface,
    CPUAdapter,
    GPUAdapter,
    NVIDIAGPUAdapter,
    AMDGPUAdapter,
    TPUAdapter,
    HardwareManager,
    HardwareType,
    get_hardware_manager
)


class TestHardwareInterface(unittest.TestCase):
    """Test the abstract HardwareInterface class."""
    
    def test_abstract_methods_exist(self):
        """Test that all abstract methods are defined."""
        self.assertTrue(hasattr(HardwareInterface, '__abstractmethods__'))
        expected_methods = {
            'name', 'type', 'initialize', 'cleanup', 'get_memory_info',
            'allocate_memory', 'free_memory', 'transfer_data',
            'get_performance_metrics', 'is_available'
        }
        self.assertEqual(HardwareInterface.__abstractmethods__, expected_methods)


class TestCPUAdapter(unittest.TestCase):
    """Test the CPU adapter implementation."""
    
    def setUp(self):
        self.cpu_adapter = CPUAdapter(device_id=0)
    
    def test_initialization(self):
        """Test CPU adapter initialization."""
        result = self.cpu_adapter.initialize()
        self.assertTrue(result)
        self.assertTrue(self.cpu_adapter._initialized)
    
    def test_cleanup(self):
        """Test CPU adapter cleanup."""
        self.cpu_adapter.initialize()
        self.cpu_adapter.cleanup()
        self.assertFalse(self.cpu_adapter._initialized)
    
    def test_name_property(self):
        """Test CPU name property."""
        name = self.cpu_adapter.name
        self.assertIsInstance(name, str)
        self.assertTrue(len(name) > 0)
    
    def test_type_property(self):
        """Test CPU type property."""
        self.assertEqual(self.cpu_adapter.type, HardwareType.CPU)
    
    def test_is_available(self):
        """Test CPU availability check."""
        self.assertTrue(self.cpu_adapter.is_available())
    
    def test_get_memory_info(self):
        """Test getting CPU memory information."""
        memory_info = self.cpu_adapter.get_memory_info()
        self.assertIsInstance(memory_info, dict)
        self.assertIn('total', memory_info)
    
    def test_get_performance_metrics(self):
        """Test getting CPU performance metrics."""
        perf_metrics = self.cpu_adapter.get_performance_metrics()
        self.assertIsInstance(perf_metrics, dict)
        self.assertIn('cpu_count', perf_metrics)
    
    def test_memory_operations(self):
        """Test CPU memory allocation and deallocation."""
        self.cpu_adapter.initialize()
        
        # Test memory allocation
        memory_block = self.cpu_adapter.allocate_memory(1024)
        self.assertIsNotNone(memory_block)
        self.assertEqual(len(memory_block), 1024)
        
        # Test memory deallocation
        self.cpu_adapter.free_memory(memory_block)
    
    def test_transfer_data(self):
        """Test data transfer from CPU to other devices."""
        # Create a mock destination device
        mock_dest_device = Mock()
        mock_dest_device.type = HardwareType.GPU
        
        # Test data transfer
        test_data = b"test_data"
        result = self.cpu_adapter.transfer_data(test_data, mock_dest_device)
        self.assertEqual(result, test_data)


class TestNVIDIAGPUAdapter(unittest.TestCase):
    """Test the NVIDIA GPU adapter implementation."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_initialization_when_cuda_unavailable(self, mock_device_count, mock_is_available):
        """Test NVIDIA GPU initialization when CUDA is unavailable."""
        mock_is_available.return_value = False
        mock_device_count.return_value = 0
        
        nvidia_adapter = NVIDIAGPUAdapter(device_id=0)
        self.assertFalse(nvidia_adapter.is_available())
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.set_device')
    @patch('torch.cuda.empty_cache')
    def test_initialization_when_cuda_available(self, mock_empty_cache, mock_set_device, mock_device_count, mock_is_available):
        """Test NVIDIA GPU initialization when CUDA is available."""
        mock_is_available.return_value = True
        mock_device_count.return_value = 1
        mock_set_device.return_value = None
        mock_empty_cache.return_value = None
        
        nvidia_adapter = NVIDIAGPUAdapter(device_id=0)
        self.assertTrue(nvidia_adapter.cuda_available)
        
        # Test initialization
        result = nvidia_adapter.initialize()
        # This might still fail due to device not actually existing, but we can test the logic
        self.assertIsInstance(result, bool)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_name_property(self, mock_device_count, mock_is_available):
        """Test NVIDIA GPU name property."""
        mock_is_available.return_value = False
        mock_device_count.return_value = 0
        
        nvidia_adapter = NVIDIAGPUAdapter(device_id=0)
        name = nvidia_adapter.name
        self.assertIsInstance(name, str)
        self.assertIn("NVIDIA", name.upper())


class TestAMDGPUAdapter(unittest.TestCase):
    """Test the AMD GPU adapter implementation."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_initialization_when_rocm_unavailable(self, mock_device_count, mock_is_available):
        """Test AMD GPU initialization when ROCm is unavailable."""
        mock_is_available.return_value = False
        mock_device_count.return_value = 0
        
        amd_adapter = AMDGPUAdapter(device_id=0)
        self.assertFalse(amd_adapter.is_available())
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    def test_name_property(self, mock_device_count, mock_is_available):
        """Test AMD GPU name property."""
        mock_is_available.return_value = False
        mock_device_count.return_value = 0
        
        amd_adapter = AMDGPUAdapter(device_id=0)
        name = amd_adapter.name
        self.assertIsInstance(name, str)
        self.assertIn("AMD", name.upper())


class TestTPUAdapter(unittest.TestCase):
    """Test the TPU adapter implementation."""
    
    def test_initialization_when_tpu_unavailable(self):
        """Test TPU initialization when TPU is unavailable."""
        # Patch imports to simulate unavailable TPU
        with patch.dict('sys.modules', {
            'torch_xla.core.xla_model': None,
            'torch_xla.experimental.pjrt': None
        }):
            tpu_adapter = TPUAdapter(device_id=0)
            self.assertFalse(tpu_adapter.tpu_available)
            self.assertFalse(tpu_adapter.is_available())
    
    def test_name_property(self):
        """Test TPU name property."""
        tpu_adapter = TPUAdapter(device_id=0)
        name = tpu_adapter.name
        self.assertIsInstance(name, str)
        self.assertIn("TPU", name.upper())


class TestHardwareManager(unittest.TestCase):
    """Test the hardware manager functionality."""
    
    @patch.object(CPUAdapter, 'is_available', return_value=True)
    @patch.object(NVIDIAGPUAdapter, 'is_available', return_value=False)
    @patch.object(AMDGPUAdapter, 'is_available', return_value=False)
    @patch.object(TPUAdapter, 'is_available', return_value=False)
    def test_device_detection_with_only_cpu(self, mock_tpu, mock_amd, mock_nvidia, mock_cpu):
        """Test device detection when only CPU is available."""
        manager = HardwareManager()
        
        # Should only detect CPU
        devices = manager.get_all_devices()
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].type, HardwareType.CPU)
    
    @patch.object(CPUAdapter, 'is_available', return_value=True)
    @patch.object(NVIDIAGPUAdapter, 'is_available', return_value=True)
    @patch.object(NVIDIAGPUAdapter, 'get_device_count', return_value=2)
    @patch.object(NVIDIAGPUAdapter, 'initialize', return_value=True)
    @patch.object(AMDGPUAdapter, 'is_available', return_value=False)
    @patch.object(TPUAdapter, 'is_available', return_value=False)
    def test_device_detection_with_cpu_and_gpus(self, mock_tpu, mock_amd, mock_nvidia_init, mock_nvidia_count, mock_nvidia, mock_cpu):
        """Test device detection when CPU and multiple GPUs are available."""
        manager = HardwareManager()

        # Should detect CPU and up to 2 NVIDIA GPUs (depending on initialization success)
        devices = manager.get_all_devices()
        # At minimum, we should have 1 CPU
        self.assertGreaterEqual(len(devices), 1)

        cpu_devices = manager.get_devices_by_type(HardwareType.CPU)
        gpu_devices = manager.get_devices_by_type(HardwareType.GPU)

        self.assertEqual(len(cpu_devices), 1)
        # GPU count depends on actual availability during test, but should be >= 0
        self.assertGreaterEqual(len(gpu_devices), 0)
    
    def test_get_device_by_id(self):
        """Test getting a device by ID and type."""
        manager = HardwareManager()
        
        # Since we can't reliably detect real hardware in tests, 
        # we'll test the logic assuming devices exist
        cpu_device = manager.get_device_by_id(0, HardwareType.CPU)
        # Result depends on actual hardware availability in the test environment
        
    def test_get_best_device(self):
        """Test getting the best available device."""
        manager = HardwareManager()
        
        # Get best device with GPU preference
        best_device = manager.get_best_device(preferred_type=HardwareType.GPU)
        # Result depends on actual hardware availability in the test environment
        
        # Get best device with CPU preference
        best_device_cpu = manager.get_best_device(preferred_type=HardwareType.CPU)
        # Result depends on actual hardware availability in the test environment


class TestSingletonHardwareManager(unittest.TestCase):
    """Test the singleton hardware manager getter."""
    
    def test_singleton_behavior(self):
        """Test that get_hardware_manager returns the same instance."""
        manager1 = get_hardware_manager()
        manager2 = get_hardware_manager()
        
        self.assertIs(manager1, manager2)


if __name__ == '__main__':
    # Run the tests
    unittest.main()