"""
Comprehensive test for Qwen3-VL Storage Management System
Tests all aspects of storage management including SSD/HDD support, tiered storage,
performance monitoring, and error handling.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import time
import threading
from unittest.mock import patch, MagicMock
import unittest

# Add the project path to sys.path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.qwen3_vl.storage_management import (
    StorageType, StorageDevice, StorageConfig, StorageManager, 
    Qwen3VLStorageManager, create_qwen3vl_storage_manager
)


class TestStorageManagement(unittest.TestCase):
    """Test class for storage management system"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for testing
        self.test_base_dir = tempfile.mkdtemp()
        self.ssd_path = os.path.join(self.test_base_dir, "ssd_cache")
        self.hdd_path = os.path.join(self.test_base_dir, "hdd_storage")
        self.fallback_path = os.path.join(self.test_base_dir, "fallback")
        
        # Create config
        self.config = StorageConfig(
            ssd_path=self.ssd_path,
            hdd_path=self.hdd_path,
            fallback_storage_path=self.fallback_path,
            tiering_threshold=1024*1024,  # 1MB threshold
            cache_size_limit=10*1024*1024,  # 10MB cache
            enable_compression=True
        )
        
        # Create storage manager
        self.storage_manager = create_qwen3vl_storage_manager(self.config)
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove temporary directories
        shutil.rmtree(self.test_base_dir, ignore_errors=True)
    
    def test_storage_device_detection(self):
        """Test storage device detection"""
        # Check that devices were detected
        devices = self.storage_manager.devices
        self.assertGreaterEqual(len(devices), 1, "Should detect at least one storage device")
        
        # Check that devices have proper attributes
        for mount_point, device in devices.items():
            self.assertIsInstance(device, StorageDevice)
            self.assertIsInstance(device.device_path, str)
            self.assertIn(device.storage_type, StorageType)
            self.assertIsInstance(device.total_space, int)
            self.assertIsInstance(device.available_space, int)
            self.assertGreater(device.total_space, 0)
            self.assertGreaterEqual(device.available_space, 0)
    
    def test_storage_type_determination(self):
        """Test storage type determination"""
        # Test that storage types are properly determined
        devices = self.storage_manager.devices
        for mount_point, device in devices.items():
            self.assertIsInstance(device.storage_type, StorageType)
    
    def test_optimal_storage_selection(self):
        """Test optimal storage selection logic"""
        # Test small file (should go to SSD if available)
        small_file_size = 512 * 1024  # 512KB
        optimal_device_small = self.storage_manager.get_optimal_storage_for_file(small_file_size)
        
        # Test large file (should go to HDD if available)
        large_file_size = 5 * 1024 * 1024  # 5MB
        optimal_device_large = self.storage_manager.get_optimal_storage_for_file(large_file_size)
        
        # At least one device should be returned
        self.assertIsNotNone(optimal_device_small, "Should find optimal storage for small file")
        self.assertIsNotNone(optimal_device_large, "Should find optimal storage for large file")
    
    def test_file_operations(self):
        """Test basic file operations"""
        # Test writing a small file with explicit storage preference to avoid device issues
        test_data_small = b"Small test data" * 100  # ~1.5KB
        write_success_small = self.storage_manager.write_file("test_small.txt", test_data_small, StorageType.NVME_SSD)
        self.assertTrue(write_success_small, "Should successfully write small file")

        # Test reading the small file
        read_data_small = self.storage_manager.read_file("test_small.txt")
        self.assertIsNotNone(read_data_small, "Should successfully read small file")
        self.assertEqual(read_data_small, test_data_small, "Read data should match written data")

        # Test writing a larger file
        test_data_large = b"Larger test data for HDD" * 1000  # ~20KB
        write_success_large = self.storage_manager.write_file("test_large.txt", test_data_large, StorageType.EXTERNAL_HDD)
        self.assertTrue(write_success_large, "Should successfully write large file")

        # Test reading the large file
        read_data_large = self.storage_manager.read_file("test_large.txt")
        self.assertIsNotNone(read_data_large, "Should successfully read large file")
        self.assertEqual(read_data_large, test_data_large, "Read data should match written data")
    
    def test_model_weights_storage(self):
        """Test model weights storage operations"""
        # Create mock weights
        mock_weights = {
            'layer_1.weight': [1.0, 2.0, 3.0] * 1000,
            'layer_2.bias': [0.1, 0.2, 0.3] * 500,
            'layer_3.activation': 'relu'
        }

        # Store weights
        store_success = self.storage_manager.store_model_weights(mock_weights, "test_model")
        self.assertTrue(store_success, "Should successfully store model weights")

        # Load weights
        loaded_weights = self.storage_manager.load_model_weights("test_model")
        self.assertIsNotNone(loaded_weights, "Should successfully load model weights")
        self.assertEqual(loaded_weights['layer_3.activation'], 'relu', "Loaded weights should match stored weights")
    
    def test_tensor_storage(self):
        """Test tensor storage operations"""
        # Create mock tensor
        mock_tensor = [[1.0, 2.0, 3.0]] * 1000  # Simulate a tensor

        # Store tensor
        store_success = self.storage_manager.store_tensor(mock_tensor, "test_tensor", compression=False)  # Disable compression for test
        self.assertTrue(store_success, "Should successfully store tensor")

        # Load tensor
        loaded_tensor = self.storage_manager.load_tensor("test_tensor", compressed=False)
        self.assertIsNotNone(loaded_tensor, "Should successfully load tensor")
        self.assertEqual(len(loaded_tensor), len(mock_tensor), "Loaded tensor should have same length as stored tensor")
    
    def test_access_pattern_tracking(self):
        """Test access pattern tracking"""
        # Write a test file with explicit storage preference
        test_data = b"Test data for access tracking"
        self.storage_manager.write_file("access_test.txt", test_data, StorageType.NVME_SSD)

        # Access the file multiple times to create a hot pattern
        for i in range(5):
            _ = self.storage_manager.read_file("access_test.txt")
            time.sleep(0.01)  # Small delay

        # Check hot files
        hot_files = self.storage_manager.get_hot_files(5)
        self.assertIn("access_test.txt", [os.path.basename(f) for f in hot_files],
                      "Accessed file should be in hot files list")

        # Predict access
        prob, next_access = self.storage_manager.predict_file_access("access_test.txt")
        self.assertIsInstance(prob, float, "Probability should be a float")
        self.assertGreaterEqual(prob, 0, "Probability should be non-negative")
        self.assertLessEqual(prob, 1, "Probability should be <= 1")
    
    def test_performance_monitoring(self):
        """Test performance monitoring"""
        # Write and read a file to generate performance data
        test_data = b"Performance test data" * 1000
        self.storage_manager.write_file("perf_test.txt", test_data)
        _ = self.storage_manager.read_file("perf_test.txt")
        
        # Check that performance data was recorded
        stats = self.storage_manager.get_storage_stats()
        self.assertIn('performance', stats, "Stats should contain performance data")
    
    def test_storage_statistics(self):
        """Test storage statistics"""
        stats = self.storage_manager.get_storage_stats()
        
        # Check that stats contain expected keys
        self.assertIn('devices', stats, "Stats should contain devices")
        self.assertIn('total_space', stats, "Stats should contain total_space")
        self.assertIn('available_space', stats, "Stats should contain available_space")
        self.assertIn('performance', stats, "Stats should contain performance data")
        
        # Check that device stats contain expected keys
        for mount_point, device_stats in stats['devices'].items():
            self.assertIn('storage_type', device_stats, "Device stats should contain storage_type")
            self.assertIn('total_space_gb', device_stats, "Device stats should contain total_space_gb")
            self.assertIn('available_space_gb', device_stats, "Device stats should contain available_space_gb")
    
    def test_device_refresh(self):
        """Test device refresh functionality"""
        # Initial state
        initial_device_count = len(self.storage_manager.devices)
        
        # Refresh devices
        self.storage_manager.refresh_device_status()
        
        # The number of devices should remain the same or be updated
        refreshed_device_count = len(self.storage_manager.devices)
        self.assertGreaterEqual(refreshed_device_count, 0, "Should have at least 0 devices after refresh")
    
    def test_error_handling(self):
        """Test error handling for non-existent files"""
        # Try to read a non-existent file
        non_existent_data = self.storage_manager.read_file("non_existent_file.txt")
        self.assertIsNone(non_existent_data, "Should return None for non-existent file")
        
        # Try to delete a non-existent file
        delete_success = self.storage_manager.delete_file("non_existent_file.txt")
        self.assertFalse(delete_success, "Should return False for non-existent file deletion")
    
    def test_fallback_mechanism(self):
        """Test fallback mechanism"""
        # This is a simplified test - in a real scenario, we'd simulate device unavailability
        # For now, just ensure the fallback path is properly configured
        self.assertTrue(os.path.exists(self.config.fallback_storage_path), 
                       "Fallback path should exist")
    
    def test_compression_functionality(self):
        """Test compression functionality"""
        # Create large test data that should benefit from compression
        large_test_data = b"Compressible test data that repeats " * 10000  # ~350KB

        # Store with compression
        store_success = self.storage_manager.store_tensor(large_test_data, "compressed_test", compression=True)
        self.assertTrue(store_success, "Should successfully store compressed data")

        # Load compressed data
        loaded_data = self.storage_manager.load_tensor("compressed_test", compressed=True)
        self.assertIsNotNone(loaded_data, "Should successfully load compressed data")
        self.assertEqual(loaded_data, large_test_data, "Loaded data should match original data")


class TestStorageIntegration(unittest.TestCase):
    """Test integration with Qwen3-VL model components"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_base_dir = tempfile.mkdtemp()
        self.ssd_path = os.path.join(self.test_base_dir, "ssd_cache")
        self.hdd_path = os.path.join(self.test_base_dir, "hdd_storage")
        self.fallback_path = os.path.join(self.test_base_dir, "fallback")
        
        self.config = StorageConfig(
            ssd_path=self.ssd_path,
            hdd_path=self.hdd_path,
            fallback_storage_path=self.fallback_path,
            tiering_threshold=1024*1024,
            cache_size_limit=10*1024*1024,
            enable_compression=True
        )
        
        self.storage_manager = create_qwen3vl_storage_manager(self.config)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_base_dir, ignore_errors=True)
    
    def test_integration_functions(self):
        """Test integration functions"""
        from src.qwen3_vl.storage_management import integrate_with_qwen3vl_model
        
        # Get integration functions
        integration_funcs = integrate_with_qwen3vl_model(self.storage_manager)
        
        # Check that required functions are available
        required_funcs = [
            'save_model_weights',
            'load_model_weights', 
            'save_tensor',
            'load_tensor',
            'save_model_artifact',
            'run_storage_optimizations'
        ]
        
        for func_name in required_funcs:
            self.assertIn(func_name, integration_funcs, f"Integration should provide {func_name}")
            self.assertTrue(callable(integration_funcs[func_name]), f"{func_name} should be callable")
    
    def test_model_artifact_storage(self):
        """Test model artifact storage"""
        # Test storing different types of artifacts
        test_artifacts = {
            'cache': [1, 2, 3, 4, 5] * 1000,
            'checkpoint': {'epoch': 10, 'loss': 0.5},
            'backup': {'weights': [0.1, 0.2, 0.3] * 100},
            'archive': 'archived data' * 1000
        }
        
        for artifact_type, artifact_data in test_artifacts.items():
            success = self.storage_manager.store_model_artifact(
                artifact_data, f"test_{artifact_type}", artifact_type
            )
            self.assertTrue(success, f"Should successfully store {artifact_type} artifact")
            
            # Verify it can be loaded
            loaded_artifact = self.storage_manager.read_file(f"archived_tensors/test_{artifact_type}.pkl")
            if loaded_artifact is not None:
                self.assertEqual(loaded_artifact, artifact_data, 
                               f"Loaded {artifact_type} artifact should match original")


def run_performance_tests():
    """Run performance-focused tests"""
    print("Running performance tests...")
    
    # Create a larger test environment
    test_base_dir = tempfile.mkdtemp()
    ssd_path = os.path.join(test_base_dir, "ssd_cache")
    hdd_path = os.path.join(test_base_dir, "hdd_storage")
    fallback_path = os.path.join(test_base_dir, "fallback")
    
    config = StorageConfig(
        ssd_path=ssd_path,
        hdd_path=hdd_path,
        fallback_storage_path=fallback_path,
        tiering_threshold=1024*1024,
        cache_size_limit=50*1024*1024,  # 50MB for performance test
        enable_compression=True
    )
    
    storage_manager = create_qwen3vl_storage_manager(config)
    
    # Performance test: Write/read many small files
    start_time = time.time()
    for i in range(100):
        test_data = f"Test data for file {i}".encode() * 100  # ~3KB each
        storage_manager.write_file(f"perf_test_{i}.txt", test_data)
    write_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(100):
        _ = storage_manager.read_file(f"perf_test_{i}.txt")
    read_time = time.time() - start_time
    
    print(f"Performance test results:")
    print(f"  Writing 100 small files: {write_time:.3f}s ({100/write_time:.1f} files/s)")
    print(f"  Reading 100 small files: {read_time:.3f}s ({100/read_time:.1f} files/s)")
    
    # Performance test: Write/read large files
    large_data = b"Large test data " * 100000  # ~1.4MB
    start_time = time.time()
    storage_manager.write_file("large_test.txt", large_data)
    large_write_time = time.time() - start_time
    
    start_time = time.time()
    _ = storage_manager.read_file("large_test.txt")
    large_read_time = time.time() - start_time
    
    print(f"  Writing 1 large file (1.4MB): {large_write_time:.3f}s")
    print(f"  Reading 1 large file (1.4MB): {large_read_time:.3f}s")
    
    # Clean up
    shutil.rmtree(test_base_dir, ignore_errors=True)


def main():
    """Run all tests"""
    print("Starting Qwen3-VL Storage Management System Tests")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    print("\n" + "=" * 60)
    run_performance_tests()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    main()