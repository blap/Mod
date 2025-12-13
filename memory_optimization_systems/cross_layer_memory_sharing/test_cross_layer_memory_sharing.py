"""
Comprehensive test suite for the Cross-Layer Memory Sharing system.
Tests all aspects of the CrossLayerMemoryManager functionality.
"""

import unittest
import torch
import numpy as np
from memory_optimization_systems.cross_layer_memory_sharing.cross_layer_memory_sharing import (
    CrossLayerMemoryManager, TensorType, TensorMetadata, ReferenceCounter
)


class TestCrossLayerMemorySharing(unittest.TestCase):
    """Test suite for CrossLayerMemoryManager functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.manager = CrossLayerMemoryManager(
            max_shared_tensors=100,
            compatibility_threshold=0.95,
            hardware_target="intel_i5_nvidia_sm61"
        )
    
    def test_register_tensor_basic(self):
        """Test basic tensor registration functionality."""
        tensor = torch.randn(10, 20)
        layer_id = 1
        tensor_type = TensorType.ACTIVATION
        
        tensor_id = self.manager.register_tensor(tensor, layer_id, tensor_type)
        
        # Check that tensor was registered
        self.assertIsNotNone(tensor_id)
        self.assertIn(tensor_id, self.manager.tensor_registry)
        
        # Check that layer-tensor mapping was updated
        self.assertIn(layer_id, self.manager.layer_tensor_map)
        self.assertIn(tensor_id, self.manager.layer_tensor_map[layer_id])
    
    def test_tensor_compatibility_check(self):
        """Test tensor compatibility verification."""
        meta1 = TensorMetadata(
            shape=(10, 20),
            dtype=torch.float32,
            tensor_type=TensorType.ACTIVATION,
            layer_id=1,
            device='cpu'
        )
        
        meta2 = TensorMetadata(
            shape=(10, 20),
            dtype=torch.float32,
            tensor_type=TensorType.ACTIVATION,
            layer_id=2,
            device='cpu'
        )
        
        # Compatible tensors should return True
        self.assertTrue(self.manager._is_compatible_for_sharing(meta1, meta2))
        
        # Incompatible due to different shapes
        meta3 = TensorMetadata(
            shape=(5, 20),
            dtype=torch.float32,
            tensor_type=TensorType.ACTIVATION,
            layer_id=2,
            device='cpu'
        )
        self.assertFalse(self.manager._is_compatible_for_sharing(meta1, meta3))
        
        # Incompatible due to different dtypes
        meta4 = TensorMetadata(
            shape=(10, 20),
            dtype=torch.float64,
            tensor_type=TensorType.ACTIVATION,
            layer_id=2,
            device='cpu'
        )
        self.assertFalse(self.manager._is_compatible_for_sharing(meta1, meta4))
        
        # Incompatible due to different devices
        meta5 = TensorMetadata(
            shape=(10, 20),
            dtype=torch.float32,
            tensor_type=TensorType.ACTIVATION,
            layer_id=2,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        if torch.cuda.is_available():
            self.assertFalse(self.manager._is_compatible_for_sharing(meta1, meta5))
        else:
            # If CUDA is not available, both will be on CPU
            self.assertTrue(self.manager._is_compatible_for_sharing(meta1, meta5))
    
    def test_find_sharing_opportunities(self):
        """Test finding sharing opportunities for tensors."""
        # Register a tensor
        original_tensor = torch.randn(5, 10)
        layer_id = 1
        tensor_type = TensorType.KEY
        original_id = self.manager.register_tensor(original_tensor, layer_id, tensor_type)
        
        # Try to find sharing opportunity with a compatible tensor
        compatible_tensor = torch.randn(5, 10)  # Same shape
        opportunities = self.manager.find_sharing_opportunities(compatible_tensor, 2, TensorType.KEY)
        
        # Should find the original tensor as an opportunity
        self.assertIn(original_id, opportunities)
        
        # Try with incompatible tensor (different shape)
        incompatible_tensor = torch.randn(3, 7)  # Different shape
        opportunities = self.manager.find_sharing_opportunities(incompatible_tensor, 3, TensorType.KEY)
        
        # Should not find any opportunities
        self.assertEqual(len(opportunities), 0)
    
    def test_acquire_and_release_tensor(self):
        """Test acquiring and releasing tensor references."""
        tensor = torch.randn(5, 10)
        layer_id = 1
        tensor_type = TensorType.VALUE

        tensor_id = self.manager.register_tensor(tensor, layer_id, tensor_type)

        # Initially, reference count should be 1
        _, _, ref_counter = self.manager.tensor_registry[tensor_id]
        self.assertEqual(ref_counter.get_count(), 1)

        # Acquire tensor from another layer
        acquired_tensor = self.manager.acquire_tensor(tensor_id, 2)
        self.assertIsNotNone(acquired_tensor)

        # Reference count should now be 2
        self.assertEqual(ref_counter.get_count(), 2)

        # Release tensor from second layer
        success = self.manager.release_tensor(tensor_id, 2)
        self.assertTrue(success)

        # Reference count should be back to 1
        self.assertEqual(ref_counter.get_count(), 1)

        # Release from original layer
        success = self.manager.release_tensor(tensor_id, 1)
        self.assertTrue(success)

        # Tensor should still be in registry but with reference count 0
        # It will be removed during cleanup
        self.assertIn(tensor_id, self.manager.tensor_registry)
        _, _, ref_counter = self.manager.tensor_registry[tensor_id]
        self.assertEqual(ref_counter.get_count(), 0)

        # Call cleanup to remove tensors with zero reference count
        self.manager.cleanup_unused_tensors()

        # Now tensor should be removed from registry after cleanup
        self.assertNotIn(tensor_id, self.manager.tensor_registry)
    
    def test_tensor_type_specific_sharing(self):
        """Test tensor-type-specific sharing strategies."""
        # Register key tensors in layer 1
        key_tensor1 = torch.randn(4, 8, 16)  # [batch, seq, embed_dim]
        key_id1 = self.manager.register_tensor(key_tensor1, 1, TensorType.KEY)
        
        # Register compatible key tensor in layer 2
        key_tensor2 = torch.randn(4, 8, 16)
        key_id2 = self.manager.register_tensor(key_tensor2, 2, TensorType.KEY)
        
        # Attempt to share key tensors
        success = self.manager.share_tensor(1, 2, TensorType.KEY)
        self.assertTrue(success)
        
        # Check that sharing was recorded in metadata
        _, metadata1, _ = self.manager.tensor_registry[key_id1]
        _, metadata2, _ = self.manager.tensor_registry[key_id2]
        
        self.assertIn(2, metadata1.shared_with)
        self.assertIn(1, metadata2.shared_with)
    
    def test_different_tensor_types_not_shared_inappropriately(self):
        """Test that different tensor types are not inappropriately shared."""
        # Register a key tensor
        key_tensor = torch.randn(4, 8, 16)
        key_id = self.manager.register_tensor(key_tensor, 1, TensorType.KEY)
        
        # Register a value tensor with same shape
        value_tensor = torch.randn(4, 8, 16)
        value_id = self.manager.register_tensor(value_tensor, 1, TensorType.VALUE)
        
        # Key and value tensors should not be considered for sharing despite same shape
        opportunities = self.manager.find_sharing_opportunities(key_tensor, 2, TensorType.VALUE)
        self.assertNotIn(key_id, opportunities)
        
        opportunities = self.manager.find_sharing_opportunities(value_tensor, 2, TensorType.KEY)
        self.assertNotIn(value_id, opportunities)
    
    def test_gradient_tensor_sharing_restriction(self):
        """Test that gradient tensor sharing is appropriately restricted."""
        # Register gradient tensors in different layers
        grad_tensor1 = torch.randn(10, 20)
        grad_tensor2 = torch.randn(10, 20)  # Same shape to test sharing logic
        
        id1 = self.manager.register_tensor(grad_tensor1, 1, TensorType.GRADIENT)
        id2 = self.manager.register_tensor(grad_tensor2, 2, TensorType.GRADIENT)
        
        # Attempt to share gradient tensors - should fail
        success = self.manager._share_gradient_tensors([id1], 1, 2, 0.9)
        self.assertFalse(success)
    
    def test_memory_usage_statistics(self):
        """Test memory usage statistics reporting."""
        # Register several tensors
        for i in range(5):
            tensor = torch.randn(10, 20)
            self.manager.register_tensor(tensor, i, TensorType.ACTIVATION)
        
        # Acquire some tensors from other layers to create sharing
        tensor_ids = list(self.manager.tensor_registry.keys())
        if len(tensor_ids) >= 2:
            self.manager.acquire_tensor(tensor_ids[0], 5)  # Layer 5 acquires tensor from layer 0
            self.manager.acquire_tensor(tensor_ids[1], 5)  # Layer 5 acquires tensor from layer 1
        
        stats = self.manager.get_memory_usage_stats()
        
        # Verify that statistics are properly calculated
        self.assertIsInstance(stats, dict)
        self.assertIn('total_tensors', stats)
        self.assertIn('total_layers', stats)
        self.assertIn('shared_tensors', stats)
        self.assertIn('total_references', stats)
        self.assertGreaterEqual(stats['total_tensors'], 5)
        self.assertGreaterEqual(stats['total_layers'], 6)  # 5 original + 1 new layer
    
    def test_hardware_optimization_setup(self):
        """Test hardware-specific optimization setup."""
        # Check that hardware-specific attributes were set
        self.assertTrue(hasattr(self.manager, '_cpu_cache_size'))
        self.assertTrue(hasattr(self.manager, '_gpu_memory_alignment'))
        self.assertTrue(hasattr(self.manager, '_max_shared_memory_per_block'))
        self.assertTrue(hasattr(self.manager, '_allocation_strategy'))
        
        # Verify values are reasonable
        self.assertEqual(self.manager._cpu_cache_size, 6 * 1024 * 1024)  # 6MB
        self.assertEqual(self.manager._gpu_memory_alignment, 256)
        self.assertEqual(self.manager._max_shared_memory_per_block, 48 * 1024)  # 48KB
        self.assertEqual(self.manager._allocation_strategy['small_tensors'], 'cpu')
    
    def test_hardware_tensor_optimization(self):
        """Test hardware-specific tensor optimization."""
        tensor = torch.randn(10, 20)
        optimized = self.manager.optimize_for_hardware(tensor, TensorType.ACTIVATION)
        
        # Result should be a tensor (might be the same or a copy)
        self.assertIsInstance(optimized, torch.Tensor)
        self.assertEqual(optimized.shape, tensor.shape)
    
    def test_cleanup_unused_tensors(self):
        """Test cleanup of tensors with zero reference counts."""
        tensor = torch.randn(5, 10)
        layer_id = 1
        tensor_type = TensorType.ACTIVATION
        
        tensor_id = self.manager.register_tensor(tensor, layer_id, tensor_type)
        
        # Release the tensor to bring reference count to 0
        self.manager.release_tensor(tensor_id, layer_id)
        
        # At this point, tensor still exists in registry but has 0 refs
        self.assertIn(tensor_id, self.manager.tensor_registry)
        
        # Cleanup should remove tensors with zero references
        self.manager.cleanup_unused_tensors()
        
        # Tensor should now be removed from registry
        self.assertNotIn(tensor_id, self.manager.tensor_registry)
    
    def test_thread_safety_simulation(self):
        """Simulate concurrent access to test thread safety."""
        import threading
        import time

        results = []
        results_lock = threading.Lock()  # Thread-safe list access

        def register_tensors(thread_id):
            for i in range(10):
                # Create tensors with different shapes to ensure unique IDs
                tensor = torch.randn(5 + thread_id, 5 + i)  # Different shape for each thread and iteration
                tensor_id = self.manager.register_tensor(tensor, thread_id, TensorType.ACTIVATION)
                with results_lock:
                    results.append((thread_id, tensor_id))
                time.sleep(0.001)  # Small delay to increase chance of race conditions

        # Create and start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_tensors, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all registrations succeeded
        self.assertEqual(len(results), 50)  # 5 threads * 10 tensors each

        # Check that registry contains expected number of tensors (should be 50 if all unique)
        # Note: Due to the nature of the sharing system, some tensors might still be shared if they
        # have compatible properties, but with our unique shapes this should be minimal
        self.assertGreaterEqual(len(self.manager.tensor_registry), 40)  # Allow some sharing but most should be unique
    
    def test_get_tensor_by_id(self):
        """Test retrieving tensor by ID without affecting reference count."""
        tensor = torch.randn(3, 4)
        layer_id = 1
        tensor_type = TensorType.INTERMEDIATE
        
        tensor_id = self.manager.register_tensor(tensor, layer_id, tensor_type)
        
        # Get tensor by ID (should not affect reference count)
        retrieved_result = self.manager.get_tensor_by_id(tensor_id)
        self.assertIsNotNone(retrieved_result)
        
        retrieved_tensor, retrieved_metadata = retrieved_result
        self.assertIsInstance(retrieved_tensor, torch.Tensor)
        self.assertIsInstance(retrieved_metadata, TensorMetadata)
        self.assertEqual(retrieved_tensor.shape, tensor.shape)
        
        # Reference count should remain unchanged
        _, _, ref_counter = self.manager.tensor_registry[tensor_id]
        self.assertEqual(ref_counter.get_count(), 1)
    
    def test_large_scale_sharing_scenario(self):
        """Test a more complex sharing scenario with multiple layers and tensor types."""
        # Create tensors for multiple layers
        layers = [0, 1, 2, 3, 4]
        tensor_types = [TensorType.KEY, TensorType.VALUE, TensorType.ACTIVATION]
        
        # Register various tensors across different layers
        for layer in layers:
            for tensor_type in tensor_types:
                tensor = torch.randn(8, 16)  # Consistent size for sharing opportunities
                self.manager.register_tensor(tensor, layer, tensor_type)
        
        # Attempt sharing between adjacent layers
        for i in range(len(layers)-1):
            for tensor_type in tensor_types:
                success = self.manager.share_tensor(layers[i], layers[i+1], tensor_type)
                # Note: Not checking for success since sharing might not always be possible
                # depending on internal logic, but operation should not crash
        
        # Verify the system remains consistent
        stats = self.manager.get_memory_usage_stats()
        self.assertGreaterEqual(stats['total_tensors'], 10)  # Should have multiple tensors
        self.assertGreaterEqual(stats['total_layers'], 5)    # Should have 5 layers


class TestTensorMetadata(unittest.TestCase):
    """Test tensor metadata functionality."""
    
    def test_metadata_creation(self):
        """Test creation of TensorMetadata."""
        metadata = TensorMetadata(
            shape=(10, 20),
            dtype=torch.float32,
            tensor_type=TensorType.ACTIVATION,
            layer_id=1,
            device='cpu'
        )
        
        self.assertEqual(metadata.shape, (10, 20))
        self.assertEqual(metadata.dtype, torch.float32)
        self.assertEqual(metadata.tensor_type, TensorType.ACTIVATION)
        self.assertEqual(metadata.layer_id, 1)
        self.assertEqual(metadata.device, 'cpu')
        self.assertEqual(metadata.reference_count, 0)
        self.assertIsInstance(metadata.shared_with, list)
        self.assertEqual(len(metadata.shared_with), 0)


class TestReferenceCounter(unittest.TestCase):
    """Test reference counter functionality."""
    
    def test_reference_counting(self):
        """Test incrementing and decrementing reference counts."""
        counter = ReferenceCounter(0)
        
        # Initial count should be 0
        self.assertEqual(counter.get_count(), 0)
        
        # Increment should increase count
        new_count = counter.increment()
        self.assertEqual(new_count, 1)
        self.assertEqual(counter.get_count(), 1)
        
        # Another increment
        new_count = counter.increment()
        self.assertEqual(new_count, 2)
        self.assertEqual(counter.get_count(), 2)
        
        # Decrement should decrease count
        new_count = counter.decrement()
        self.assertEqual(new_count, 1)
        self.assertEqual(counter.get_count(), 1)
        
        # Final decrement to 0
        new_count = counter.decrement()
        self.assertEqual(new_count, 0)
        self.assertEqual(counter.get_count(), 0)
    
    def test_reference_count_error_handling(self):
        """Test error handling when reference count goes below zero."""
        counter = ReferenceCounter(1)
        
        # Decrement once to reach 0
        counter.decrement()
        
        # Further decrements should raise an error
        with self.assertRaises(RuntimeError):
            counter.decrement()


def run_tests():
    """Run all tests in the test suite."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()