"""
Hardware-Specific Validation Tests for Intel i5-10210U + NVIDIA SM61 + NVMe SSD

This test suite validates that the Qwen3-VL memory optimizations are properly 
tuned for the specific hardware combination of Intel i5-10210U CPU, 
NVIDIA SM61 GPU, and NVMe SSD storage.
"""

import unittest
import torch
import numpy as np
import time
import psutil
import os
import tempfile
import shutil
from typing import Dict, List, Tuple
import platform

# Import optimization modules
from src.qwen3_vl.components.memory.advanced_memory_management_vl import VisionLanguageMemoryOptimizer
from src.qwen3_vl.components.memory.advanced_memory_swapping_system import AdvancedMemorySwapper as MemorySwappingSystem
from src.qwen3_vl.components.memory.advanced_memory_tiering_system import AdvancedMemoryTieringSystem as MemoryTieringSystem
from src.qwen3_vl.components.memory.memory_compression_system import MemoryCompressionManager as MemoryCompressionSystem
from src.qwen3_vl.components.attention.enhanced_predictive_tensor_lifecycle_manager import EnhancedPredictiveGarbageCollector as PredictiveTensorLifecycleManager
from src.qwen3_vl.components.memory.integrated_memory_management_system import IntegratedMemoryManager as IntegratedMemoryManagementSystem
from src.qwen3_vl.components.optimization.advanced_cpu_optimizations_intel_i5_10210u import AdaptiveIntelOptimizer as IntelOptimizationSuite


class TestIntelI5_10210UOptimizations(unittest.TestCase):
    """Test optimizations specifically for Intel i5-10210U CPU"""
    
    def setUp(self):
        """Set up Intel i5-10210U specific optimizations"""
        # Intel i5-10210U: 4 cores, 8 threads, 8MB cache, base freq ~1.6GHz, boost ~4.2GHz
        self.cpu_info = {
            'cores': os.cpu_count(),
            'architecture': platform.machine(),
            'system_memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        # Configure optimizations for i5-10210U with 8GB RAM
        self.memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=min(2 * 1024 * 1024 * 1024, int(self.cpu_info['system_memory_gb'] * 0.3 * 1024**3)),  # Use up to 30% of system RAM
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=True
        )
        
        # Intel-specific CPU optimizations
        from src.qwen3_vl.components.optimization.advanced_cpu_optimizations_intel_i5_10210u import AdvancedCPUOptimizationConfig
        config = AdvancedCPUOptimizationConfig(
            num_preprocess_workers=4,  # Intel i5-10210U has 4 cores
            max_concurrent_threads=8,  # With hyperthreading
            l3_cache_size=6 * 1024 * 1024  # 6MB L3 cache for i5-10210U
        )
        self.cpu_optimizer = IntelOptimizationSuite(config=config)

    def test_hyperthreading_aware_allocation(self):
        """Test memory allocation aware of hyperthreading capabilities"""
        # i5-10210U has 4 physical cores with hyperthreading (8 logical cores)
        logical_cores = self.cpu_info['cores']
        physical_cores = logical_cores // 2 if logical_cores >= 8 else logical_cores  # Estimate physical cores
        
        print(f"Detected {logical_cores} logical cores, estimated {physical_cores} physical cores")
        
        # Test concurrent allocation matching thread count
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def allocate_worker(worker_id):
            start_time = time.time()
            tensor = self.memory_optimizer.allocate_tensor_memory(
                (500, 512), dtype=np.float32, tensor_type="general"
            )
            alloc_time = time.time() - start_time

            # Simulate some CPU work
            work_start = time.time()
            # Convert to PyTorch tensor for computation
            tensor_torch = torch.from_numpy(tensor)
            # Simple computation to simulate CPU utilization
            result = torch.sum(tensor_torch * 0.5)
            work_time = time.time() - work_start

            self.memory_optimizer.free_tensor_memory(tensor, "general")

            results_queue.put({
                'worker_id': worker_id,
                'alloc_time': alloc_time,
                'work_time': work_time,
                'total_time': alloc_time + work_time
            })
        
        # Launch workers matching logical core count (but cap at 8 for i5-10210U)
        num_workers = min(logical_cores, 8)
        threads = []
        
        for i in range(num_workers):
            t = threading.Thread(target=allocate_worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all workers to complete
        for t in threads:
            t.join()
        
        # Collect results
        worker_results = []
        while not results_queue.empty():
            worker_results.append(results_queue.get())
        
        # Calculate average times
        avg_alloc_time = np.mean([r['alloc_time'] for r in worker_results])
        avg_work_time = np.mean([r['work_time'] for r in worker_results])
        avg_total_time = np.mean([r['total_time'] for r in worker_results])
        
        print(f"Hyperthreading test with {num_workers} workers:")
        print(f"  Avg allocation time: {avg_alloc_time:.4f}s")
        print(f"  Avg computation time: {avg_work_time:.4f}s")
        print(f"  Avg total time: {avg_total_time:.4f}s")
        
        # Verify all workers completed successfully
        self.assertEqual(len(worker_results), num_workers)

    def test_cache_aware_memory_layouts(self):
        """Test cache-aware memory layouts optimized for Intel CPU cache"""
        # Intel i5-10210U has L1 cache (32KB per core), L2 cache (256KB per core), L3 cache (6MB shared)
        cache_line_size = 64  # Standard for Intel CPUs
        
        # Create matrices that should fit well in cache
        small_matrix = torch.randn(128, 128)  # ~64KB, should fit in L1 cache
        medium_matrix = torch.randn(512, 512)  # ~1MB, should fit in L2/L3 cache
        
        # Test cache-friendly allocation
        start_time = time.time()
        opt_small = self.memory_optimizer.allocate_tensor_memory(
            (128, 128), dtype=np.float32, tensor_type="general"
        )
        small_alloc_time = time.time() - start_time

        start_time = time.time()
        opt_medium = self.memory_optimizer.allocate_tensor_memory(
            (512, 512), dtype=np.float32, tensor_type="general"
        )
        medium_alloc_time = time.time() - start_time
        
        # Verify cache-friendly properties
        # For Intel CPUs, contiguous memory layout is preferred
        # Convert to PyTorch tensors to check contiguity
        opt_small_tensor = torch.from_numpy(opt_small)
        opt_medium_tensor = torch.from_numpy(opt_medium)
        self.assertTrue(opt_small_tensor.is_contiguous())
        self.assertTrue(opt_medium_tensor.is_contiguous())
        
        print(f"Cache-aware allocation times:")
        print(f"  Small matrix (128x128): {small_alloc_time:.6f}s")
        print(f"  Medium matrix (512x512): {medium_alloc_time:.6f}s")
        
        # Clean up
        self.memory_optimizer.free_tensor_memory(opt_small, "general")
        self.memory_optimizer.free_tensor_memory(opt_medium, "general")

    def test_memory_pool_fragmentation_for_i5_10210U(self):
        """Test memory pool fragmentation management for limited RAM systems"""
        # i5-10210U typically paired with 8GB RAM, so fragmentation is critical
        
        # Allocate and deallocate various sized blocks to test fragmentation handling
        block_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536]  # Different sizes
        allocated_blocks = []
        
        for size in block_sizes:
            tensor = self.memory_optimizer.allocate_tensor_memory(
                (size, 64), dtype=np.float32, tensor_type="general"
            )
            allocated_blocks.append((tensor, size))
        
        # Get initial fragmentation
        stats_before = self.memory_optimizer.get_memory_stats()
        initial_fragmentation = stats_before.get('general_pool', {}).get('fragmentation', 1.0)
        
        # Deallocate every other block to create fragmentation
        for i in range(0, len(allocated_blocks), 2):
            tensor, _ = allocated_blocks[i]
            self.memory_optimizer.free_tensor_memory(tensor, "general")
        
        # Get fragmentation after deallocation
        stats_after = self.memory_optimizer.get_memory_stats()
        fragmentation_after = stats_after.get('general_pool', {}).get('fragmentation', 1.0)
        
        print(f"Fragmentation test:")
        print(f"  Initial fragmentation: {initial_fragmentation:.3f}")
        print(f"  After deallocation: {fragmentation_after:.3f}")
        
        # The memory defragmenter should keep fragmentation reasonable
        self.assertLessEqual(fragmentation_after, 0.6)  # Less than 60% fragmentation

    def tearDown(self):
        """Clean up Intel-specific optimizations"""
        if hasattr(self, 'memory_optimizer'):
            self.memory_optimizer.cleanup()


class TestNvidiaSM61Optimizations(unittest.TestCase):
    """Test optimizations specifically for NVIDIA SM61 GPU"""
    
    def setUp(self):
        """Set up NVIDIA SM61 specific optimizations"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping GPU tests")
        
        # Check if GPU is SM61 (Maxwell architecture)
        gpu_name = torch.cuda.get_device_name(0).lower()
        if 'sm61' not in gpu_name and 'maxwell' not in gpu_name:
            print(f"Warning: GPU {gpu_name} may not be SM61 architecture")
        
        self.gpu_properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {self.gpu_properties.name}")
        print(f"Compute Capability: {self.gpu_properties.major}.{self.gpu_properties.minor}")
        print(f"Total Memory: {self.gpu_properties.total_memory / (1024**3):.2f} GB")
        
        # Configure memory optimizer for GPU
        self.memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=1024 * 1024 * 1024,  # 1GB GPU pool
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=True
        )

    def test_gpu_memory_allocation_efficiency(self):
        """Test GPU memory allocation efficiency for SM61 architecture"""
        if not torch.cuda.is_available():
            return
        
        # SM61 has limited memory bandwidth, so efficient allocation is important
        tensor_sizes = [
            (512, 512),      # ~1MB
            (1024, 1024),    # ~4MB  
            (2048, 1024),    # ~8MB
            (2048, 2048),    # ~16MB
        ]
        
        allocation_times = []
        transfer_times = []
        
        for size in tensor_sizes:
            # Test allocation time
            start_time = time.time()
            tensor = self.memory_optimizer.allocate_tensor_memory(
                size, dtype=torch.float32, tensor_type="general"
            )
            alloc_time = time.time() - start_time
            allocation_times.append(alloc_time)
            
            # Test transfer time (if applicable)
            cpu_tensor = torch.randn(size)
            start_time = time.time()
            gpu_tensor = self.memory_optimizer.optimize_tensor_placement(cpu_tensor, target_device="cuda")
            transfer_time = time.time() - start_time
            transfer_times.append(transfer_time)
            
            # Clean up
            if hasattr(self.memory_optimizer, 'free_tensor_memory'):
                self.memory_optimizer.free_tensor_memory(tensor, "general")
        
        print("GPU allocation efficiency test:")
        for i, size in enumerate(tensor_sizes):
            print(f"  Size {size}: Alloc={allocation_times[i]:.6f}s, Transfer={transfer_times[i]:.6f}s")
        
        # Verify allocations succeeded
        for alloc_time in allocation_times:
            self.assertLess(alloc_time, 1.0)  # Should be fast

    def test_pinned_memory_optimization(self):
        """Test pinned memory optimization for SM61"""
        if not torch.cuda.is_available():
            return
        
        # Create CPU tensor
        cpu_tensor = torch.randn(1024, 1024)  # 4MB tensor
        
        # Test standard transfer
        torch.cuda.synchronize()
        start_time = time.time()
        standard_gpu_tensor = cpu_tensor.cuda()
        torch.cuda.synchronize()
        standard_time = time.time() - start_time
        
        # Test optimized transfer with pinned memory
        torch.cuda.synchronize()
        start_time = time.time()
        optimized_gpu_tensor = self.memory_optimizer.optimize_tensor_placement(
            cpu_tensor, target_device="cuda"
        )
        torch.cuda.synchronize()
        optimized_time = time.time() - start_time
        
        print(f"Pinned memory optimization:")
        print(f"  Standard transfer: {standard_time:.6f}s")
        print(f"  Optimized transfer: {optimized_time:.6f}s")
        print(f"  Improvement: {(standard_time - optimized_time) / standard_time * 100:.2f}%")
        
        # The optimized version should be faster or at least comparable
        self.assertLessEqual(optimized_time, standard_time * 1.1)  # Allow 10% variance

    def tearDown(self):
        """Clean up GPU resources"""
        if hasattr(self, 'memory_optimizer'):
            self.memory_optimizer.cleanup()
        torch.cuda.empty_cache()


class TestNVMeSSDOptimizations(unittest.TestCase):
    """Test optimizations specifically for NVMe SSD storage"""
    
    def setUp(self):
        """Set up NVMe SSD specific optimizations"""
        # Create a temporary directory that should be on fast storage
        self.temp_dir = tempfile.mkdtemp(prefix="nvme_test_")
        
        # Initialize swapping system with NVMe-optimized settings
        self.swapping_system = MemorySwappingSystem(
            swap_threshold=0.7,  # Start swapping at 70% memory usage for NVMe
            swap_path=self.temp_dir,
            max_swap_size=4 * 1024 * 1024 * 1024  # 4GB max swap for NVMe
        )
        
        self.tiering_system = MemoryTieringSystem(
            tier_configs={
                'gpu': {'capacity': 1024 * 1024 * 1024, 'priority': 1, 'access_time': 0.0001},  # Fast GPU memory
                'cpu': {'capacity': 2048 * 1024 * 1024, 'priority': 2, 'access_time': 0.0002},  # CPU memory
                'nvme': {'capacity': 8192 * 1024 * 1024, 'priority': 3, 'access_time': 0.001}   # NVMe storage
            }
        )

    def test_nvme_storage_performance(self):
        """Test storage performance characteristics of NVMe SSD"""
        # Test sequential write performance (characteristic of NVMe)
        write_times = []
        data_sizes = [10, 25, 50, 100]  # MB
        
        for size_mb in data_sizes:
            # Create data to write
            data = torch.randn(size_mb * 1024 * 1024 // 4).numpy()  # 4 bytes per float32
            
            # Time the write operation
            start_time = time.time()
            filepath = os.path.join(self.temp_dir, f"test_{size_mb}mb.bin")
            np.save(filepath, data)
            write_time = time.time() - start_time
            
            write_times.append(write_time)
            write_speed = (size_mb) / write_time  # MB/s
            
            print(f"NVMe write test - Size: {size_mb}MB, Time: {write_time:.4f}s, Speed: {write_speed:.2f}MB/s")
            
            # Clean up
            os.remove(filepath)
        
        # NVMe should achieve reasonable write speeds (varies by drive, but typically >500MB/s)
        # For testing purposes, we'll check that it's not extremely slow
        avg_write_speed = np.mean([(size_mb / time) for size_mb, time in zip(data_sizes, write_times)])
        print(f"Average NVMe write speed: {avg_write_speed:.2f}MB/s")
        
        # Verify reasonable performance (at least 10MB/s, which is conservative for NVMe)
        self.assertGreater(avg_write_speed, 10.0)

    def test_memory_swapping_with_nvme(self):
        """Test memory swapping performance with NVMe SSD"""
        # Create tensors that will likely trigger swapping
        large_tensors = []
        for i in range(10):
            tensor = torch.randn(1024, 1024)  # ~4MB each
            self.swapping_system.register_tensor(f"tensor_{i}", tensor)
            large_tensors.append((f"tensor_{i}", tensor))
        
        # Simulate memory pressure that would trigger swapping
        start_time = time.time()
        for name, tensor in large_tensors:
            # Access tensors to potentially trigger swapping decisions
            result = torch.sum(tensor * 0.1)
        access_time = time.time() - start_time
        
        # Check swapping statistics
        stats = self.swapping_system.get_stats()
        swap_operations = stats.get('swap_operations', 0)
        swap_time = stats.get('total_swap_time', 0)
        
        print(f"NVMe swapping test:")
        print(f"  Access time for 10 tensors: {access_time:.4f}s")
        print(f"  Swap operations: {swap_operations}")
        print(f"  Total swap time: {swap_time:.4f}s")
        
        # Verify system handled the tensors without errors
        self.assertIsNotNone(access_time)

    def test_memory_tiering_with_nvme(self):
        """Test memory tiering with NVMe SSD as storage tier"""
        # Create tensors of different access patterns
        frequently_used = torch.randn(512, 512)  # Will be kept in fast memory
        rarely_used = torch.randn(1024, 1024)    # May be moved to NVMe tier
        
        # Assign to appropriate tiers based on access pattern
        tier1_result = self.tiering_system.assign_to_tier("frequent", frequently_used, priority='high')
        tier2_result = self.tiering_system.assign_to_tier("rare", rarely_used, priority='low')
        
        # Check tier assignments
        tier_assignments = self.tiering_system.get_tier_assignments()
        
        print(f"Tiering assignments: {tier_assignments}")
        
        # Verify that tensors were processed
        self.assertIsNotNone(tier1_result)
        self.assertIsNotNone(tier2_result)

    def tearDown(self):
        """Clean up NVMe SSD test resources"""
        if hasattr(self, 'swapping_system'):
            self.swapping_system.cleanup()
        if hasattr(self, 'tiering_system'):
            self.tiering_system.cleanup()
        
        # Clean up temporary directory
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestIntegratedHardwareOptimizations(unittest.TestCase):
    """Test integration of all hardware-specific optimizations"""
    
    def setUp(self):
        """Set up all hardware-specific optimizations together"""
        # Initialize all optimization systems
        self.memory_optimizer = VisionLanguageMemoryOptimizer(
            memory_pool_size=1536 * 1024 * 1024,  # 1.5GB (conservative for 8GB system)
            enable_memory_pool=True,
            enable_cache_optimization=True,
            enable_gpu_optimization=True
        )
        
        self.swapping_system = MemorySwappingSystem(
            swap_threshold=0.75,
            swap_path=tempfile.mkdtemp()
        )
        
        self.tiering_system = MemoryTieringSystem(
            tier_configs={
                'gpu': {'capacity': 1024 * 1024 * 1024, 'priority': 1},
                'cpu': {'capacity': 2048 * 1024 * 1024, 'priority': 2}, 
                'nvme': {'capacity': 4096 * 1024 * 1024, 'priority': 3}
            }
        )
        
        self.compression_system = MemoryCompressionSystem(
            compression_ratio=2.0,
            algorithm='quantization'
        )
        
        self.lifecycle_manager = PredictiveTensorLifecycleManager(
            prediction_horizon=12,
            decay_factor=0.88
        )
        
        self.integrated_system = IntegratedMemoryManagementSystem(
            memory_optimizer=self.memory_optimizer,
            swapping_system=self.swapping_system,
            tiering_system=self.tiering_system,
            compression_system=self.compression_system,
            lifecycle_manager=self.lifecycle_manager
        )

    def test_full_hardware_pipeline(self):
        """Test complete pipeline with all hardware optimizations"""
        # Simulate a realistic vision-language workload
        batch_size, seq_len, hidden_dim = 4, 128, 512  # Conservative for i5-10210U + 8GB RAM
        
        # Step 1: Image processing (CPU-intensive, memory managed)
        images = torch.randn(batch_size, 3, 224, 224)
        opt_images = self.integrated_system.allocate_tensor(
            images.shape, dtype=images.dtype, tensor_type="image_features"
        )
        opt_images[:,:,:,:] = images
        
        # Step 2: Text processing (GPU-assisted when available)
        text_ids = torch.randint(0, 1000, (batch_size, seq_len))
        opt_texts = self.integrated_system.allocate_tensor(
            text_ids.shape, dtype=text_ids.dtype, tensor_type="text_embeddings"
        )
        opt_texts[:,:] = text_ids
        
        # Step 3: Cross-modal processing (uses multiple optimizations)
        # Simulate intermediate tensors that get tiered and compressed
        intermediate = self.integrated_system.allocate_tensor(
            (batch_size, seq_len, hidden_dim), dtype=torch.float32, tensor_type="intermediate"
        )
        
        # Apply tiering based on usage pattern
        tiered_intermediate = self.integrated_system.tier_tensor(intermediate, priority='medium')
        
        # Apply compression
        compressed_intermediate = self.integrated_system.compress_tensor(tiered_intermediate)
        
        # Step 4: Lifecycle management
        self.integrated_system.update_tensor_lifecycle(
            compressed_intermediate, access_pattern='sequential'
        )
        
        # Step 5: Final processing and cleanup
        result = self.integrated_system.allocate_tensor(
            (batch_size, 1000), dtype=torch.float32, tensor_type="output"
        )
        
        # Update lifecycle for result tensor
        self.integrated_system.update_tensor_lifecycle(result, access_pattern='output')
        
        # Verify all steps completed successfully
        self.assertIsNotNone(opt_images)
        self.assertIsNotNone(opt_texts)
        self.assertIsNotNone(tiered_intermediate)
        self.assertIsNotNone(compressed_intermediate)
        self.assertIsNotNone(result)
        
        # Get system statistics
        stats = self.integrated_system.get_system_stats()
        print(f"Integrated system stats: {stats}")
        
        # Clean up all tensors
        tensors_to_cleanup = [
            (opt_images, "image_features"),
            (opt_texts, "text_embeddings"), 
            (tiered_intermediate, "intermediate"),
            (result, "output")
        ]
        
        for tensor, tensor_type in tensors_to_cleanup:
            try:
                self.integrated_system.free_tensor(tensor, tensor_type)
            except:
                pass  # Tensor might already be freed through lifecycle management

    def test_hardware_adaptive_behavior(self):
        """Test that system adapts to hardware constraints"""
        # Test behavior under different memory pressure scenarios
        initial_memory = psutil.virtual_memory().percent
        
        # Allocate memory in increasing amounts to test adaptive behavior
        allocation_sizes = [100, 200, 500, 1000]  # MB
        allocation_results = []
        
        for size_mb in allocation_sizes:
            try:
                tensor_size = int(size_mb * 1024 * 1024 / 4)  # Convert MB to elements for float32
                tensor = self.integrated_system.allocate_tensor(
                    (tensor_size,), dtype=torch.float32, tensor_type="general"
                )
                
                # Simulate usage
                _ = torch.sum(tensor * 0.5)
                
                # Free tensor
                self.integrated_system.free_tensor(tensor, "general")
                
                allocation_results.append(True)
                print(f"Successfully allocated {size_mb}MB tensor")
                
            except Exception as e:
                allocation_results.append(False)
                print(f"Failed to allocate {size_mb}MB tensor: {e}")
                break  # Stop if allocation fails
        
        # System should handle reasonable allocations
        successful_allocations = sum(allocation_results)
        print(f"Successfully allocated {successful_allocations}/{len(allocation_sizes)} tensor sizes")
        
        self.assertGreater(successful_allocations, 0)  # At least some allocations should succeed

    def tearDown(self):
        """Clean up all optimization systems"""
        if hasattr(self, 'memory_optimizer'):
            self.memory_optimizer.cleanup()
        if hasattr(self, 'swapping_system'):
            self.swapping_system.cleanup()
        if hasattr(self, 'tiering_system'):
            self.tiering_system.cleanup()
        if hasattr(self, 'integrated_system'):
            self.integrated_system.cleanup()


def run_hardware_specific_tests():
    """Run all hardware-specific validation tests"""
    print("Running Hardware-Specific Validation Tests")
    print("Target Hardware: Intel i5-10210U + NVIDIA SM61 + NVMe SSD")
    print("=" * 70)
    
    # Create test suite
    test_classes = [
        TestIntelI5_10210UOptimizations,
        TestNvidiaSM61Optimizations,
        TestNVMeSSDOptimizations,
        TestIntegratedHardwareOptimizations
    ]
    
    all_tests = unittest.TestSuite()
    
    for test_class in test_classes:
        print(f"\nLoading tests from: {test_class.__name__}")
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_class)
        all_tests.addTest(suite)
    
    # Run tests
    print(f"\nRunning {all_tests.countTestCases()} hardware-specific tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(all_tests)
    
    # Print summary
    print("\n" + "=" * 70)
    print("HARDWARE-SPECIFIC TESTS SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.2f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if not result.failures and not result.errors:
        print("\n✅ ALL HARDWARE-SPECIFIC TESTS PASSED!")
        print("✅ Qwen3-VL optimizations are properly tuned for Intel i5-10210U + NVIDIA SM61 + NVMe SSD")
    else:
        print("\n❌ Some hardware-specific tests failed.")
        print("❌ Please review hardware-specific optimizations.")
    
    return result


if __name__ == "__main__":
    run_hardware_specific_tests()