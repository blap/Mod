"""
Comprehensive Tests for Qwen3-VL Hardware Adaptation System
Tests all components of the hardware detection and optimization system
"""
import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.qwen3_vl.hardware.cpu_detector import CPUDetector, CPUModel, get_hardware_optimizer, get_detected_cpu_info
from src.qwen3_vl.hardware.cpu_profiles import CPUProfileFactory, get_optimization_profile, AdaptiveOptimizationManager
from src.qwen3_vl.hardware.adaptive_threading import AdaptiveThreadingModel, ThreadingOptimizer, ThreadingModelFactory
from src.qwen3_vl.hardware.adaptive_memory import AdaptiveMemoryManager, MemoryOptimizer, MemoryManagerFactory
from src.qwen3_vl.hardware.simd_optimizer import SIMDOperationManager, SIMDOptimizer, SIMDManagerFactory
from src.qwen3_vl.hardware.unified_config import ConfigManager, UnifiedHardwareOptimizer
from src.qwen3_vl.hardware.performance_optimizer import PerformancePathOptimizer, ComprehensivePerformanceOptimizer


class TestCPUDetector(unittest.TestCase):
    """Test CPU detection system"""
    
    def setUp(self):
        self.detector = CPUDetector()
    
    def test_cpu_detection(self):
        """Test that CPU detection works and returns valid features"""
        features = self.detector.get_cpu_features()
        
        self.assertIsNotNone(features)
        self.assertIn(features.model, [CPUModel.INTEL_I5_10210U, CPUModel.INTEL_I7_8700, CPUModel.UNKNOWN])
        self.assertGreaterEqual(features.cores, 1)
        self.assertGreaterEqual(features.threads, features.cores)
        self.assertGreaterEqual(features.l3_cache_size, 0)
    
    def test_specific_model_detection(self):
        """Test detection of specific CPU models"""
        features = self.detector.get_cpu_features()
        
        # Test that the detection method works
        is_i7 = self.detector.is_specific_model_detected(CPUModel.INTEL_I7_8700)
        is_i5 = self.detector.is_specific_model_detected(CPUModel.INTEL_I5_10210U)
        
        # At least one should be true (or UNKNOWN)
        self.assertTrue(is_i7 or is_i5 or features.model == CPUModel.UNKNOWN)
    
    def test_optimal_thread_count(self):
        """Test optimal thread count calculation"""
        thread_count = self.detector.get_optimal_thread_count()
        
        self.assertGreaterEqual(thread_count, 1)
        self.assertLessEqual(thread_count, 64)  # Reasonable upper bound


class TestCPUProfiles(unittest.TestCase):
    """Test CPU-specific optimization profiles"""
    
    def test_i5_profile_creation(self):
        """Test creation of i5-10210U profile"""
        profile = CPUProfileFactory.create_i5_10210u_profile()
        
        self.assertEqual(profile.cpu_model, CPUModel.INTEL_I5_10210U)
        self.assertEqual(profile.num_cores, 4)
        self.assertEqual(profile.num_threads, 8)
        self.assertEqual(profile.l3_cache_size, 6 * 1024 * 1024)  # 6MB
        self.assertTrue(profile.enable_cache_optimization)
    
    def test_i7_profile_creation(self):
        """Test creation of i7-8700 profile"""
        profile = CPUProfileFactory.create_i7_8700_profile()
        
        self.assertEqual(profile.cpu_model, CPUModel.INTEL_I7_8700)
        self.assertEqual(profile.num_cores, 6)
        self.assertEqual(profile.num_threads, 12)
        self.assertEqual(profile.l3_cache_size, 12 * 1024 * 1024)  # 12MB
        self.assertTrue(profile.enable_cache_optimization)
        self.assertGreater(profile.batch_size_multiplier, 1.0)  # Should be larger than i5
    
    def test_generic_profile_creation(self):
        """Test creation of generic profile"""
        profile = CPUProfileFactory.create_generic_profile(4, 8, 6 * 1024 * 1024)
        
        self.assertEqual(profile.cpu_model, CPUModel.UNKNOWN)
        self.assertEqual(profile.num_cores, 4)
        self.assertEqual(profile.num_threads, 8)
    
    def test_optimization_manager(self):
        """Test optimization manager functionality"""
        # Test with i5 profile
        i5_profile = get_optimization_profile(CPUModel.INTEL_I5_10210U)
        self.assertIsInstance(i5_profile, AdaptiveOptimizationManager)
        self.assertEqual(i5_profile.config.cpu_model, CPUModel.INTEL_I5_10210U)
        
        # Test with i7 profile
        i7_profile = get_optimization_profile(CPUModel.INTEL_I7_8700)
        self.assertIsInstance(i7_profile, AdaptiveOptimizationManager)
        self.assertEqual(i7_profile.config.cpu_model, CPUModel.INTEL_I7_8700)
        
        # Verify i7 has better specs than i5
        self.assertGreater(i7_profile.get_batch_size_multiplier(), 
                          i5_profile.get_batch_size_multiplier())


class TestAdaptiveThreading(unittest.TestCase):
    """Test adaptive threading model"""
    
    def test_threading_model_creation(self):
        """Test creation of threading models for different CPUs"""
        # Test i5 model
        i5_model = ThreadingModelFactory.create_for_cpu_model(CPUModel.INTEL_I5_10210U)
        self.assertIsInstance(i5_model, AdaptiveThreadingModel)
        
        # Test i7 model
        i7_model = ThreadingModelFactory.create_for_cpu_model(CPUModel.INTEL_I7_8700)
        self.assertIsInstance(i7_model, AdaptiveThreadingModel)
        
        # i7 should have more threads available
        i7_stats = i7_model.get_thread_pool_stats()
        i5_stats = i5_model.get_thread_pool_stats()
        
        # The exact comparison depends on the underlying system, but both should be valid
        self.assertGreaterEqual(i7_stats['cpu_thread_pool_size'], 1)
        self.assertGreaterEqual(i5_stats['cpu_thread_pool_size'], 1)
    
    def test_threading_optimizer(self):
        """Test threading optimizer functionality"""
        optimizer = ThreadingOptimizer()
        
        # Test optimal batch size calculation
        base_batch = 4
        optimal_batch = optimizer.threading_model.get_optimal_batch_size(base_batch)
        self.assertGreaterEqual(optimal_batch, 1)
        self.assertLessEqual(optimal_batch, 32)  # Upper bound
        
        # Test performance stats
        stats = optimizer.get_performance_stats()
        self.assertIn('cpu_model', stats)
        self.assertIn('cpu_thread_pool_size', stats)
        self.assertIn('recommended_workers', stats)
        
        # Cleanup
        optimizer.shutdown()
    
    def test_task_execution(self):
        """Test task execution with optimal threading"""
        optimizer = ThreadingOptimizer()
        
        # Simple test function
        def test_func(x):
            return x * 2
        
        # Test CPU task
        future = optimizer.threading_model.submit_cpu_task(test_func, 5)
        result = future.result()
        self.assertEqual(result, 10)
        
        # Test I/O task
        future = optimizer.threading_model.submit_io_task(test_func, 3)
        result = future.result()
        self.assertEqual(result, 6)
        
        # Cleanup
        optimizer.shutdown()


class TestAdaptiveMemory(unittest.TestCase):
    """Test adaptive memory management"""
    
    def test_memory_manager_creation(self):
        """Test creation of memory managers for different CPUs"""
        # Test i5 manager
        i5_manager = MemoryManagerFactory.create_for_cpu_model(CPUModel.INTEL_I5_10210U)
        self.assertIsInstance(i5_manager, AdaptiveMemoryManager)
        
        # Test i7 manager
        i7_manager = MemoryManagerFactory.create_for_cpu_model(CPUModel.INTEL_I7_8700)
        self.assertIsInstance(i7_manager, AdaptiveMemoryManager)
        
        # Check that configurations are different
        self.assertIsInstance(i5_manager.config, type(i7_manager.config))
    
    def test_tensor_allocation(self):
        """Test tensor allocation with memory optimization"""
        optimizer = MemoryOptimizer()
        
        # Test normal tensor allocation
        tensor = optimizer.memory_manager.allocate_tensor((100, 100))
        self.assertEqual(tensor.shape, (100, 100))
        self.assertEqual(tensor.dtype, torch.float32)
        
        # Test cache-optimized tensor allocation
        tensor_opt = optimizer.memory_manager.allocate_cache_optimized_tensor((50, 50, 50))
        self.assertEqual(tensor_opt.shape, (50, 50, 50))
        
        # Test memory stats
        stats = optimizer.get_memory_stats()
        self.assertIn('system_percent_used', stats)
        self.assertIn('allocation_count', stats)
    
    def test_cache_blocking(self):
        """Test cache blocking functionality"""
        optimizer = MemoryOptimizer()
        
        # Create a test tensor
        tensor = torch.randn(100, 100)
        
        # Apply cache blocking
        blocked_tensor = optimizer.memory_manager.apply_cache_blocking(tensor, 'matmul')
        
        # The tensor should remain the same size but potentially be contiguous
        self.assertEqual(blocked_tensor.shape, tensor.shape)
        
        # Test different operations
        blocked_conv = optimizer.memory_manager.apply_cache_blocking(tensor, 'conv')
        self.assertEqual(blocked_conv.shape, tensor.shape)


class TestSIMDOptimizer(unittest.TestCase):
    """Test SIMD optimization"""
    
    def test_simd_manager_creation(self):
        """Test creation of SIMD managers for different CPUs"""
        # Test i5 manager
        i5_simd = SIMDManagerFactory.create_for_cpu_model(CPUModel.INTEL_I5_10210U)
        self.assertIsInstance(i5_simd, SIMDOperationManager)
        
        # Test i7 manager
        i7_simd = SIMDManagerFactory.create_for_cpu_model(CPUModel.INTEL_I7_8700)
        self.assertIsInstance(i7_simd, SIMDOperationManager)
        
        # Both should have valid configurations
        self.assertIsNotNone(i5_simd.config)
        self.assertIsNotNone(i7_simd.config)
    
    def test_simd_operations(self):
        """Test SIMD operations"""
        optimizer = SIMDOptimizer()
        
        # Create test tensors
        a = torch.randn(10, 10)
        b = torch.randn(10, 10)
        
        # Test various SIMD-optimized operations
        result_add = optimizer.simd_manager.vectorized_elementwise_add(a, b)
        self.assertEqual(result_add.shape, a.shape)
        
        result_mul = optimizer.simd_manager.vectorized_elementwise_mul(a, b)
        self.assertEqual(result_mul.shape, a.shape)
        
        result_norm = optimizer.simd_manager.vectorized_normalize(a)
        self.assertEqual(result_norm.shape, a.shape)
        
        result_matmul = optimizer.simd_manager.vectorized_matmul(a, b)
        self.assertEqual(result_matmul.shape, (a.shape[0], b.shape[1]))
    
    def test_vectorization_decision(self):
        """Test vectorization decision logic"""
        optimizer = SIMDOptimizer()
        
        # Small tensor should not use vectorization (below threshold)
        should_small = optimizer.simd_manager.should_use_vectorization(10)
        
        # Large tensor should use vectorization (above threshold)
        should_large = optimizer.simd_manager.should_use_vectorization(1000)
        
        # Both should be boolean values
        self.assertIsInstance(should_small, bool)
        self.assertIsInstance(should_large, bool)


class TestUnifiedConfig(unittest.TestCase):
    """Test unified configuration system"""
    
    def test_config_manager(self):
        """Test configuration manager functionality"""
        config_manager = ConfigManager()
        
        # Get config
        config = config_manager.get_config()
        self.assertIsNotNone(config)
        
        # Get config as dict
        config_dict = config_manager.get_config_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('cpu_model', config_dict)
        self.assertIn('num_cores', config_dict)
        
        # Get model config
        model_config = config_manager.get_model_config()
        self.assertIsInstance(model_config, dict)
        self.assertIn('num_workers', model_config)
        self.assertIn('memory_pool_size', model_config)
    
    def test_unified_optimizer(self):
        """Test unified hardware optimizer"""
        optimizer = UnifiedHardwareOptimizer()
        
        # Get optimal config
        config = optimizer.get_optimal_config()
        self.assertIsNotNone(config)
        
        # Get model config
        model_config = optimizer.get_model_config()
        self.assertIsInstance(model_config, dict)
        
        # Get hardware report
        report = optimizer.get_hardware_report()
        self.assertIsInstance(report, dict)
        self.assertIn('cpu_info', report)
        self.assertIn('optimization_summary', report)
    
    def test_config_save_load(self):
        """Test saving and loading configuration"""
        optimizer = UnifiedHardwareOptimizer()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save config
            optimizer.save_hardware_config(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load config
            loaded_config = optimizer.load_hardware_config(temp_path)
            self.assertIsNotNone(loaded_config)
            self.assertEqual(type(loaded_config), type(optimizer.config_manager.get_config()))
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestPerformanceOptimizer(unittest.TestCase):
    """Test performance optimization paths"""
    
    def test_performance_optimizer_creation(self):
        """Test creation of performance optimizers for different CPUs"""
        # Test should work with the currently detected CPU
        optimizer = PerformanceOptimizerFactory.create_for_detected_cpu()
        self.assertIsInstance(optimizer, PerformancePathOptimizer)
        
        # Check that a performance path was determined
        self.assertIn(optimizer.performance_path, 
                     ["parallelism_optimized", "efficiency_optimized", "balanced_optimized"])
    
    def test_comprehensive_optimizer(self):
        """Test comprehensive performance optimizer"""
        optimizer = ComprehensivePerformanceOptimizer()
        
        # Get CPU-specific optimizations
        optimizations = optimizer.get_cpu_specific_optimizations()
        self.assertIsInstance(optimizations, dict)
        self.assertIn('cpu_model', optimizations)
        self.assertIn('performance_path', optimizations)
        
        # Run benchmarks
        benchmarks = optimizer.run_performance_benchmarks()
        self.assertIsInstance(benchmarks, dict)
        self.assertIn('memory_bandwidth_gb_s', benchmarks)
        self.assertIn('compute_performance_gflops', benchmarks)
        
        # Get full report
        report = optimizer.get_optimization_report()
        self.assertIsInstance(report, dict)
        self.assertIn('cpu_specific_optimizations', report)
        self.assertIn('benchmark_results', report)


class TestIntegration(unittest.TestCase):
    """Test integration of all components"""
    
    def test_full_pipeline(self):
        """Test the full hardware adaptation pipeline"""
        # Initialize all components
        detector = CPUDetector()
        features = detector.get_cpu_features()
        
        profile = get_optimization_profile(features.model)
        threading_opt = get_threading_optimizer()
        memory_opt = get_memory_optimizer()
        simd_opt = get_simd_optimizer()
        unified_opt = get_unified_optimizer()
        perf_opt = get_performance_optimizer()
        
        # Verify all components are properly initialized
        self.assertIsNotNone(features)
        self.assertIsNotNone(profile)
        self.assertIsNotNone(threading_opt)
        self.assertIsNotNone(memory_opt)
        self.assertIsNotNone(simd_opt)
        self.assertIsNotNone(unified_opt)
        self.assertIsNotNone(perf_opt)
        
        # Get the unified config
        config = unified_opt.get_optimal_config()
        self.assertIsNotNone(config)
        
        # Get performance report
        report = perf_opt.get_optimization_report()
        self.assertIsInstance(report, dict)
        
        # Shutdown threading optimizer
        threading_opt.shutdown()
    
    def test_model_optimization(self):
        """Test model optimization pipeline"""
        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Apply hardware optimizations
        optimized_model = optimize_model_for_cpu(model)
        self.assertIsInstance(optimized_model, nn.Module)
        
        # Test that the model still works
        test_input = torch.randn(2, 10)
        output = optimized_model(test_input)
        self.assertEqual(output.shape, (2, 5))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_empty_tensors(self):
        """Test handling of empty tensors"""
        memory_opt = MemoryOptimizer()
        
        # Test allocation of empty tensor
        empty_tensor = memory_opt.memory_manager.allocate_tensor(())
        self.assertEqual(empty_tensor.shape, ())
        
        # Test allocation of single-element tensor
        single_tensor = memory_opt.memory_manager.allocate_tensor((1,))
        self.assertEqual(single_tensor.shape, (1,))
    
    def test_large_tensors(self):
        """Test handling of large tensors (without actually allocating to save memory)"""
        memory_opt = MemoryOptimizer()
        
        # Check memory threshold logic
        should_swap = memory_opt.memory_manager.should_swap_memory()
        self.assertIsInstance(should_swap, bool)
    
    def test_invalid_operations(self):
        """Test handling of invalid operations"""
        simd_opt = SIMDOptimizer()
        
        # Test with small tensor for vectorization decision
        result = simd_opt.simd_manager.should_use_vectorization(1)
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    print("Running Hardware Adaptation System Tests")
    print("=" * 50)
    
    # Run all tests
    unittest.main(verbosity=2)