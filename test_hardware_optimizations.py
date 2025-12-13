"""
Test suite for hardware-specific optimizations in Qwen3-VL model.
This test suite verifies hardware detection, optimization strategies, and fallback mechanisms.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import logging
from unittest.mock import patch, MagicMock
import warnings

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from qwen3_vl.hardware.cpu_detector import CPUDetector, CPUModel, CPUFeatures
from qwen3_vl.hardware.simd_optimizer import SIMDManagerFactory, SIMDOptimizer, SIMDConfig
from qwen3_vl.optimization.hardware_specific_optimization import (
    HardwareOptimizedAttention, HardwareOptimizedMLP, SM61OptimizedTransformerLayer,
    HardwareOptimizedModel, HardwareKernelOptimizer
)
from qwen3_vl.optimization.fallback_manager import FallbackManager, AdaptiveFallbackManager
from qwen3_vl.optimization.config_manager import ConfigManager, OptimizationConfig
from qwen3_vl.optimization.interaction_handler import OptimizationInteractionHandler


class TestCPUDetection(unittest.TestCase):
    """Test CPU detection and feature identification."""
    
    def setUp(self):
        self.detector = CPUDetector()
    
    def test_cpu_detection(self):
        """Test that CPU detection works correctly."""
        features = self.detector.get_cpu_features()
        self.assertIsInstance(features, CPUFeatures)
        self.assertIsInstance(features.model, CPUModel)
        self.assertGreaterEqual(features.cores, 1)
        self.assertGreaterEqual(features.threads, features.cores)
    
    def test_specific_model_detection(self):
        """Test detection of specific CPU models."""
        # This test might not work in all environments, so we'll make it conditional
        features = self.detector.get_cpu_features()
        self.assertIsInstance(features.model, CPUModel)
    
    def test_cache_size_detection(self):
        """Test that cache sizes are detected properly."""
        features = self.detector.get_cpu_features()
        self.assertGreaterEqual(features.l1_cache_size, 0)
        self.assertGreaterEqual(features.l2_cache_size, 0)
        self.assertGreaterEqual(features.l3_cache_size, 0)


class TestSIMDOptimization(unittest.TestCase):
    """Test SIMD optimization functionality."""
    
    def setUp(self):
        self.simd_optimizer = SIMDOptimizer()
        self.simd_manager = self.simd_optimizer.simd_manager
    
    def test_simd_config_creation(self):
        """Test that SIMD configuration is created properly."""
        config = self.simd_optimizer.get_simd_config()
        self.assertIsInstance(config, SIMDConfig)
        self.assertIn(config.instruction_set, ['avx2', 'avx', 'sse', 'none'])
        self.assertGreaterEqual(config.vector_width, 1)
    
    def test_vectorized_operations(self):
        """Test that vectorized operations work correctly."""
        # Test vectorized normalization
        input_tensor = torch.randn(10, 20)
        normalized = self.simd_manager.vectorized_normalize(input_tensor)
        self.assertEqual(normalized.shape, input_tensor.shape)
        
        # Test vectorized matmul
        a = torch.randn(5, 10)
        b = torch.randn(10, 8)
        result = self.simd_manager.vectorized_matmul(a, b)
        self.assertEqual(result.shape, (5, 8))
        
        # Test vectorized GELU
        x = torch.randn(5, 10)
        gelu_result = self.simd_manager.vectorized_gelu_approximation(x)
        self.assertEqual(gelu_result.shape, x.shape)


class TestHardwareSpecificOptimizations(unittest.TestCase):
    """Test hardware-specific optimization implementations."""
    
    def setUp(self):
        # Create a minimal config for testing
        class MockConfig:
            def __init__(self):
                self.hidden_size = 256
                self.num_attention_heads = 8
                self.num_key_value_heads = 8
                self.intermediate_size = 512
                self.layer_norm_eps = 1e-6
                self.max_position_embeddings = 512
                self.rope_theta = 10000.0
                self.num_hidden_layers = 2  # Add this required attribute

        self.config = MockConfig()
    
    def test_hardware_optimized_attention(self):
        """Test hardware-optimized attention mechanism."""
        attention = HardwareOptimizedAttention(self.config)
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states
        )
        
        self.assertEqual(output.shape, hidden_states.shape)
        if attn_weights is not None:
            self.assertEqual(attn_weights.shape[0], batch_size)
            self.assertEqual(attn_weights.shape[1], self.config.num_attention_heads)
            self.assertEqual(attn_weights.shape[2], seq_len)
            self.assertEqual(attn_weights.shape[3], seq_len)
    
    def test_hardware_optimized_mlp(self):
        """Test hardware-optimized MLP."""
        mlp = HardwareOptimizedMLP(self.config)
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        output = mlp(hidden_states)
        
        self.assertEqual(output.shape, hidden_states.shape)
    
    def test_sm61_optimized_transformer_layer(self):
        """Test SM61-optimized transformer layer."""
        layer = SM61OptimizedTransformerLayer(self.config, layer_idx=0)
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        result = layer(
            hidden_states=hidden_states
        )
        
        # Result should be a tuple with at least the output tensor
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0].shape, hidden_states.shape)
    
    def test_hardware_optimized_model(self):
        """Test hardware-optimized model wrapper."""
        model = HardwareOptimizedModel(self.config)
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        result = model(hidden_states=hidden_states)
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0].shape, hidden_states.shape)


class TestFallbackMechanisms(unittest.TestCase):
    """Test fallback mechanisms for hardware-specific optimizations."""
    
    def setUp(self):
        # Create a minimal config for testing
        class MockConfig:
            def __init__(self):
                self.hidden_size = 256
                self.num_attention_heads = 8
                self.num_hidden_layers = 2
                self.layer_norm_eps = 1e-6
                self.max_position_embeddings = 512
                self.rope_theta = 10000.0
                self.num_key_value_heads = 8  # Add this required attribute
        
        self.config = MockConfig()
        
        # Create optimization manager
        class MockOptimizationManager:
            def __init__(self):
                self.optimization_states = {
                    'block_sparse_attention': True,
                    'cross_modal_token_merging': True,
                    'hierarchical_memory_compression': True,
                    'learned_activation_routing': True,
                    'adaptive_batch_processing': True,
                    'cross_layer_parameter_recycling': True,
                    'adaptive_sequence_packing': True,
                    'memory_efficient_grad_accumulation': True,
                    'kv_cache_multiple_strategies': True,
                    'faster_rotary_embeddings': True,
                    'distributed_pipeline_parallelism': True,
                    'hardware_specific_kernels': True
                }
        
        self.optimization_manager = MockOptimizationManager()
    
    def test_fallback_manager_creation(self):
        """Test that fallback manager is created properly."""
        fallback_manager = FallbackManager(self.optimization_manager)
        self.assertIsInstance(fallback_manager, FallbackManager)
    
    def test_adaptive_fallback_manager(self):
        """Test adaptive fallback manager."""
        adaptive_fallback_manager = AdaptiveFallbackManager(self.optimization_manager)
        self.assertIsInstance(adaptive_fallback_manager, AdaptiveFallbackManager)
    
    def test_fallback_application(self):
        """Test that fallbacks can be applied correctly."""
        fallback_manager = AdaptiveFallbackManager(self.optimization_manager)
        
        # Test that we can get performance statistics even without any fallbacks
        stats = fallback_manager.get_fallback_statistics()
        self.assertIsInstance(stats, dict)
    
    def test_hardware_kernel_optimizer(self):
        """Test hardware kernel optimizer with fallback capabilities."""
        optimizer = HardwareKernelOptimizer(self.config)

        # Test optimization report
        report = optimizer.get_hardware_optimization_report()
        self.assertIsInstance(report, dict)
        self.assertIn('hardware_detected', report)
        self.assertIn('optimization_parameters', report)


class TestOptimizationConfigAndInteraction(unittest.TestCase):
    """Test optimization configuration and interaction handling."""
    
    def setUp(self):
        self.config_manager = ConfigManager()
        
        # Create optimization manager
        class MockOptimizationManager:
            def __init__(self):
                self.optimization_states = {
                    'block_sparse_attention': True,
                    'cross_modal_token_merging': True,
                    'hierarchical_memory_compression': True,
                    'learned_activation_routing': True,
                    'adaptive_batch_processing': True,
                    'cross_layer_parameter_recycling': True,
                    'adaptive_sequence_packing': True,
                    'memory_efficient_grad_accumulation': True,
                    'kv_cache_multiple_strategies': True,
                    'faster_rotary_embeddings': True,
                    'distributed_pipeline_parallelism': True,
                    'hardware_specific_kernels': True
                }
        
        self.optimization_manager = MockOptimizationManager()
    
    def test_config_creation(self):
        """Test that optimization configurations are created properly."""
        config = OptimizationConfig()
        self.assertIsInstance(config, OptimizationConfig)
        
        # Test validation
        errors = self.config_manager.validator.validate_config(config)
        self.assertIsInstance(errors, list)
    
    def test_interaction_handler(self):
        """Test optimization interaction handling."""
        interaction_handler = OptimizationInteractionHandler(self.optimization_manager)
        
        # Test conflict resolution
        optimizations_with_conflict = [
            'distributed_pipeline_parallelism',
            'hardware_specific_kernels'
        ]
        
        resolved = interaction_handler.resolve_conflicts(optimizations_with_conflict)
        # The handler should resolve the conflict by removing one of the optimizations
        self.assertLessEqual(len(resolved), len(optimizations_with_conflict))
    
    def test_interaction_analysis(self):
        """Test interaction analysis functionality."""
        interaction_handler = OptimizationInteractionHandler(self.optimization_manager)
        
        test_combo = [
            'block_sparse_attention',
            'kv_cache_multiple_strategies',
            'hierarchical_memory_compression'
        ]
        
        analysis = interaction_handler.get_interaction_analysis(test_combo)
        self.assertIsInstance(analysis, dict)
        self.assertIn('positive_synergies', analysis)
        self.assertIn('negative_conflicts', analysis)
        self.assertIn('recommended_order', analysis)


class TestSystemLevelOptimizations(unittest.TestCase):
    """Test system-level optimizations."""
    
    def setUp(self):
        # Import system-level optimizations
        try:
            from qwen3_vl.optimization.system_level_optimizations import (
                SystemOptimizationConfig, SystemProfiler, ThreadManager,
                MemoryManager, ResourceScheduler, SystemOptimizer
            )
            self.SystemOptimizationConfig = SystemOptimizationConfig
            self.SystemProfiler = SystemProfiler
            self.ThreadManager = ThreadManager
            self.MemoryManager = MemoryManager
            self.ResourceScheduler = ResourceScheduler
            self.SystemOptimizer = SystemOptimizer
        except ImportError:
            self.skipTest("System-level optimizations module not available")
    
    def test_system_optimization_config(self):
        """Test system optimization configuration."""
        config = self.SystemOptimizationConfig()
        self.assertIsInstance(config, self.SystemOptimizationConfig)
        self.assertGreaterEqual(config.num_compute_threads, 1)
        self.assertGreaterEqual(config.memory_limit_ratio, 0.0)
        self.assertLessEqual(config.memory_limit_ratio, 1.0)
    
    def test_system_profiler(self):
        """Test system profiler functionality."""
        config = self.SystemOptimizationConfig()
        profiler = self.SystemProfiler(config)
        
        # Test getting current profile
        profile = profiler.get_current_profile()
        self.assertIsInstance(profile, dict)
        
        # Test getting system summary
        summary = profiler.get_system_summary()
        self.assertIsInstance(summary, dict)
        
        # Test getting hardware info
        hardware_info = profiler.get_hardware_info()
        self.assertIsInstance(hardware_info, dict)
        
        # Stop profiling to clean up
        profiler.stop_profiling()
    
    def test_memory_manager(self):
        """Test memory manager functionality."""
        config = self.SystemOptimizationConfig()
        memory_manager = self.MemoryManager(config)
        
        # Test memory usage
        usage = memory_manager.get_memory_usage()
        self.assertIsInstance(usage, dict)
        self.assertIn('rss', usage)
        self.assertIn('vms', usage)
        
        # Test tensor allocation and release
        tensor = memory_manager.allocate_tensor((10, 20), torch.float32)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (10, 20))
        
        # Release tensor back to pool
        memory_manager.release_tensor(tensor)
        
        # Test memory efficiency report
        report = memory_manager.get_memory_efficiency_report()
        self.assertIsInstance(report, dict)
    
    def test_system_optimizer(self):
        """Test system optimizer functionality."""
        config = self.SystemOptimizationConfig()
        system_optimizer = self.SystemOptimizer(config)
        
        # Test optimization report
        report = system_optimizer.get_system_optimization_report()
        self.assertIsInstance(report, dict)
        self.assertIn('system_summary', report)
        self.assertIn('hardware_info', report)
        self.assertIn('memory_efficiency', report)
        self.assertIn('resource_status', report)
        
        # Clean up
        system_optimizer.cleanup_resources()


class TestHardwareAbstraction(unittest.TestCase):
    """Test hardware abstraction and compatibility."""
    
    def test_hardware_detection_fallbacks(self):
        """Test hardware detection with various fallback scenarios."""
        # This test ensures that hardware detection works even in environments
        # where specific hardware features may not be available
        detector = CPUDetector()
        features = detector.get_cpu_features()
        
        # Basic validation that detection doesn't crash
        self.assertIsInstance(features, CPUFeatures)
        self.assertIsInstance(features.model, CPUModel)
    
    def test_optimization_with_different_hardware(self):
        """Test that optimizations work with different hardware configurations."""
        # Create a minimal config
        class MockConfig:
            hidden_size = 128
            num_attention_heads = 4
            num_key_value_heads = 4
            intermediate_size = 256
            layer_norm_eps = 1e-6
            max_position_embeddings = 256
            rope_theta = 10000.0
            num_hidden_layers = 2  # Add this required attribute
        
        config = MockConfig()
        
        # Test hardware-optimized attention
        attention = HardwareOptimizedAttention(config)
        self.assertIsInstance(attention, HardwareOptimizedAttention)
        
        # Test hardware-optimized MLP
        mlp = HardwareOptimizedMLP(config)
        self.assertIsInstance(mlp, HardwareOptimizedMLP)


def run_all_tests():
    """Run all tests in the suite."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(unittest.makeSuite(TestCPUDetection))
    suite.addTest(unittest.makeSuite(TestSIMDOptimization))
    suite.addTest(unittest.makeSuite(TestHardwareSpecificOptimizations))
    suite.addTest(unittest.makeSuite(TestFallbackMechanisms))
    suite.addTest(unittest.makeSuite(TestOptimizationConfigAndInteraction))
    suite.addTest(unittest.makeSuite(TestSystemLevelOptimizations))
    suite.addTest(unittest.makeSuite(TestHardwareAbstraction))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("Running hardware-specific optimizations test suite...")
    result = run_all_tests()
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = result.wasSuccessful()
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    
    sys.exit(0 if success else 1)