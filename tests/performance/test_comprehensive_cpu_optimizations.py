"""
Test suite for comprehensive low-level CPU optimizations and kernel fusion for Qwen3-VL model
"""
import torch
import torch.nn as nn
import unittest
import time
import numpy as np
from typing import Optional, Tuple
from comprehensive_cpu_optimizations import (
    OptimizationConfig,
    LoopTilingOptimizer,
    CacheBlockingOptimizer,
    ManualSIMDOptimizer,
    MemoryPrefetchOptimizer,
    KernelFusionOptimizer,
    JITCompiler,
    MemoryPool,
    OptimizedAttention,
    OptimizedMLP,
    OptimizedDecoderLayer,
    apply_low_level_optimizations_to_model,
    benchmark_optimizations
)


class TestLoopTilingOptimizer(unittest.TestCase):
    """Test loop tiling optimizations"""
    
    def setUp(self):
        self.config = OptimizationConfig()
        self.optimizer = LoopTilingOptimizer(self.config)

    def test_tiled_matmul(self):
        """Test tiled matrix multiplication"""
        A = torch.randn(128, 256)
        B = torch.randn(256, 64)
        
        result_tiled = self.optimizer.tiled_matmul(A, B, tile_size=32)
        result_standard = torch.matmul(A, B)
        
        self.assertTrue(torch.allclose(result_tiled, result_standard, atol=1e-5))
        self.assertEqual(result_tiled.shape, (128, 64))

    def test_tiled_attention(self):
        """Test tiled attention computation"""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 32, 64
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        result_tiled = self.optimizer.tiled_attention(query, key, value, tile_size=16)
        # Standard attention computation
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        result_standard = torch.matmul(attn_weights, value)
        
        self.assertTrue(torch.allclose(result_tiled, result_standard, atol=1e-4))
        self.assertEqual(result_tiled.shape, (batch_size, num_heads, seq_len, head_dim))


class TestCacheBlockingOptimizer(unittest.TestCase):
    """Test cache blocking optimizations"""
    
    def setUp(self):
        self.config = OptimizationConfig()
        self.optimizer = CacheBlockingOptimizer(self.config)

    def test_cache_blocked_layer_norm(self):
        """Test cache-blocked layer normalization"""
        x = torch.randn(4, 32, 512)
        weight = torch.ones(512)
        bias = torch.zeros(512)
        
        result_blocked = self.optimizer.cache_blocked_layer_norm(x, weight, bias, block_size=64)
        result_standard = torch.layer_norm(x, x.shape[-1:], weight, bias, 1e-5)
        
        self.assertTrue(torch.allclose(result_blocked, result_standard, atol=1e-5))
        self.assertEqual(result_blocked.shape, x.shape)

    def test_cache_blocked_softmax(self):
        """Test cache-blocked softmax"""
        x = torch.randn(4, 32, 128)
        
        result_blocked = self.optimizer.cache_blocked_softmax(x, dim=-1, block_size=32)
        result_standard = torch.softmax(x, dim=-1)
        
        self.assertTrue(torch.allclose(result_blocked, result_standard, atol=1e-5))
        self.assertEqual(result_blocked.shape, x.shape)


class TestManualSIMDOptimizer(unittest.TestCase):
    """Test manual SIMD optimizations"""
    
    def setUp(self):
        self.config = OptimizationConfig()
        self.optimizer = ManualSIMDOptimizer(self.config)

    def test_simd_gelu(self):
        """Test SIMD-optimized GELU"""
        x = torch.randn(4, 32, 512)

        result_simd = self.optimizer.simd_gelu(x)
        result_standard = torch.nn.functional.gelu(x)

        # Use a more relaxed tolerance since GELU approximations can differ slightly
        self.assertTrue(torch.allclose(result_simd, result_standard, atol=1e-3))
        self.assertEqual(result_simd.shape, x.shape)

    def test_simd_silu(self):
        """Test SIMD-optimized SiLU"""
        x = torch.randn(4, 32, 512)
        
        result_simd = self.optimizer.simd_silu(x)
        result_standard = torch.nn.functional.silu(x)
        
        self.assertTrue(torch.allclose(result_simd, result_standard, atol=1e-5))
        self.assertEqual(result_simd.shape, x.shape)

    def test_simd_add(self):
        """Test SIMD-optimized addition"""
        a = torch.randn(4, 32, 512)
        b = torch.randn(4, 32, 512)
        
        result_simd = self.optimizer.simd_add(a, b)
        result_standard = a + b
        
        self.assertTrue(torch.allclose(result_simd, result_standard, atol=1e-5))
        self.assertEqual(result_simd.shape, a.shape)

    def test_simd_multiply(self):
        """Test SIMD-optimized multiplication"""
        a = torch.randn(4, 32, 512)
        b = torch.randn(4, 32, 512)
        
        result_simd = self.optimizer.simd_multiply(a, b)
        result_standard = a * b
        
        self.assertTrue(torch.allclose(result_simd, result_standard, atol=1e-5))
        self.assertEqual(result_simd.shape, a.shape)


class TestMemoryPrefetchOptimizer(unittest.TestCase):
    """Test memory prefetching optimizations"""
    
    def setUp(self):
        self.config = OptimizationConfig()
        self.optimizer = MemoryPrefetchOptimizer(self.config)

    def test_prefetch_tensor(self):
        """Test tensor prefetching"""
        tensor = torch.randn(4, 32, 512)
        key = "test_tensor"
        
        self.optimizer.prefetch_tensor(key, tensor)
        retrieved_tensor = self.optimizer.get_prefetched_tensor(key)
        
        self.assertIsNotNone(retrieved_tensor)
        self.assertTrue(torch.equal(tensor, retrieved_tensor))

    def test_predict_next_access(self):
        """Test access pattern prediction"""
        # This is a simple test since the prediction logic is basic
        result = self.optimizer.predict_next_access("test_access")
        # The result might be None since there's no history yet
        self.assertIsNone(result)  # or not None if history exists


class TestKernelFusionOptimizer(unittest.TestCase):
    """Test kernel fusion optimizations"""
    
    def setUp(self):
        self.config = OptimizationConfig()
        self.optimizer = KernelFusionOptimizer(self.config)

    def test_fused_attention_softmax(self):
        """Test fused attention-softmax"""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 32, 64
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        result_fused = self.optimizer.fused_attention_softmax(query, key, value)
        
        # Standard computation
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        result_standard = torch.matmul(attn_weights, value)
        
        self.assertTrue(torch.allclose(result_fused, result_standard, atol=1e-4))
        self.assertEqual(result_fused.shape, (batch_size, num_heads, seq_len, head_dim))

    def test_fused_mlp_block(self):
        """Test fused MLP block"""
        batch_size, seq_len, hidden_size, intermediate_size = 2, 32, 512, 2048
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Create random weights and biases
        gate_weight = torch.randn(intermediate_size, hidden_size)
        up_weight = torch.randn(intermediate_size, hidden_size)
        down_weight = torch.randn(hidden_size, intermediate_size)
        
        result_fused = self.optimizer.fused_mlp_block(
            x, gate_weight, up_weight, down_weight
        )
        
        # Standard computation
        gate_output = torch.nn.functional.linear(x, gate_weight)
        up_output = torch.nn.functional.linear(x, up_weight)
        activated_gate = torch.nn.functional.silu(gate_output)
        intermediate_output = activated_gate * up_output
        result_standard = torch.nn.functional.linear(intermediate_output, down_weight)
        
        self.assertTrue(torch.allclose(result_fused, result_standard, atol=1e-4))
        self.assertEqual(result_fused.shape, (batch_size, seq_len, hidden_size))

    def test_fused_layer_norm_linear(self):
        """Test fused layer norm + linear"""
        batch_size, seq_len, hidden_size = 2, 32, 512
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        ln_weight = torch.ones(hidden_size)
        ln_bias = torch.zeros(hidden_size)
        linear_weight = torch.randn(256, hidden_size)
        linear_bias = torch.randn(256)
        
        result_fused = self.optimizer.fused_layer_norm_linear(
            x, ln_weight, ln_bias, linear_weight, linear_bias
        )
        
        # Standard computation
        x_norm = torch.layer_norm(x, x.shape[-1:], ln_weight, ln_bias, 1e-5)
        result_standard = torch.nn.functional.linear(x_norm, linear_weight, linear_bias)
        
        self.assertTrue(torch.allclose(result_fused, result_standard, atol=1e-4))
        self.assertEqual(result_fused.shape, (batch_size, seq_len, 256))

    def test_fused_residual_add_layer_norm(self):
        """Test fused residual add + layer norm"""
        batch_size, seq_len, hidden_size = 2, 32, 512
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        residual = torch.randn(batch_size, seq_len, hidden_size)
        
        ln_weight = torch.ones(hidden_size)
        ln_bias = torch.zeros(hidden_size)
        
        result_fused = self.optimizer.fused_residual_add_layer_norm(
            hidden_states, residual, ln_weight, ln_bias
        )
        
        # Standard computation
        hidden_states = hidden_states + residual
        result_standard = torch.layer_norm(hidden_states, hidden_states.shape[-1:], ln_weight, ln_bias, 1e-5)
        
        self.assertTrue(torch.allclose(result_fused, result_standard, atol=1e-4))
        self.assertEqual(result_fused.shape, (batch_size, seq_len, hidden_size))


class TestMemoryPool(unittest.TestCase):
    """Test memory pooling"""
    
    def setUp(self):
        self.config = OptimizationConfig()
        self.memory_pool = MemoryPool(self.config)

    def test_get_and_return_tensor(self):
        """Test getting and returning tensors from the pool"""
        shape = (4, 32, 512)
        dtype = torch.float32
        device = torch.device('cpu')
        
        # Get a tensor from the pool
        tensor1 = self.memory_pool.get_tensor(shape, dtype, device)
        self.assertEqual(tensor1.shape, shape)
        self.assertEqual(tensor1.dtype, dtype)
        self.assertEqual(tensor1.device, device)
        
        # Return the tensor to the pool
        self.memory_pool.return_tensor(tensor1)
        
        # Get another tensor from the pool (should be the same one)
        tensor2 = self.memory_pool.get_tensor(shape, dtype, device)
        self.assertEqual(tensor2.shape, shape)
        self.assertEqual(tensor2.dtype, dtype)
        self.assertEqual(tensor2.device, device)


class TestOptimizedModules(unittest.TestCase):
    """Test optimized modules"""
    
    def setUp(self):
        # Create a mock config
        class MockConfig:
            def __init__(self):
                self.hidden_size = 512
                self.intermediate_size = 2048
                self.num_attention_heads = 8
                self.num_hidden_layers = 4
                self.layer_norm_eps = 1e-5
                self.max_position_embeddings = 512
                self.rope_theta = 10000
                self.vocab_size = 32000
                self.use_cache = True
                self.num_key_value_heads = 8

        self.config = MockConfig()

    def test_optimized_attention(self):
        """Test optimized attention module"""
        attention = OptimizedAttention(self.config, layer_idx=0)
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        
        self.assertEqual(output.shape, hidden_states.shape)
        if attn_weights is not None:
            self.assertEqual(attn_weights.shape, (batch_size, self.config.num_attention_heads, seq_len, seq_len))

    def test_optimized_mlp(self):
        """Test optimized MLP module"""
        mlp = OptimizedMLP(self.config)
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        output = mlp(hidden_states)
        
        self.assertEqual(output.shape, hidden_states.shape)

    def test_optimized_decoder_layer(self):
        """Test optimized decoder layer"""
        layer = OptimizedDecoderLayer(self.config, layer_idx=0)
        
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        
        output_tuple = layer(
            hidden_states=hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        
        output = output_tuple[0]
        self.assertEqual(output.shape, hidden_states.shape)


class TestModelOptimization(unittest.TestCase):
    """Test model optimization application"""
    
    def setUp(self):
        # Create a mock config
        class MockConfig:
            def __init__(self):
                self.hidden_size = 512
                self.intermediate_size = 2048
                self.num_attention_heads = 8
                self.num_hidden_layers = 4
                self.layer_norm_eps = 1e-5
                self.max_position_embeddings = 512
                self.rope_theta = 10000
                self.vocab_size = 32000
                self.use_cache = True
                self.num_key_value_heads = 8

        self.config = MockConfig()

    def test_apply_low_level_optimizations(self):
        """Test applying low-level optimizations to a model"""
        # Create a mock model with the expected structure
        class MockModel(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.language_model = nn.Module()
                
                # Create mock layers with attention and MLP components
                layers = []
                for i in range(config.num_hidden_layers):
                    layer = nn.Module()
                    layer.self_attn = nn.Module()
                    layer.mlp = nn.Module()
                    
                    # Add the expected projection layers
                    layer.self_attn.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
                    layer.self_attn.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
                    layer.self_attn.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
                    layer.self_attn.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
                    
                    layer.mlp.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
                    layer.mlp.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
                    layer.mlp.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
                    
                    layer.input_layernorm = nn.LayerNorm(config.hidden_size)
                    layer.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
                    
                    layers.append(layer)
                
                self.language_model.layers = nn.ModuleList(layers)

        model = MockModel(self.config)
        
        # Apply optimizations
        optimized_model = apply_low_level_optimizations_to_model(model, OptimizationConfig())
        
        # Check that the model still has the expected structure
        self.assertEqual(len(optimized_model.language_model.layers), self.config.num_hidden_layers)
        
        # Check that the first layer has been replaced with optimized versions
        first_layer = optimized_model.language_model.layers[0]
        self.assertIsInstance(first_layer.self_attn, OptimizedAttention)
        self.assertIsInstance(first_layer.mlp, OptimizedMLP)


class TestJITCompiler(unittest.TestCase):
    """Test JIT compiler"""
    
    def setUp(self):
        self.jit_compiler = JITCompiler()

    def test_compile_if_frequent(self):
        """Test JIT compilation based on frequency"""
        def simple_func(x):
            return x * 2 + 1
        
        # Call the function multiple times to trigger compilation
        func = simple_func
        for i in range(15):  # More than the threshold
            func = self.jit_compiler.compile_if_frequent("test_func", simple_func)
            result = func(torch.tensor(5.0))
        
        # Result should be correct regardless of JIT compilation
        self.assertEqual(result.item(), 11.0)


class TestBenchmarkOptimizations(unittest.TestCase):
    """Test benchmarking function"""
    
    def test_benchmark_optimizations(self):
        """Test the benchmarking function runs without errors"""
        results = benchmark_optimizations()
        
        # Check that results contain expected keys
        self.assertIn('matmul_speedup', results)
        self.assertIn('norm_speedup', results)
        self.assertIn('gelu_speedup', results)
        self.assertIn('fused_attn_time', results)


def run_all_tests():
    """Run all tests in the suite"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test cases
    test_classes = [
        TestLoopTilingOptimizer,
        TestCacheBlockingOptimizer,
        TestManualSIMDOptimizer,
        TestMemoryPrefetchOptimizer,
        TestKernelFusionOptimizer,
        TestMemoryPool,
        TestOptimizedModules,
        TestModelOptimization,
        TestJITCompiler,
        TestBenchmarkOptimizations
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running comprehensive low-level CPU optimization tests...")
    success = run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! 🎉")
    else:
        print("\n❌ SOME TESTS FAILED!")
        exit(1)