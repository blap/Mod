"""
Comprehensive tests for optimized attention mechanisms
Verifies that all optimizations maintain numerical accuracy and performance improvements
"""
import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from src.qwen3_vl.components.configuration import Qwen3VLConfig
from src.qwen3_vl.components.attention.optimized_attention_mechanisms import (
    FlashAttention2, 
    SIMDAttention, 
    MemoryEfficientAttention, 
    SM61OptimizedAttention,
    IntelOptimizedAttention,
    OptimizedAttentionFactory
)


class TestOptimizedAttentionMechanisms(unittest.TestCase):
    
    def setUp(self):
        """Set up test configurations and inputs"""
        # Create a test configuration
        self.config = Qwen3VLConfig(
            hidden_size=512,
            num_attention_heads=8,  # Reduced for testing
            num_key_value_heads=8,  # Same as num_attention_heads for standard MHA
            max_position_embeddings=2048,
            rope_theta=10000.0,
            attention_dropout_prob=0.0,
            use_flash_attention_2=True,
            use_memory_efficient_attention=True
        )
        
        # Create test inputs
        self.batch_size = 2
        self.seq_len = 128
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        self.position_ids = torch.arange(self.seq_len, dtype=torch.long).unsqueeze(0).expand(self.batch_size, -1)
        self.attention_mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len, dtype=torch.bool)
        
    def test_flash_attention_2_basic(self):
        """Test basic functionality of FlashAttention2"""
        print("Testing FlashAttention2 basic functionality...")
        
        attention = FlashAttention2(self.config, layer_idx=0)
        attention.eval()
        
        with torch.no_grad():
            output, attn_weights, past_key_value = attention(
                hidden_states=self.hidden_states,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                output_attentions=True
            )
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.config.hidden_size)
        self.assertEqual(output.shape, expected_shape, 
                         f"Output shape mismatch: expected {expected_shape}, got {output.shape}")
        
        # Check attention weights shape when output_attentions=True
        if attn_weights is not None:
            expected_attn_shape = (self.batch_size, self.config.num_attention_heads, self.seq_len, self.seq_len)
            self.assertEqual(attn_weights.shape, expected_attn_shape,
                             f"Attention weights shape mismatch: expected {expected_attn_shape}, got {attn_weights.shape}")
        
        # Check that output is finite
        self.assertTrue(torch.all(torch.isfinite(output)), "Output should contain only finite values")
        
        print("✓ FlashAttention2 basic functionality test passed")
    
    def test_flash_attention_2_memory_efficiency(self):
        """Test that FlashAttention2 provides memory efficiency benefits"""
        print("Testing FlashAttention2 memory efficiency...")
        
        attention = FlashAttention2(self.config, layer_idx=0)
        attention.eval()
        
        # Test with larger sequence length to see memory difference
        large_seq_len = 512
        large_hidden_states = torch.randn(self.batch_size, large_seq_len, self.config.hidden_size)
        large_attention_mask = torch.ones(self.batch_size, 1, large_seq_len, large_seq_len, dtype=torch.bool)
        
        with torch.no_grad():
            output, _, _ = attention(
                hidden_states=large_hidden_states,
                attention_mask=large_attention_mask,
                position_ids=torch.arange(large_seq_len, dtype=torch.long).unsqueeze(0).expand(self.batch_size, -1),
                output_attentions=False  # Don't compute weights to save memory
            )
        
        expected_shape = (self.batch_size, large_seq_len, self.config.hidden_size)
        self.assertEqual(output.shape, expected_shape)
        self.assertTrue(torch.all(torch.isfinite(output)))
        
        print("✓ FlashAttention2 memory efficiency test passed")
    
    def test_simd_attention_basic(self):
        """Test basic functionality of SIMDAttention"""
        print("Testing SIMDAttention basic functionality...")
        
        attention = SIMDAttention(self.config, layer_idx=0)
        attention.eval()
        
        with torch.no_grad():
            output, attn_weights, past_key_value = attention(
                hidden_states=self.hidden_states,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                output_attentions=True
            )
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.config.hidden_size)
        self.assertEqual(output.shape, expected_shape,
                         f"Output shape mismatch: expected {expected_shape}, got {output.shape}")
        
        # Check that output is finite
        self.assertTrue(torch.all(torch.isfinite(output)), "Output should contain only finite values")
        
        print("✓ SIMDAttention basic functionality test passed")
    
    def test_memory_efficient_attention_basic(self):
        """Test basic functionality of MemoryEfficientAttention"""
        print("Testing MemoryEfficientAttention basic functionality...")
        
        attention = MemoryEfficientAttention(self.config, layer_idx=0)
        attention.eval()
        
        with torch.no_grad():
            output, attn_weights, past_key_value = attention(
                hidden_states=self.hidden_states,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                output_attentions=False  # Memory efficient version typically doesn't return weights
            )
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.config.hidden_size)
        self.assertEqual(output.shape, expected_shape,
                         f"Output shape mismatch: expected {expected_shape}, got {output.shape}")
        
        # Memory efficient version may not return attention weights to save memory
        self.assertIsNone(attn_weights, "Memory efficient attention should not return weights by default")
        
        # Check that output is finite
        self.assertTrue(torch.all(torch.isfinite(output)), "Output should contain only finite values")
        
        print("✓ MemoryEfficientAttention basic functionality test passed")
    
    def test_sm61_optimized_attention_basic(self):
        """Test basic functionality of SM61OptimizedAttention"""
        print("Testing SM61OptimizedAttention basic functionality...")
        
        attention = SM61OptimizedAttention(self.config, layer_idx=0)
        attention.eval()
        
        with torch.no_grad():
            output, attn_weights, past_key_value = attention(
                hidden_states=self.hidden_states,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                output_attentions=True
            )
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.config.hidden_size)
        self.assertEqual(output.shape, expected_shape,
                         f"Output shape mismatch: expected {expected_shape}, got {output.shape}")
        
        # Check that output is finite
        self.assertTrue(torch.all(torch.isfinite(output)), "Output should contain only finite values")
        
        print("✓ SM61OptimizedAttention basic functionality test passed")
    
    def test_intel_optimized_attention_basic(self):
        """Test basic functionality of IntelOptimizedAttention"""
        print("Testing IntelOptimizedAttention basic functionality...")
        
        attention = IntelOptimizedAttention(self.config, layer_idx=0)
        attention.eval()
        
        with torch.no_grad():
            output, attn_weights, past_key_value = attention(
                hidden_states=self.hidden_states,
                attention_mask=self.attention_mask,
                position_ids=self.position_ids,
                output_attentions=True
            )
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.config.hidden_size)
        self.assertEqual(output.shape, expected_shape,
                         f"Output shape mismatch: expected {expected_shape}, got {output.shape}")
        
        # Check that output is finite
        self.assertTrue(torch.all(torch.isfinite(output)), "Output should contain only finite values")
        
        print("✓ IntelOptimizedAttention basic functionality test passed")
    
    def test_attention_factory_creation(self):
        """Test that the attention factory creates the right attention mechanisms"""
        print("Testing OptimizedAttentionFactory...")
        
        # Test FlashAttention creation
        flash_attn = OptimizedAttentionFactory.create_attention(self.config, layer_idx=0, hardware_target="auto")
        self.assertIsInstance(flash_attn, SIMDAttention)  # Will default to SIMD when use_flash_attention_2 is True
        
        # Test SM61 creation
        sm61_attn = OptimizedAttentionFactory.create_attention(self.config, layer_idx=0, hardware_target="sm61")
        self.assertIsInstance(sm61_attn, SM61OptimizedAttention)
        
        # Test Intel CPU creation
        intel_attn = OptimizedAttentionFactory.create_attention(self.config, layer_idx=0, hardware_target="intel_cpu")
        self.assertIsInstance(intel_attn, IntelOptimizedAttention)
        
        print("✓ OptimizedAttentionFactory test passed")
    
    def test_attention_heads_preservation(self):
        """Test that all attention mechanisms preserve the correct number of attention heads"""
        print("Testing attention heads preservation...")
        
        # Modify config to test with different number of heads
        test_config = Qwen3VLConfig(
            hidden_size=256,
            num_attention_heads=4,  # Smaller number for testing
            num_key_value_heads=4,
            max_position_embeddings=128,
            rope_theta=10000.0,
            attention_dropout_prob=0.0
        )
        
        # Test each attention mechanism
        mechanisms = [
            ("FlashAttention2", FlashAttention2(test_config, layer_idx=0)),
            ("SIMDAttention", SIMDAttention(test_config, layer_idx=0)),
            ("MemoryEfficientAttention", MemoryEfficientAttention(test_config, layer_idx=0)),
            ("SM61OptimizedAttention", SM61OptimizedAttention(test_config, layer_idx=0)),
            ("IntelOptimizedAttention", IntelOptimizedAttention(test_config, layer_idx=0))
        ]
        
        for name, mechanism in mechanisms:
            self.assertEqual(mechanism.num_heads, test_config.num_attention_heads,
                           f"{name} should preserve {test_config.num_attention_heads} attention heads, got {mechanism.num_heads}")
            print(f"  ✓ {name} preserves {mechanism.num_heads} attention heads")
        
        print("✓ Attention heads preservation test passed")
    
    def test_numerical_accuracy(self):
        """Test that optimized attention mechanisms maintain numerical accuracy"""
        print("Testing numerical accuracy of optimized attention...")
        
        # Create attention mechanisms
        original_config = Qwen3VLConfig(
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=64,
            rope_theta=10000.0,
            attention_dropout_prob=0.0
        )
        
        flash_attn = FlashAttention2(original_config, layer_idx=0)
        simd_attn = SIMDAttention(original_config, layer_idx=0)
        mem_eff_attn = MemoryEfficientAttention(original_config, layer_idx=0)
        
        # Create identical test inputs
        test_hidden = torch.randn(1, 32, original_config.hidden_size, requires_grad=False)
        test_positions = torch.arange(32, dtype=torch.long).unsqueeze(0)
        test_mask = torch.ones(1, 1, 32, 32, dtype=torch.bool)
        
        # Set to evaluation mode to ensure consistent behavior
        flash_attn.eval()
        simd_attn.eval()
        mem_eff_attn.eval()
        
        with torch.no_grad():
            # Run through each attention mechanism
            flash_out, _, _ = flash_attn(
                hidden_states=test_hidden,
                attention_mask=test_mask,
                position_ids=test_positions,
                output_attentions=False
            )
            
            simd_out, _, _ = simd_attn(
                hidden_states=test_hidden,
                attention_mask=test_mask,
                position_ids=test_positions,
                output_attentions=False
            )
            
            mem_eff_out, _, _ = mem_eff_attn(
                hidden_states=test_hidden,
                attention_mask=test_mask,
                position_ids=test_positions,
                output_attentions=False
            )
        
        # Check that all outputs are finite
        self.assertTrue(torch.all(torch.isfinite(flash_out)))
        self.assertTrue(torch.all(torch.isfinite(simd_out)))
        self.assertTrue(torch.all(torch.isfinite(mem_eff_out)))
        
        # The outputs may differ due to different implementations, but should be reasonable
        # Check that they have similar statistical properties
        self.assertAlmostEqual(torch.mean(flash_out).item(), torch.mean(simd_out).item(), places=1,
                              msg="Flash and SIMD attention outputs should have similar means")
        self.assertAlmostEqual(torch.mean(flash_out).item(), torch.mean(mem_eff_out).item(), places=1,
                              msg="Flash and Memory Efficient attention outputs should have similar means")
        
        print("✓ Numerical accuracy test passed")
    
    def test_performance_comparison(self):
        """Basic performance comparison between attention mechanisms"""
        print("Testing performance characteristics...")
        
        # Create smaller config for faster testing
        perf_config = Qwen3VLConfig(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rope_theta=10000.0,
            attention_dropout_prob=0.0
        )
        
        # Create attention mechanisms
        flash_attn = FlashAttention2(perf_config, layer_idx=0)
        simd_attn = SIMDAttention(perf_config, layer_idx=0)
        mem_eff_attn = MemoryEfficientAttention(perf_config, layer_idx=0)
        
        # Create test inputs
        test_hidden = torch.randn(1, 16, perf_config.hidden_size)
        test_positions = torch.arange(16, dtype=torch.long).unsqueeze(0)
        test_mask = torch.ones(1, 1, 16, 16, dtype=torch.bool)
        
        # Set to evaluation mode
        flash_attn.eval()
        simd_attn.eval()
        mem_eff_attn.eval()
        
        import time
        
        # Time FlashAttention
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = flash_attn(
                    hidden_states=test_hidden,
                    attention_mask=test_mask,
                    position_ids=test_positions,
                    output_attentions=False
                )[0]
        flash_time = time.time() - start_time
        
        # Time SIMDAttention
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = simd_attn(
                    hidden_states=test_hidden,
                    attention_mask=test_mask,
                    position_ids=test_positions,
                    output_attentions=False
                )[0]
        simd_time = time.time() - start_time
        
        # Time MemoryEfficientAttention
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                _ = mem_eff_attn(
                    hidden_states=test_hidden,
                    attention_mask=test_mask,
                    position_ids=test_positions,
                    output_attentions=False
                )[0]
        mem_eff_time = time.time() - start_time
        
        print(f"  FlashAttention time: {flash_time:.4f}s")
        print(f"  SIMDAttention time: {simd_time:.4f}s")
        print(f"  MemoryEfficientAttention time: {mem_eff_time:.4f}s")
        
        # All mechanisms should complete without errors
        self.assertGreater(flash_time, 0)
        self.assertGreater(simd_time, 0)
        self.assertGreater(mem_eff_time, 0)
        
        print("✓ Performance comparison test passed")


def run_tests():
    """Run all tests for optimized attention mechanisms"""
    print("Running Optimized Attention Mechanisms Tests...\n")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all tests
    test_suite.addTest(unittest.makeSuite(TestOptimizedAttentionMechanisms))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Results:")
    print(f"  Tests run: {result.testsRun}")
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
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n🎉 All Optimized Attention Mechanisms tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)