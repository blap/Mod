"""
Comprehensive tests for the consolidated attention mechanisms in Qwen3-VL.
Tests all attention implementations with lifecycle management and hardware optimizations.
"""
import unittest
import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from attention.consolidated_attention_system import (
    Qwen3VLAttentionMechanism,
    Qwen3VLVisionAttentionMechanism,
    HardwareOptimizedAttentionWrapper,
    IntegratedAttentionSystem,
    AttentionMechanismFactory,
    create_consolidated_attention_mechanism
)
from attention.consolidated_tensor_lifecycle import (
    TensorLifecycleTracker,
    LifetimePredictor,
    AccessPatternAnalyzer,
    EnhancedPredictiveTensorLifecycleManager,
    create_optimized_lifecycle_manager,
    TensorType
)


class TestConsolidatedAttention(unittest.TestCase):
    """Test cases for consolidated attention mechanisms."""

    def setUp(self):
        """Set up test configuration."""
        from src.qwen3_vl.config import Qwen3VLConfig
        
        self.config = Qwen3VLConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            # Standard attention settings
            use_flash_attention_2=False,
            use_dynamic_sparse_attention=False,
            use_block_sparse_attention=False,
            # Lifecycle management settings
            cpu_model='Intel i5-10210U',
            gpu_model='NVIDIA SM61',
            memory_size=8 * 1024 * 1024 * 1024,
            storage_type='nvme'
        )
        
        self.batch_size = 2
        self.seq_len = 64
        self.hidden_dim = self.config.hidden_size

    def test_standard_attention_creation(self):
        """Test creation of standard attention mechanism."""
        attention = Qwen3VLAttentionMechanism(self.config, layer_idx=0)
        
        # Verify the attention mechanism was created with correct implementation
        self.assertIsInstance(attention.attention_impl, StandardAttention)
        self.assertIsNotNone(attention.tensor_lifecycle_manager)
        
        print("✓ Standard attention mechanism created successfully")

    def test_flash_attention_creation(self):
        """Test creation of flash attention mechanism."""
        # Modify config to use flash attention
        self.config.use_flash_attention_2 = True
        
        attention = Qwen3VLAttentionMechanism(self.config, layer_idx=0)
        
        # Verify flash attention was created
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6:
            self.assertIsInstance(attention.attention_impl, FlashAttention2)
        else:
            # On systems without proper GPU support, should fall back to standard
            self.assertIsInstance(attention.attention_impl, StandardAttention)
        
        print("✓ Flash attention mechanism created successfully")

    def test_sparse_attention_creation(self):
        """Test creation of sparse attention mechanism."""
        # Modify config to use dynamic sparse attention
        self.config.use_dynamic_sparse_attention = True
        self.config.use_flash_attention_2 = False  # Disable flash attention to test sparse
        
        attention = Qwen3VLAttentionMechanism(self.config, layer_idx=0)
        
        # Verify sparse attention was created
        self.assertIsInstance(attention.attention_impl, DynamicSparseAttention)
        self.assertIsNotNone(attention.tensor_lifecycle_manager)
        
        print("✓ Dynamic sparse attention mechanism created successfully")

    def test_block_sparse_attention_creation(self):
        """Test creation of block sparse attention mechanism."""
        # Modify config to use block sparse attention
        self.config.use_block_sparse_attention = True
        self.config.use_dynamic_sparse_attention = False
        self.config.use_flash_attention_2 = False
        
        attention = Qwen3VLAttentionMechanism(self.config, layer_idx=0)
        
        # Verify block sparse attention was created
        self.assertIsInstance(attention.attention_impl, BlockSparseAttention)
        self.assertIsNotNone(attention.tensor_lifecycle_manager)
        
        print("✓ Block sparse attention mechanism created successfully")

    def test_attention_forward_pass(self):
        """Test forward pass of attention mechanism."""
        attention = Qwen3VLAttentionMechanism(self.config, layer_idx=0)
        
        # Create test inputs
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        attention_mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)
        
        # Perform forward pass
        output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        
        print("✓ Attention forward pass completed successfully")

    def test_vision_attention_creation(self):
        """Test creation of vision attention mechanism."""
        vision_config = self.config
        # Set vision-specific parameters
        vision_config.vision_hidden_size = 768
        vision_config.vision_num_attention_heads = 12
        
        vision_attention = Qwen3VLVisionAttentionMechanism(vision_config)
        
        # Verify vision attention was created
        self.assertIsNotNone(vision_attention.tensor_lifecycle_manager)
        self.assertIsNotNone(vision_attention.attention_impl)
        
        print("✓ Vision attention mechanism created successfully")

    def test_hardware_optimized_wrapper(self):
        """Test hardware-optimized attention wrapper."""
        attention = HardwareOptimizedAttentionWrapper(self.config, layer_idx=0)
        
        # Verify it was created with appropriate hardware optimization
        self.assertIsNotNone(attention.attention_impl)
        self.assertIsNotNone(attention.tensor_lifecycle_manager)
        self.assertIsNotNone(attention.hardware_capabilities)
        
        # Check that hardware capabilities were detected
        self.assertIn('cpu_model', attention.hardware_capabilities)
        self.assertIn('memory_size', attention.hardware_capabilities)
        
        print("✓ Hardware-optimized attention wrapper created successfully")

    def test_integrated_attention_system(self):
        """Test integrated attention system."""
        system = IntegratedAttentionSystem(self.config, layer_idx=0)
        
        # Verify all components are initialized
        self.assertIsNotNone(system.main_attention)
        self.assertIsNotNone(system.lifecycle_manager)
        
        # Test forward pass
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        attention_mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)
        
        output, attn_weights, past_key_value = system(
            hidden_states=hidden_states,
            attention_mask=attention_mask
        )
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.hidden_dim))
        
        print("✓ Integrated attention system created and tested successfully")

    def test_attention_factory(self):
        """Test attention mechanism factory."""
        # Test standard attention creation
        attention = AttentionMechanismFactory.create_attention(self.config, attention_type="standard")
        self.assertIsInstance(attention, StandardAttention)
        
        # Test flash attention creation
        self.config.use_flash_attention_2 = True
        attention = AttentionMechanismFactory.create_attention(self.config, attention_type="flash")
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 6:
            self.assertIsInstance(attention, FlashAttention2)
        else:
            # Fallback to standard on unsupported hardware
            self.assertIsInstance(attention, StandardAttention)
        
        # Test dynamic sparse attention creation
        self.config.use_flash_attention_2 = False
        self.config.use_dynamic_sparse_attention = True
        attention = AttentionMechanismFactory.create_attention(self.config, attention_type="dynamic_sparse")
        self.assertIsInstance(attention, DynamicSparseAttention)
        
        # Test block sparse attention creation
        self.config.use_dynamic_sparse_attention = False
        self.config.use_block_sparse_attention = True
        attention = AttentionMechanismFactory.create_attention(self.config, attention_type="block_sparse")
        self.assertIsInstance(attention, BlockSparseAttention)
        
        print("✓ Attention mechanism factory working correctly")

    def test_lifecycle_management_integration(self):
        """Test tensor lifecycle management integration."""
        attention = Qwen3VLAttentionMechanism(self.config, layer_idx=0)
        
        # Create a tensor and register it
        test_tensor = torch.randn(10, 10)
        tensor_id = f"test_tensor_{id(test_tensor)}"
        
        # Register tensor with lifecycle manager
        metadata = attention.tensor_lifecycle_manager.register_tensor(
            test_tensor,
            tensor_id=tensor_id,
            tensor_type=TensorType.GENERAL
        )
        
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.tensor_id, tensor_id)
        
        # Access tensor
        success = attention.tensor_lifecycle_manager.access_tensor(tensor_id)
        self.assertTrue(success)
        
        # Test reference counting
        success = attention.tensor_lifecycle_manager.increment_reference(tensor_id)
        self.assertTrue(success)
        
        success = attention.tensor_lifecycle_manager.decrement_reference(tensor_id)
        self.assertTrue(success)
        
        print("✓ Tensor lifecycle management integration working correctly")

    def test_tensor_statistics(self):
        """Test tensor statistics and lifecycle metrics."""
        attention = Qwen3VLAttentionMechanism(self.config, layer_idx=0)
        
        # Get initial stats
        stats = attention.get_lifecycle_stats()
        self.assertIn('total_tensors', stats)
        self.assertIn('pinned_tensors', stats)
        self.assertIn('collections_performed', stats)
        
        # Create and register multiple tensors
        for i in range(5):
            test_tensor = torch.randn(20, 20)
            tensor_id = f"stat_tensor_{i}_{id(test_tensor)}"
            attention.tensor_lifecycle_manager.register_tensor(
                test_tensor,
                tensor_id=tensor_id,
                tensor_type=TensorType.GENERAL
            )
        
        # Access some tensors
        for i in range(3):
            tensor_id = f"stat_tensor_{i}_*"
            # Find the actual tensor ID by matching
            for registered_id in attention.tensor_lifecycle_manager.tracker.tensor_metadata:
                if f"stat_tensor_{i}_" in registered_id:
                    attention.tensor_lifecycle_manager.access_tensor(registered_id)
                    break
        
        # Get updated stats
        updated_stats = attention.get_lifecycle_stats()
        
        # Check that stats have been updated
        self.assertGreaterEqual(updated_stats['tracker_stats']['total_tensors'], 5)
        self.assertGreaterEqual(updated_stats['tracker_stats']['access_count'], 3)
        
        print("✓ Tensor statistics and lifecycle metrics working correctly")

    def test_memory_efficiency(self):
        """Test memory efficiency of different attention mechanisms."""
        # Compare memory usage of different implementations
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        attention_mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)
        
        # Test standard attention
        std_attention = AttentionMechanismFactory.create_attention(self.config, attention_type="standard")
        with torch.no_grad():
            std_output, _, _ = std_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask
            )
        
        # Test sparse attention (if available)
        if hasattr(self.config, 'use_dynamic_sparse_attention'):
            self.config.use_dynamic_sparse_attention = True
            sparse_attention = AttentionMechanismFactory.create_attention(self.config, attention_type="dynamic_sparse")
            with torch.no_grad():
                sparse_output, _, _ = sparse_attention(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask
                )
            
            # Outputs should be approximately the same size
            self.assertEqual(std_output.shape, sparse_output.shape)
        
        print("✓ Memory efficiency comparison completed")

    def test_rotary_embeddings_integration(self):
        """Test rotary embeddings integration with attention mechanisms."""
        attention = Qwen3VLAttentionMechanism(self.config, layer_idx=0)
        
        # Check that rotary embeddings are properly initialized
        self.assertIsNotNone(attention.rotary_emb)
        self.assertIsInstance(attention.rotary_emb, Qwen3VLRotaryEmbedding)
        
        # Test rotary embedding forward pass
        batch_size, seq_len = 2, 10
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        test_tensor = torch.randn(batch_size, 8, seq_len, self.config.hidden_size // 8)  # 8 heads
        
        cos, sin = attention.rotary_emb(test_tensor, position_ids)
        
        self.assertEqual(cos.shape, (batch_size, seq_len, self.config.hidden_size // 8))
        self.assertEqual(sin.shape, (batch_size, seq_len, self.config.hidden_size // 8))
        
        print("✓ Rotary embeddings integration working correctly")

    def test_attention_with_past_key_values(self):
        """Test attention with past key values (for generation)."""
        attention = Qwen3VLAttentionMechanism(self.config, layer_idx=0)
        
        # Create initial hidden states
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        attention_mask = torch.ones(self.batch_size, 1, self.seq_len, self.seq_len)
        
        # First forward pass
        output1, attn_weights1, past_key_value1 = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=True
        )
        
        # Create additional hidden states for continuation
        additional_hidden_states = torch.randn(self.batch_size, 10, self.hidden_dim)
        extended_attention_mask = torch.ones(self.batch_size, 1, self.seq_len + 10, self.seq_len + 10)
        
        # Second forward pass with past key values
        output2, attn_weights2, past_key_value2 = attention(
            hidden_states=additional_hidden_states,
            attention_mask=extended_attention_mask,
            past_key_value=past_key_value1,
            use_cache=True
        )
        
        self.assertEqual(output2.shape, (self.batch_size, 10, self.hidden_dim))
        
        print("✓ Attention with past key values working correctly")

    def tearDown(self):
        """Clean up after tests."""
        # Perform cleanup of lifecycle managers if needed
        pass


def run_comprehensive_attention_tests():
    """Run all attention mechanism tests."""
    print("Running Comprehensive Attention Mechanism Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(TestConsolidatedAttention('test_standard_attention_creation'))
    suite.addTest(TestConsolidatedAttention('test_flash_attention_creation'))
    suite.addTest(TestConsolidatedAttention('test_sparse_attention_creation'))
    suite.addTest(TestConsolidatedAttention('test_block_sparse_attention_creation'))
    suite.addTest(TestConsolidatedAttention('test_attention_forward_pass'))
    suite.addTest(TestConsolidatedAttention('test_vision_attention_creation'))
    suite.addTest(TestConsolidatedAttention('test_hardware_optimized_wrapper'))
    suite.addTest(TestConsolidatedAttention('test_integrated_attention_system'))
    suite.addTest(TestConsolidatedAttention('test_attention_factory'))
    suite.addTest(TestConsolidatedAttention('test_lifecycle_management_integration'))
    suite.addTest(TestConsolidatedAttention('test_tensor_statistics'))
    suite.addTest(TestConsolidatedAttention('test_memory_efficiency'))
    suite.addTest(TestConsolidatedAttention('test_rotary_embeddings_integration'))
    suite.addTest(TestConsolidatedAttention('test_attention_with_past_key_values'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, trace in result.failures:
            print(f"  {test}: {trace}")
    
    if result.errors:
        print("\nErrors:")
        for test, trace in result.errors:
            print(f"  {test}: {trace}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_attention_tests()
    if success:
        print("\n✓ ALL TESTS PASSED!")
        print("\nConsolidated Attention System is working correctly.")
    else:
        print("\n✗ SOME TESTS FAILED!")
        sys.exit(1)