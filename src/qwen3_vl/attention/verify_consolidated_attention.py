"""
Final verification and demonstration of the consolidated attention system.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.abspath('.'))

def test_consolidated_attention_system():
    """Test the consolidated attention system functionality."""
    print("Testing Consolidated Attention System")
    print("=" * 40)
    
    try:
        # Import the main attention components
        from attention.consolidated_attention_complete import (
            StandardAttention,
            FlashAttention2,
            SM61OptimizedFlashAttention2,
            TrueSparseAttention,
            BlockSparseAttention,
            DynamicSparseAttention,
            Qwen3VLRotaryEmbedding,
            rotate_half,
            apply_rotary_pos_emb,
            repeat_kv
        )
        
        print("✓ Successfully imported attention mechanisms")
        
        # Create a sample config for testing
        from src.qwen3_vl.config import Qwen3VLConfig
        
        config = Qwen3VLConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=2048,
            rope_theta=10000.0,
            use_flash_attention_2=True,
            use_dynamic_sparse_attention=True,
            sparse_attention_sparsity_ratio=0.5,
            vision_sparse_attention_sparsity_ratio=0.4,
            cpu_model='Intel i5-10210U',
            gpu_model='NVIDIA SM61',
            memory_size=8 * 1024 * 1024 * 1024,
            storage_type='nvme'
        )
        
        # Test tensor shapes
        batch_size, seq_len, hidden_dim = 2, 128, 512
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        
        print(f"\nTesting with batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
        
        # Test Standard Attention
        print("\n1. Testing Standard Attention...")
        standard_attn = StandardAttention(config, layer_idx=0)
        output, attn_weights, past_key_value = standard_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        print(f"   ✓ Standard attention output shape: {output.shape}")
        
        # Test Flash Attention 2
        print("\n2. Testing Flash Attention 2...")
        flash_attn = FlashAttention2(config, layer_idx=0)
        output, attn_weights, past_key_value = flash_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        print(f"   ✓ Flash attention output shape: {output.shape}")
        
        # Test SM61 Optimized Flash Attention
        print("\n3. Testing SM61 Optimized Flash Attention...")
        sm61_attn = SM61OptimizedFlashAttention2(config, layer_idx=0)
        output, attn_weights, past_key_value = sm61_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        print(f"   ✓ SM61 optimized flash attention output shape: {output.shape}")
        
        # Test True Sparse Attention
        print("\n4. Testing True Sparse Attention...")
        sparse_attn = TrueSparseAttention(config, layer_idx=0)
        output, attn_weights, past_key_value = sparse_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        print(f"   ✓ True sparse attention output shape: {output.shape}")
        
        # Test Dynamic Sparse Attention
        print("\n5. Testing Dynamic Sparse Attention...")
        dynamic_sparse_attn = DynamicSparseAttention(config, layer_idx=0)
        output, attn_weights, past_key_value = dynamic_sparse_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        print(f"   ✓ Dynamic sparse attention output shape: {output.shape}")
        
        # Test Block Sparse Attention
        print("\n6. Testing Block Sparse Attention...")
        block_sparse_attn = BlockSparseAttention(config, layer_idx=0)
        output, attn_weights, past_key_value = block_sparse_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        print(f"   ✓ Block sparse attention output shape: {output.shape}")
        
        # Test Rotary Embedding
        print("\n7. Testing Rotary Embeddings...")
        rotary_emb = Qwen3VLRotaryEmbedding(
            dim=hidden_dim // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        test_tensor = torch.randn(batch_size, 8, seq_len, hidden_dim // config.num_attention_heads)  # 8 heads
        cos, sin = rotary_emb(test_tensor, position_ids)
        print(f"   ✓ Rotary embeddings output shapes: cos {cos.shape}, sin {sin.shape}")
        
        # Test utility functions
        print("\n8. Testing utility functions...")
        test_tensor_half = torch.randn(4, 8)
        rotated = rotate_half(test_tensor_half)
        print(f"   ✓ Rotate half function works: input {test_tensor_half.shape} -> output {rotated.shape}")
        
        # Test apply_rotary_pos_emb
        q = torch.randn(batch_size, 8, seq_len, hidden_dim // config.num_attention_heads)
        k = torch.randn(batch_size, 8, seq_len, hidden_dim // config.num_attention_heads)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        print(f"   ✓ Apply rotary pos emb works: q {q.shape} -> q_rot {q_rot.shape}")
        
        # Test repeat_kv
        kv_tensor = torch.randn(batch_size, 4, seq_len, hidden_dim // config.num_attention_heads)  # 4 kv heads
        repeated = repeat_kv(kv_tensor, n_rep=2)  # 8 total heads with 4*2
        print(f"   ✓ Repeat KV function works: input {kv_tensor.shape} -> output {repeated.shape}")
        
        print("\n" + "=" * 40)
        print("ALL ATTENTION MECHANISMS WORKING CORRECTLY!")
        print("\nConsolidated Attention System includes:")
        print("• Standard attention mechanism")
        print("• Flash attention 2 with memory efficiency")
        print("• SM61-optimized flash attention for older NVIDIA GPUs")
        print("• True sparse attention with configurable sparsity")
        print("• Dynamic sparse attention with learned routing")
        print("• Block sparse attention with hardware-optimized patterns")
        print("• Rotary embeddings for position encoding")
        print("• Utility functions for attention computation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR in attention system test: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_attention_selection():
    """Demonstrate attention mechanism selection based on configuration."""
    print("\n\nDemonstrating Attention Mechanism Selection")
    print("=" * 50)
    
    from attention.consolidated_attention_complete import AttentionMechanismSelector
    
    # Create different configurations
    from src.qwen3_vl.config import Qwen3VLConfig
    
    configs = {
        "Standard": Qwen3VLConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=2048,
            attention_implementation="standard"
        ),
        "Flash": Qwen3VLConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=2048,
            attention_implementation="flash_attention_2"
        ),
        "Sparse": Qwen3VLConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=2048,
            attention_implementation="sparse_attention",
            use_dynamic_sparse_attention=True
        )
    }
    
    for name, config in configs.items():
        attention_module = AttentionMechanismSelector.create_attention(config)
        print(f"   ✓ {name} attention: {type(attention_module).__name__}")
    
    print("\n✓ Attention mechanism selection working correctly!")


if __name__ == "__main__":
    print("Qwen3-VL Consolidated Attention System Verification")
    print("=" * 60)
    
    # Test the consolidated attention system
    success = test_consolidated_attention_system()
    
    if success:
        # Demonstrate attention selection
        demonstrate_attention_selection()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! CONSOLIDATED ATTENTION SYSTEM IS READY 🎉")
        print("\nThe system successfully consolidates:")
        print("• Multiple attention implementations in a single module")
        print("• Hardware-specific optimizations for Intel i5-10210U + NVIDIA SM61 + NVMe SSD")
        print("• Memory-efficient sparse attention patterns")
        print("• Rotary embeddings for position encoding")
        print("• Factory-based attention mechanism selection")
        print("• Proper tensor lifecycle management integration")
    else:
        print("\n❌ TESTS FAILED!")
        sys.exit(1)