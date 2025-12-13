"""
Simple test to verify consolidated modules can be imported
"""
import sys
import os
# Add src to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from attention import (
    Qwen3VLAttention,
    FlashAttention2,
    DynamicSparseAttention,
    BlockSparseAttention,
    PerformerAttention,
    Qwen3VLRotaryEmbedding,
    apply_rotary_pos_emb,
    IntegratedTensorLifecycleManager
)
from qwen3_vl.config.config import Qwen3VLConfig


def test_imports():
    """Test that all consolidated modules can be imported without errors"""
    print("Testing consolidated module imports...")
    
    # Create a simple config for testing
    config = Qwen3VLConfig(
        hidden_size=512,
        num_attention_heads=32,  # Must be 32 to preserve full capacity
        num_key_value_heads=8,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        intermediate_size=2048,
        attention_dropout_prob=0.0,
        num_hidden_layers=32  # Must be 32 to preserve full capacity
    )
    
    # Test creating each attention mechanism
    print("Creating Qwen3VLAttention...")
    attention = Qwen3VLAttention(config)
    print("OK Qwen3VLAttention created successfully")

    print("Creating FlashAttention2...")
    flash_attention = FlashAttention2(config)
    print("OK FlashAttention2 created successfully")

    print("Creating DynamicSparseAttention...")
    sparse_attention = DynamicSparseAttention(config)
    print("OK DynamicSparseAttention created successfully")

    print("Creating BlockSparseAttention...")
    block_sparse_attention = BlockSparseAttention(config)
    print("OK BlockSparseAttention created successfully")

    print("Creating PerformerAttention...")
    performer_attention = PerformerAttention(config)
    print("OK PerformerAttention created successfully")

    print("Creating Rotary Embedding...")
    rotary_emb = Qwen3VLRotaryEmbedding(64)
    print("OK Rotary Embedding created successfully")

    print("Creating IntegratedTensorLifecycleManager...")
    lifecycle_manager = IntegratedTensorLifecycleManager({
        'cpu_model': 'Intel i5-10210U',
        'gpu_model': 'NVIDIA SM61',
        'memory_size': 8 * 1024 * 1024 * 1024,
        'storage_type': 'nvme'
    })
    print("OK IntegratedTensorLifecycleManager created successfully")
    
    print("\nAll consolidated modules imported and instantiated successfully!")


if __name__ == "__main__":
    test_imports()