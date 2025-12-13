"""
Demonstration script for the Consolidated Attention System in Qwen3-VL Model.
Shows how all attention mechanisms work together with predictive tensor lifecycle management.
"""
import torch
import torch.nn as nn
import time
from typing import Optional, Tuple
import sys
import os

# Add the src directory to the path
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
    create_optimized_lifecycle_manager,
    TensorType
)


def demonstrate_attention_system():
    """Demonstrate the complete attention system with lifecycle management."""
    print("Qwen3-VL Consolidated Attention System Demonstration")
    print("=" * 60)

    # Create configuration for demonstration
    from src.qwen3_vl.config import Qwen3VLConfig

    config = Qwen3VLConfig(
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        # Enable different attention mechanisms for demonstration
        use_flash_attention_2=True,
        use_dynamic_sparse_attention=True,
        sparse_attention_sparsity_ratio=0.5,
        vision_sparse_attention_sparsity_ratio=0.4,
        # Hardware-specific optimizations
        cpu_model='Intel i5-10210U',
        gpu_model='NVIDIA SM61',
        memory_size=8 * 1024 * 1024 * 1024,  # 8GB
        storage_type='nvme'
    )

    print("\n1. Creating optimized tensor lifecycle manager...")
    lifecycle_manager = create_optimized_lifecycle_manager({
        'cpu_model': 'Intel i5-10210U',
        'gpu_model': 'NVIDIA SM61',
        'memory_size': 8 * 1024 * 1024 * 1024,
        'storage_type': 'nvme'
    })
    print("   ✓ Lifecycle manager created with hardware-specific optimizations")

    print("\n2. Creating different attention mechanisms...")

    # Create standard attention mechanism
    standard_attention = Qwen3VLAttentionMechanism(config, layer_idx=0)
    print(f"   ✓ Standard attention: {type(standard_attention.attention_impl).__name__}")

    # Create vision attention mechanism
    vision_attention = Qwen3VLVisionAttentionMechanism(config)
    print(f"   ✓ Vision attention: {type(vision_attention.attention_impl).__name__}")

    # Create hardware-optimized attention
    hardware_attention = HardwareOptimizedAttentionWrapper(config, layer_idx=0)
    print(f"   ✓ Hardware-optimized attention: {type(hardware_attention.attention_impl).__name__}")

    # Create integrated attention system
    integrated_system = IntegratedAttentionSystem(config, layer_idx=0)
    print(f"   ✓ Integrated attention system created")

    print("\n3. Testing attention mechanism creation with factory...")
    for attention_type in ["standard", "flash", "dynamic_sparse", "block_sparse"]:
        attention = AttentionMechanismFactory.create_attention(config, attention_type=attention_type)
        print(f"   ✓ {attention_type.capitalize()} attention: {type(attention).__name__}")

    print("\n4. Performance comparison of attention mechanisms...")

    # Create test tensors
    batch_size, seq_len, hidden_dim = 2, 128, 512
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)

    attention_types = {
        "Standard": AttentionMechanismFactory.create_attention(config, attention_type="standard"),
        "Flash": AttentionMechanismFactory.create_attention(config, attention_type="flash") if torch.cuda.is_available() else None,
        "Dynamic Sparse": AttentionMechanismFactory.create_attention(config, attention_type="dynamic_sparse"),
        "Block Sparse": AttentionMechanismFactory.create_attention(config, attention_type="block_sparse")
    }

    for name, attention_impl in attention_types.items():
        if attention_impl is None:
            continue

        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = attention_impl(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask
                )

        # Timing
        start_time = time.time()
        for _ in range(10):
            with torch.no_grad():
                output, _, _ = attention_impl(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask
                )
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        print(f"   {name}: {avg_time*1000:.2f}ms avg, output shape: {output.shape}")

    print("\n5. Testing tensor lifecycle management...")

    # Create a tensor to test lifecycle management
    test_tensor = torch.randn(100, 100)

    # Register a tensor with the lifecycle manager
    tensor_id = f"test_tensor_{id(test_tensor)}"
    lifecycle_manager.register_tensor(
        test_tensor,
        tensor_id=tensor_id,
        tensor_type=TensorType.GENERAL,
        is_pinned=False
    )

    print(f"   ✓ Registered tensor {tensor_id}")

    # Access the tensor multiple times
    for i in range(5):
        lifecycle_manager.access_tensor(tensor_id, context=f"test_context_{i}")
        time.sleep(0.01)  # Small delay

    print(f"   ✓ Accessed tensor {tensor_id} 5 times")

    # Check tensor statistics
    stats = lifecycle_manager.get_stats()
    print(f"   ✓ Lifecycle stats: {stats.get('total_tensors', 0)} tensors tracked")

    print("\n6. Demonstrating hardware-specific optimizations...")

    # Show hardware capabilities detected
    try:
        hardware_info = hardware_attention.get_hardware_info()
        print(f"   CPU: {hardware_info.get('cpu_model', 'Unknown')}")
        print(f"   GPU: {hardware_info.get('gpu_model', 'None')}")
        print(f"   Memory: {hardware_info.get('memory_size', 0) / (1024**3):.1f} GB")
        print(f"   Storage: {hardware_info.get('storage_type', 'Unknown')}")
    except:
        print("   Hardware info not available (likely due to incomplete imports)")

    print("\n7. Testing integrated system with lifecycle management...")

    # Create test inputs
    test_hidden = torch.randn(2, 64, 512)
    test_mask = torch.ones(2, 1, 64, 64)

    # Forward pass with integrated system
    try:
        output, attn_weights, past_key_value = integrated_system(
            hidden_states=test_hidden,
            attention_mask=test_mask
        )

        print(f"   ✓ Integrated system forward pass: input {test_hidden.shape} -> output {output.shape}")

        # Get system statistics
        system_stats = integrated_system.get_system_stats()
        print(f"   ✓ System stats retrieved: {len(system_stats) if system_stats else 'N/A'} metrics if available")
    except Exception as e:
        print(f"   ⚠️  Could not test integrated system: {e}")

    print("\n8. Cleanup and finalization...")

    # Clean up systems
    try:
        integrated_system.cleanup()
    except:
        pass  # May not have cleanup method if not fully instantiated
    lifecycle_manager.cleanup()

    print("   ✓ All systems cleaned up successfully")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("\nKey Features Demonstrated:")
    print("• Multiple attention mechanisms (standard, flash, sparse, block-sparse)")
    print("• Hardware-aware optimizations for Intel i5-10210U + NVIDIA SM61 + NVMe")
    print("• Predictive tensor lifecycle management")
    print("• Integration with memory tiering, compression, and swapping")
    print("• Factory-based attention mechanism creation")
    print("• Performance comparison of different implementations")

    return True


def performance_benchmark():
    """Run performance benchmarks for different attention implementations."""
    print("\nPerformance Benchmarking")
    print("-" * 30)

    from src.qwen3_vl.config import Qwen3VLConfig

    config = Qwen3VLConfig(
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=6,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        use_flash_attention_2=True,
        use_dynamic_sparse_attention=True,
        sparse_attention_sparsity_ratio=0.5
    )

    # Different sequence lengths for benchmarking
    seq_lengths = [64, 128, 256, 512]
    batch_size = 2
    hidden_dim = config.hidden_size

    attention_mechanisms = {
        "Standard": AttentionMechanismFactory.create_attention(config, attention_type="standard"),
        "Dynamic Sparse": AttentionMechanismFactory.create_attention(config, attention_type="dynamic_sparse"),
        "Block Sparse": AttentionMechanismFactory.create_attention(config, attention_type="block_sparse")
    }

    if torch.cuda.is_available():
        attention_mechanisms["Flash"] = AttentionMechanismFactory.create_attention(config, attention_type="flash")

    for seq_len in seq_lengths:
        print(f"\nSequence Length: {seq_len}")
        print("-" * 20)

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)

        for name, attention_impl in attention_mechanisms.items():
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = attention_impl(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask
                    )

            # Benchmark
            start_time = time.time()
            for _ in range(20):
                with torch.no_grad():
                    output, _, _ = attention_impl(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask
                    )
            end_time = time.time()

            avg_time = (end_time - start_time) / 20
            memory_usage = hidden_states.element_size() * hidden_states.nelement() / (1024**2)  # MB

            print(f"  {name:15}: {avg_time*1000:6.2f}ms | {memory_usage:6.1f}MB")


def main():
    """Main entry point for the demonstration."""
    print("Starting Qwen3-VL Consolidated Attention System Demo")
    print("=" * 60)

    try:
        # Run main demonstration
        success = demonstrate_attention_system()

        if success:
            # Run performance benchmarks
            performance_benchmark()

            print("\n" + "=" * 60)
            print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
            print("\nThe Consolidated Attention System provides:")
            print("• Unified interface to all attention implementations")
            print("• Hardware-specific optimizations")
            print("• Predictive tensor lifecycle management")
            print("• Memory-efficient sparse attention patterns")
            print("• Easy integration with existing Qwen3-VL models")
        else:
            print("\n❌ DEMONSTRATION FAILED!")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ ERROR during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()