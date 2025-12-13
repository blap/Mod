"""
Simple Usage Example for Advanced Predictive Tensor Lifecycle Management System

This example demonstrates basic usage of the system in a Qwen3-VL context.
"""

from src.qwen3_vl.memory_management.predictive_tensor_lifecycle_manager import (
    create_optimized_lifecycle_manager,
    TensorType,
    integrate_with_existing_systems
)
import torch


def main():
    print("Advanced Predictive Tensor Lifecycle Management - Usage Example")
    print("=" * 65)
    
    # Step 1: Create an optimized lifecycle manager for your hardware
    print("\n1. Creating optimized lifecycle manager...")
    lifecycle_manager = create_optimized_lifecycle_manager({
        'cpu_model': 'Intel i5-10210U',
        'gpu_model': 'NVIDIA SM61',
        'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
        'storage_type': 'nvme'
    })
    print("   ✓ Lifecycle manager created")
    
    # Step 2: Register tensors with lifecycle management
    print("\n2. Registering tensors with lifecycle management...")
    
    # Register a KV cache tensor (common in transformer models)
    kv_tensor = torch.randn(4, 1024, 64)  # batch, seq_len, head_dim
    kv_tensor_id = lifecycle_manager.register_tensor(
        kv_tensor,
        tensor_type=TensorType.KV_CACHE,
        is_pinned=False,
        initial_ref_count=2  # Two components will use this
    )
    print(f"   ✓ KV cache tensor registered with ID: {kv_tensor_id}")
    
    # Register an image features tensor
    image_features = torch.randn(1, 50, 768)  # batch, patches, features
    image_tensor_id = lifecycle_manager.register_tensor(
        image_features,
        tensor_type=TensorType.IMAGE_FEATURES,
        is_pinned=False,
        initial_ref_count=1
    )
    print(f"   ✓ Image features tensor registered with ID: {image_tensor_id}")
    
    # Step 3: Use tensors in your application (access recording)
    print("\n3. Using tensors in application...")
    
    # Access tensors during processing
    for i in range(3):
        lifecycle_manager.access_tensor(kv_tensor_id, context=f"attention_layer_{i}")
        lifecycle_manager.access_tensor(image_tensor_id, context="vision_encoder")
        print(f"   ✓ Accessed tensors in context iteration {i+1}")
    
    # Step 4: Manage references
    print("\n4. Managing tensor references...")
    
    # Component A is done with the KV tensor
    lifecycle_manager.decrement_reference(kv_tensor_id, "component_A")
    print("   ✓ Component A released reference to KV tensor")
    
    # Component B is also done with the KV tensor
    lifecycle_manager.decrement_reference(kv_tensor_id, "component_B")
    print("   ✓ Component B released reference to KV tensor")
    
    # Step 5: Check lifecycle statistics
    print("\n5. Checking lifecycle statistics...")
    stats = lifecycle_manager.get_tensor_lifecycle_stats()
    
    print(f"   Total tensors: {stats['total_tensors']}")
    print(f"   Pinned tensors: {stats['pinned_tensors']}")
    print(f"   In-use tensors: {stats['in_use_tensors']}")
    print(f"   Collections performed: {stats['collections_performed']}")
    print(f"   Tensors collected: {stats['tensors_collected']}")
    print(f"   Memory freed: {stats['memory_freed_bytes'] / (1024**2):.2f} MB")
    print(f"   Average lifetime prediction: {stats['average_lifetime_prediction']:.2f}s")
    
    # Step 6: Hardware-aware optimization
    print("\n6. Hardware-aware optimization...")
    
    # Optimize a tensor for matrix multiplication on your hardware
    large_tensor = torch.randn(512, 512)
    optimized_tensor = lifecycle_manager.optimize_tensor(
        large_tensor,
        tensor_type=TensorType.GENERAL,
        operation="matmul"
    )
    print(f"   ✓ Tensor optimized for {lifecycle_manager.hardware_manager.cpu_model}")
    
    # Determine optimal placement
    placement = lifecycle_manager.hardware_manager.optimize_tensor_placement(
        large_tensor,
        TensorType.GENERAL
    )
    print(f"   ✓ Optimal placement determined: {placement}")
    
    # Step 7: Integration with existing systems (conceptual)
    print("\n7. Integration with existing systems...")
    
    # In a real implementation, you would connect to actual systems like:
    # memory_tiering_system = get_tiering_system()  # From advanced_memory_tiering_system.py
    # compression_manager = get_compression_manager()  # From memory_compression_system.py
    # swapping_system = get_swapping_system()  # From advanced_memory_swapping_system.py
    
    # For this example, we'll show the integration pattern:
    print("   ✓ Integration pattern ready for existing systems")
    print("     - Memory tiering system connection")
    print("     - Compression manager connection") 
    print("     - Swapping system connection")
    
    # Step 8: Cleanup
    print("\n8. Cleanup...")
    lifecycle_manager.cleanup()
    print("   ✓ Lifecycle manager cleaned up")
    
    print("\n" + "=" * 65)
    print("Example completed successfully!")
    print("\nThe predictive tensor lifecycle management system provides:")
    print("- Proactive memory management with predictive collection")
    print("- Hardware-aware optimizations")
    print("- Integration with existing memory systems")
    print("- Detailed lifecycle statistics and monitoring")


def advanced_example():
    """More advanced usage example with custom configurations"""
    print("\nAdvanced Usage Example")
    print("=" * 22)
    
    # Create lifecycle manager with custom settings
    lifecycle_manager = create_optimized_lifecycle_manager({
        'cpu_model': 'Intel i5-10210U',
        'gpu_model': 'NVIDIA SM61',
        'memory_size': 16 * 1024 * 1024 * 1024,  # 16GB
        'storage_type': 'nvme'
    })
    
    # Create multiple tensors for a transformer layer
    batch_size, seq_len, hidden_dim = 2, 512, 768
    
    tensors = {
        'input_embeddings': torch.randn(batch_size, seq_len, hidden_dim),
        'attention_weights': torch.randn(batch_size, 8, seq_len, seq_len),  # 8 attention heads
        'ffn_intermediate': torch.randn(batch_size, seq_len, hidden_dim * 4),
        'layer_norm_input': torch.randn(batch_size, seq_len, hidden_dim)
    }
    
    tensor_ids = {}
    for name, tensor in tensors.items():
        tensor_type = TensorType.INTERMEDIATE if 'intermediate' in name else TensorType.GENERAL
        tensor_ids[name] = lifecycle_manager.register_tensor(
            tensor,
            tensor_type=tensor_type,
            is_pinned=False,
            initial_ref_count=1
        )
        print(f"   Registered {name}: {list(tensor.shape)}, type: {tensor_type.value}")
    
    # Simulate a forward pass with multiple access patterns
    for step in range(5):
        for name, tensor_id in tensor_ids.items():
            lifecycle_manager.access_tensor(tensor_id, context=f"forward_pass_{step}")
    
    # Check statistics
    stats = lifecycle_manager.get_tensor_lifecycle_stats()
    print(f"\n   Lifecycle stats after 5 forward passes:")
    print(f"   - Total tensors: {stats['total_tensors']}")
    print(f"   - Access count influence: {stats['average_lifetime_prediction']:.2f}s avg lifetime")
    
    lifecycle_manager.cleanup()


if __name__ == "__main__":
    main()
    advanced_example()