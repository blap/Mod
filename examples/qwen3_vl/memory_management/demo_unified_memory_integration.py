"""
Demonstration: Updating Existing Component to Use Unified Memory Manager

This module demonstrates how an existing component that previously used separate
memory management systems can be updated to use the new unified memory manager.
"""

import torch
from unified_memory_manager import UnifiedMemoryManager, UnifiedTensorType
from hardware_abstraction_layer import HardwareManager


class LegacyVisionLanguageModel:
    """
    A simplified example of a vision-language model that previously used
    separate memory management systems.
    """
    
    def __init__(self):
        # OLD WAY: Initialize separate memory management systems
        # self.memory_pool = SomeMemoryPoolSystem()
        # self.compression_system = SomeCompressionSystem()
        # self.tiering_system = SomeTieringSystem()
        # self.swapping_system = SomeSwappingSystem()
        
        # NEW WAY: Use unified memory manager
        self.hw_manager = HardwareManager()
        self.memory_manager = UnifiedMemoryManager(
            kv_cache_pool_size=512*1024*1024,  # 512MB
            image_features_pool_size=1*1024*1024*1024,  # 1GB
            text_embeddings_pool_size=512*1024*1024,  # 512MB
        )
        
        # Integrate with hardware abstraction layer
        self.memory_manager.integrate_with_hardware_abstraction(self.hw_manager)
        
        # Track tensor IDs for the model components
        self.tensor_ids = {}
        
    def process_image_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process image features using unified memory management.
        """
        print("Processing image features with unified memory management...")
        
        # OLD WAY:
        # raw_features = self.memory_pool.allocate(image_tensor.numel() * 4, 'image_features')
        # compressed_features = self.compression_system.compress(image_tensor)
        # self.tiering_system.store_in_gpu(compressed_features)
        
        # NEW WAY: Use unified allocation
        tensor_size = image_tensor.numel() * image_tensor.element_size()
        unified_block = self.memory_manager.alloc_image_features(
            tensor_size, 
            tensor_id="image_features_1",
            pinned=False
        )
        
        if unified_block:
            print(f"  Allocated {unified_block.size_bytes} bytes for image features")
            print(f"  Tensor placed in tier: {unified_block.tier.value if unified_block.tier else 'N/A'}")
            print(f"  Compression enabled: {unified_block.is_compressed}")
            
            # Store the tensor ID for later access
            self.tensor_ids['image_features'] = "image_features_1"
            
            # Return the original tensor (in a real system, this would work with the unified tensor)
            return image_tensor
        else:
            print("  Failed to allocate memory for image features")
            return None
    
    def process_text_embeddings(self, text_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process text embeddings using unified memory management.
        """
        print("Processing text embeddings with unified memory management...")
        
        tensor_size = text_tensor.numel() * text_tensor.element_size()
        unified_block = self.memory_manager.alloc_text_embeddings(
            tensor_size,
            tensor_id="text_embeddings_1",
            pinned=False
        )
        
        if unified_block:
            print(f"  Allocated {unified_block.size_bytes} bytes for text embeddings")
            print(f"  Tensor placed in tier: {unified_block.tier.value if unified_block.tier else 'N/A'}")
            print(f"  Compression enabled: {unified_block.is_compressed}")
            
            # Store the tensor ID for later access
            self.tensor_ids['text_embeddings'] = "text_embeddings_1"
            
            return text_tensor
        else:
            print("  Failed to allocate memory for text embeddings")
            return None
    
    def manage_kv_cache(self, kv_tensor: torch.Tensor) -> torch.Tensor:
        """
        Manage KV cache using unified memory management.
        """
        print("Managing KV cache with unified memory management...")
        
        tensor_size = kv_tensor.numel() * kv_tensor.element_size()
        unified_block = self.memory_manager.alloc_kv_cache(
            tensor_size,
            tensor_id="kv_cache_1",
            pinned=True  # KV cache typically needs to be pinned for performance
        )
        
        if unified_block:
            print(f"  Allocated {unified_block.size_bytes} bytes for KV cache")
            print(f"  Tensor placed in tier: {unified_block.tier.value if unified_block.tier else 'N/A'}")
            print(f"  Pinned: {unified_block.pinned}")
            
            # Store the tensor ID for later access
            self.tensor_ids['kv_cache'] = "kv_cache_1"
            
            return kv_tensor
        else:
            print("  Failed to allocate memory for KV cache")
            return None
    
    def access_tensor(self, tensor_name: str) -> torch.Tensor:
        """
        Access a tensor using unified memory management.
        """
        if tensor_name not in self.tensor_ids:
            print(f"Tensor {tensor_name} not found")
            return None
        
        tensor_id = self.tensor_ids[tensor_name]
        print(f"Accessing tensor '{tensor_name}' (ID: {tensor_id})...")
        
        # Use unified access method
        tensor = self.memory_manager.access_tensor(tensor_id)
        
        if tensor is not None:
            print(f"  Successfully accessed tensor from unified memory system")
            return tensor
        else:
            print(f"  Failed to access tensor '{tensor_name}'")
            return None
    
    def get_memory_stats(self):
        """
        Get unified memory statistics.
        """
        print("\nUnified Memory System Statistics:")
        stats = self.memory_manager.get_system_stats()
        
        print(f"  Total allocations: {stats['unified_stats']['total_allocations']}")
        print(f"  Total deallocations: {stats['unified_stats']['total_deallocations']}")
        print(f"  Active tensors: {stats['active_tensors']}")
        print(f"  Current memory usage: {stats['unified_stats']['current_memory_usage'] / (1024**2):.2f} MB")
        
        # Get utilization breakdown
        utilization = self.memory_manager.get_memory_utilization()
        print(f"  Overall utilization: {utilization['overall_utilization']:.2%}")
        print(f"  GPU utilization: {utilization['gpu_utilization']:.2%}")
        print(f"  CPU utilization: {utilization['cpu_utilization']:.2%}")
        print(f"  SSD utilization: {utilization['ssd_utilization']:.2%}")
        print(f"  Average fragmentation: {utilization['average_fragmentation']:.2%}")
    
    def cleanup(self):
        """
        Clean up allocated tensors.
        """
        print("\nCleaning up allocated tensors...")
        
        for name, tensor_id in self.tensor_ids.items():
            success = self.memory_manager.deallocate(tensor_id)
            print(f"  Deallocated {name} (ID: {tensor_id}): {success}")
        
        self.tensor_ids.clear()


def demonstrate_unified_memory_usage():
    """
    Demonstrate how the unified memory manager can replace separate systems
    in an existing component.
    """
    print("Demonstrating Unified Memory Manager Integration")
    print("=" * 50)
    
    # Create the legacy model with unified memory management
    model = LegacyVisionLanguageModel()
    
    # Create some sample tensors
    image_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)  # Image features
    text_tensor = torch.randn(1, 512, 768, dtype=torch.float32)       # Text embeddings  
    kv_tensor = torch.randn(1, 12, 1024, 64, dtype=torch.float32)    # KV cache
    
    print(f"Created sample tensors:")
    print(f"  Image tensor: {image_tensor.shape}, {image_tensor.numel() * 4:,} bytes")
    print(f"  Text tensor: {text_tensor.shape}, {text_tensor.numel() * 4:,} bytes")
    print(f"  KV tensor: {kv_tensor.shape}, {kv_tensor.numel() * 4:,} bytes")
    
    # Process tensors using unified memory management
    processed_image = model.process_image_features(image_tensor)
    processed_text = model.process_text_embeddings(text_tensor)
    processed_kv = model.manage_kv_cache(kv_tensor)
    
    # Access tensors
    accessed_image = model.access_tensor('image_features')
    accessed_text = model.access_tensor('text_embeddings')
    accessed_kv = model.access_tensor('kv_cache')
    
    # Show memory statistics
    model.get_memory_stats()
    
    # Show optimization recommendations
    print("\nOptimization Recommendations:")
    for name, tensor_id in model.tensor_ids.items():
        recommendations = model.memory_manager.get_optimization_recommendations(tensor_id)
        if recommendations:
            print(f"  {name}: {', '.join(recommendations)}")
        else:
            print(f"  {name}: No specific recommendations")
    
    # Clean up
    model.cleanup()
    
    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    demonstrate_unified_memory_usage()