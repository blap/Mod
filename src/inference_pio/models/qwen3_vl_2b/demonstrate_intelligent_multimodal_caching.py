"""
Demonstration of Intelligent Multimodal Caching for Qwen3-VL-2B Model

This module demonstrates the implementation and usage of the intelligent multimodal caching system
specifically designed for the Qwen3-VL-2B model. It shows how the system optimizes caching for both
text and image modalities with adaptive strategies based on access patterns and content similarity.
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from io import BytesIO
import time
import hashlib
from typing import Dict, Any, Optional, Tuple

from inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from inference_pio.common.intelligent_multimodal_caching import (
    Qwen3VL2BIntelligentCachingManager,
    CacheEvictionPolicy,
    CacheEntryType
)


def demonstrate_intelligent_multimodal_caching():
    """
    Demonstrate the intelligent multimodal caching system for Qwen3-VL-2B model.
    """
    print("="*80)
    print("INTELLIGENT MULTIMODAL CACHING DEMONSTRATION FOR QWEN3-VL-2B MODEL")
    print("="*80)
    
    # Create configuration with caching enabled
    config = Qwen3VL2BConfig()
    config.enable_intelligent_multimodal_caching = True
    config.intelligent_multimodal_cache_size_gb = 0.5  # 500MB cache for demonstration
    config.intelligent_multimodal_cache_eviction_policy = "predictive"
    config.intelligent_multimodal_cache_enable_similarity = True
    config.intelligent_multimodal_cache_similarity_threshold = 0.85
    config.intelligent_multimodal_cache_enable_ttl = True
    config.intelligent_multimodal_cache_default_ttl = 3600.0  # 1 hour
    config.intelligent_multimodal_cache_enable_compression = True
    config.intelligent_multimodal_cache_compression_ratio = 0.6
    
    print(f"1. Created Qwen3-VL-2B configuration with caching enabled")
    print(f"   - Cache size: {config.intelligent_multimodal_cache_size_gb}GB")
    print(f"   - Eviction policy: {config.intelligent_multimodal_cache_eviction_policy}")
    print(f"   - Similarity caching: {config.intelligent_multimodal_cache_enable_similarity}")
    print(f"   - Similarity threshold: {config.intelligent_multimodal_cache_similarity_threshold}")
    print(f"   - TTL enabled: {config.intelligent_multimodal_cache_enable_ttl}")
    print(f"   - Compression enabled: {config.intelligent_multimodal_cache_enable_compression}")
    
    # Create caching manager
    caching_manager = Qwen3VL2BIntelligentCachingManager(
        cache_size_gb=config.intelligent_multimodal_cache_size_gb,
        eviction_policy=CacheEvictionPolicy.PREDICTIVE,
        enable_similarity_caching=config.intelligent_multimodal_cache_enable_similarity,
        similarity_threshold=config.intelligent_multimodal_cache_similarity_threshold,
        enable_ttl=config.intelligent_multimodal_cache_enable_ttl,
        default_ttl=config.intelligent_multimodal_cache_default_ttl,
        enable_compression=config.intelligent_multimodal_cache_enable_compression,
        compression_ratio=config.intelligent_multimodal_cache_compression_ratio
    )
    
    print(f"\n2. Created intelligent caching manager with predictive policy")
    print(f"   - Max cache size: {caching_manager.cache.max_size_bytes / (1024**3):.2f}GB")
    print(f"   - Similarity threshold: {caching_manager.cache.similarity_threshold}")
    print(f"   - Compression enabled: {caching_manager.cache.enable_compression}")
    
    # Create sample text and image data
    sample_texts = [
        "Describe this image in detail.",
        "What objects do you see in this picture?",
        "Analyze the content of this image.",
        "Summarize what is happening in this scene.",
        "Identify the main subject in this photo."
    ]
    
    sample_images = [
        Image.new('RGB', (224, 224), color='red'),
        Image.new('RGB', (224, 224), color='blue'),
        Image.new('RGB', (224, 224), color='green'),
        Image.new('RGB', (224, 224), color='yellow'),
        Image.new('RGB', (224, 224), color='purple')
    ]
    
    print(f"\n3. Created sample data for caching demonstration")
    print(f"   - {len(sample_texts)} sample texts")
    print(f"   - {len(sample_images)} sample images")
    
    # Demonstrate text caching
    print(f"\n4. Demonstrating text caching...")
    text_tensors = []
    for i, text in enumerate(sample_texts):
        tensor = torch.randn(1, 10, config.hidden_size)  # Simulated processed text tensor
        text_tensors.append(tensor)
        
        start_time = time.time()
        caching_manager.cache_text_input(text, tensor, priority=0.5 + i*0.1)  # Increasing priority
        cache_time = time.time() - start_time
        
        print(f"   - Cached text {i+1}: '{text[:30]}...' (size: {tensor.numel()*tensor.element_size()/(1024**2):.2f}MB, time: {cache_time*1000:.2f}ms)")
    
    # Demonstrate image caching
    print(f"\n5. Demonstrating image caching...")
    image_tensors = []
    for i, image in enumerate(sample_images):
        tensor = torch.randn(1, 197, config.hidden_size)  # Simulated processed image tensor (patch embeddings)
        image_tensors.append(tensor)
        
        start_time = time.time()
        caching_manager.cache_image_input(image, tensor, priority=0.5 + i*0.1)  # Increasing priority
        cache_time = time.time() - start_time
        
        print(f"   - Cached image {i+1}: {image.size} RGB (size: {tensor.numel()*tensor.element_size()/(1024**2):.2f}MB, time: {cache_time*1000:.2f}ms)")
    
    # Demonstrate text-image pair caching
    print(f"\n6. Demonstrating text-image pair caching...")
    pair_data = []
    for i, (text, image) in enumerate(zip(sample_texts[:3], sample_images[:3])):
        pair_output = {
            'text_features': torch.randn(1, 10, config.hidden_size),
            'image_features': torch.randn(1, 197, config.hidden_size),
            'fused_features': torch.randn(1, 207, config.hidden_size)
        }
        pair_data.append(pair_output)
        
        start_time = time.time()
        caching_manager.cache_text_image_pair(text, image, pair_output, priority=0.7)
        cache_time = time.time() - start_time
        
        print(f"   - Cached pair {i+1}: text+image (size: ~{(sum(t.numel()*t.element_size() for t in pair_output.values())/(1024**2)):.2f}MB, time: {cache_time*1000:.2f}ms)")
    
    # Demonstrate retrieval of cached text
    print(f"\n7. Demonstrating retrieval of cached text...")
    for i, text in enumerate(sample_texts[:2]):
        start_time = time.time()
        retrieved_tensor = caching_manager.get_cached_text_input(text)
        retrieval_time = time.time() - start_time
        
        if retrieved_tensor is not None:
            print(f"   - Retrieved text {i+1}: '{text[:30]}...' (time: {retrieval_time*1000:.2f}ms, match: OK)")
        else:
            print(f"   - Retrieved text {i+1}: '{text[:30]}...' (time: {retrieval_time*1000:.2f}ms, match: NOT FOUND)")
    
    # Demonstrate retrieval of cached image
    print(f"\n8. Demonstrating retrieval of cached images...")
    for i, image in enumerate(sample_images[:2]):
        start_time = time.time()
        retrieved_tensor = caching_manager.get_cached_image_input(image)
        retrieval_time = time.time() - start_time
        
        if retrieved_tensor is not None:
            print(f"   - Retrieved image {i+1}: {image.size} RGB (time: {retrieval_time*1000:.2f}ms, match: OK)")
        else:
            print(f"   - Retrieved image {i+1}: {image.size} RGB (time: {retrieval_time*1000:.2f}ms, match: NOT FOUND)")
    
    # Demonstrate similarity search for text
    print(f"\n9. Demonstrating similarity search for text...")
    for i, text in enumerate(sample_texts[:2]):
        start_time = time.time()
        similar_result = caching_manager.find_similar_text(text)
        similarity_time = time.time() - start_time
        
        if similar_result is not None:
            key, cached_data = similar_result
            print(f"   - Found similar text {i+1}: '{text[:30]}...' (time: {similarity_time*1000:.2f}ms, key: {key[:10]}...)")
        else:
            print(f"   - No similar text found for: '{text[:30]}...' (time: {similarity_time*1000:.2f}ms)")
    
    # Demonstrate similarity search for images
    print(f"\n10. Demonstrating similarity search for images...")
    for i, image in enumerate(sample_images[:2]):
        start_time = time.time()
        similar_result = caching_manager.find_similar_image(image)
        similarity_time = time.time() - start_time
        
        if similar_result is not None:
            key, cached_data = similar_result
            print(f"   - Found similar image {i+1}: {image.size} RGB (time: {similarity_time*1000:.2f}ms, key: {key[:10]}...)")
        else:
            print(f"   - No similar image found for: {image.size} RGB (time: {similarity_time*1000:.2f}ms)")
    
    # Show cache statistics
    print(f"\n11. Cache statistics:")
    stats = caching_manager.get_cache_stats()
    print(f"   - Total entries: {stats['total_entries']}")
    print(f"   - Active entries: {stats['active_entries']}")
    print(f"   - Expired entries: {stats['expired_entries']}")
    print(f"   - Current size: {stats['current_size_bytes']/(1024**2):.2f}MB")
    print(f"   - Max size: {stats['max_size_bytes']/(1024**2):.2f}MB")
    print(f"   - Usage percentage: {stats['usage_percentage']:.2f}%")
    print(f"   - Average access frequency: {stats['average_access_frequency']:.2f}")
    print(f"   - Eviction policy: {stats['eviction_policy']}")
    print(f"   - Compression enabled: {stats['compression_enabled']}")
    print(f"   - Similarity caching enabled: {stats['similarity_caching_enabled']}")
    
    # Show model-specific breakdown
    print(f"\n12. Model-specific cache breakdown:")
    print(f"   - Text cache entries: {stats['text_cache_entries']}")
    print(f"   - Image cache entries: {stats['image_cache_entries']}")
    print(f"   - Vision encoder cache entries: {stats['vision_encoder_cache_entries']}")
    print(f"   - Language encoder cache entries: {stats['language_encoder_cache_entries']}")
    print(f"   - Cross-modal cache entries: {stats['cross_modal_cache_entries']}")
    print(f"   - Fusion cache entries: {stats['fusion_cache_entries']}")
    
    print(f"\n13. Demonstrating cache clearing...")
    initial_entries = stats['active_entries']
    caching_manager.clear_cache()
    final_stats = caching_manager.get_cache_stats()
    print(f"   - Before clearing: {initial_entries} entries")
    print(f"   - After clearing: {final_stats['active_entries']} entries")
    print(f"   - Cache size after clearing: {final_stats['current_size_bytes']/(1024**2):.2f}MB")
    
    print("\n" + "="*80)
    print("INTELLIGENT MULTIMODAL CACHING DEMONSTRATION COMPLETE")
    print("="*80)
    print("\nThe intelligent multimodal caching system for Qwen3-VL-2B has been successfully demonstrated.")
    print("Key features implemented:")
    print("- Efficient caching of text, image, and multimodal data")
    print("- Adaptive eviction policies (LRU, LFU, FIFO, predictive)")
    print("- Similarity-based retrieval for related content")
    print("- Compression and TTL for memory efficiency")
    print("- Integration with Qwen3-VL-2B model architecture")


def demonstrate_performance_improvement():
    """
    Demonstrate performance improvement with intelligent caching.
    """
    print("\n" + "="*80)
    print("PERFORMANCE IMPROVEMENT DEMONSTRATION WITH CACHING")
    print("="*80)
    
    # Create caching manager
    caching_manager = Qwen3VL2BIntelligentCachingManager(
        cache_size_gb=0.5,
        eviction_policy=CacheEvictionPolicy.LRU,
        enable_similarity_caching=True,
        similarity_threshold=0.8
    )
    
    # Create sample data
    text = "Describe this image in detail."
    image = Image.new('RGB', (224, 224), color='orange')
    tensor = torch.randn(1, 50, 2048)  # Larger tensor for more realistic timing
    
    print("Testing caching performance improvement...")
    
    # Time uncached operation
    uncached_times = []
    for i in range(5):
        start_time = time.time()
        # Simulate processing that would normally happen
        time.sleep(0.01)  # Simulate processing time
        result = torch.randn(1, 50, 2048)  # Simulate output
        uncached_time = time.time() - start_time
        uncached_times.append(uncached_time)
    
    avg_uncached_time = sum(uncached_times) / len(uncached_times)
    
    # Cache the data
    caching_manager.cache_text_input(text, tensor)
    
    # Time cached operation
    cached_times = []
    for i in range(5):
        start_time = time.time()
        # Retrieve from cache
        result = caching_manager.get_cached_text_input(text)
        cached_time = time.time() - start_time
        cached_times.append(cached_time)
    
    avg_cached_time = sum(cached_times) / len(cached_times)
    
    # Time similarity search
    similarity_times = []
    for i in range(5):
        start_time = time.time()
        # Find similar text (which should be exact match)
        result = caching_manager.find_similar_text(text)
        similarity_time = time.time() - start_time
        similarity_times.append(similarity_time)
    
    avg_similarity_time = sum(similarity_times) / len(similarity_times)
    
    print(f"Average uncached processing time: {avg_uncached_time*1000:.2f}ms")
    print(f"Average cached retrieval time: {avg_cached_time*1000:.2f}ms")
    print(f"Average similarity search time: {avg_similarity_time*1000:.2f}ms")
    if avg_cached_time > 0:
        print(f"Performance improvement with caching: {(avg_uncached_time/avg_cached_time):.2f}x faster")
    else:
        print(f"Performance improvement with caching: >{(avg_uncached_time/0.001):.2f}x faster (cached retrieval time too small to measure)")
    
    print("\n" + "="*80)
    print("PERFORMANCE DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    demonstrate_intelligent_multimodal_caching()
    demonstrate_performance_improvement()