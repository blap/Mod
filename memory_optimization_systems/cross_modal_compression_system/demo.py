"""
Example usage of Cross-Modal Memory Compression System
======================================================

This script demonstrates the usage of the CrossModalCompressor for compressing
multimodal data in the Qwen3-VL model.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cross_modal_compression import (
    CrossModalCompressor,
    CompressionMode,
    cross_modal_fusion_compress
)


def main():
    """Main function to demonstrate cross-modal compression."""
    print("Cross-Modal Memory Compression System Demo")
    print("=" * 50)
    
    # Initialize the compressor
    compressor = CrossModalCompressor(
        compression_threshold=0.6,
        quality_preservation_factor=0.9,
        hardware_target="intel_i5_nvidia_sm61",
        compression_mode=CompressionMode.LOSSY
    )
    
    print(f"Compression threshold: {compressor.compression_threshold}")
    print(f"Quality preservation factor: {compressor.quality_preservation_factor}")
    print(f"Hardware target: {compressor.hardware_target}")
    print()
    
    # Create sample visual and text activations (simulating data from a VL model)
    visual_activations = torch.randn(2, 196, 768)  # Batch=2, Patches=196, Features=768
    text_activations = torch.randn(2, 64, 768)     # Batch=2, Sequence=64, Features=768
    
    print(f"Visual activations shape: {visual_activations.shape}")
    print(f"Text activations shape: {text_activations.shape}")
    print()
    
    # Check if compression is recommended
    visual_compressible = compressor.detect_compression_opportunity(visual_activations)
    text_compressible = compressor.detect_compression_opportunity(text_activations)
    
    print(f"Visual activations compressible: {visual_compressible}")
    print(f"Text activations compressible: {text_compressible}")
    print()
    
    # Perform different types of compression
    print("1. Lossy Compression:")
    compressed_data, metrics = compressor.compress_activations(
        visual_activations, 
        text_activations, 
        mode=CompressionMode.LOSSY
    )
    print(f"   Original size: {metrics.original_size} bytes")
    print(f"   Compressed size: {metrics.compressed_size} bytes")
    print(f"   Compression ratio: {metrics.compression_ratio:.2f}")
    print(f"   Quality loss: {metrics.quality_loss:.4f}")
    print(f"   Memory saved: {metrics.memory_saved} bytes")
    
    # Evaluate trade-offs
    tradeoff_results = compressor.evaluate_tradeoff(metrics)
    print(f"   Memory efficiency score: {tradeoff_results['memory_efficiency_score']:.2f}")
    print(f"   Quality preservation score: {tradeoff_results['quality_preservation_score']:.2f}")
    print(f"   Combined tradeoff score: {tradeoff_results['combined_tradeoff_score']:.2f}")
    print()
    
    print("2. Quantized Compression:")
    compressed_data_q, metrics_q = compressor.compress_activations(
        visual_activations, 
        text_activations, 
        mode=CompressionMode.QUANTIZED
    )
    print(f"   Original size: {metrics_q.original_size} bytes")
    print(f"   Compressed size: {metrics_q.compressed_size} bytes")
    print(f"   Compression ratio: {metrics_q.compression_ratio:.2f}")
    print()
    
    print("3. Sparse Compression:")
    compressed_data_s, metrics_s = compressor.compress_activations(
        visual_activations, 
        text_activations, 
        mode=CompressionMode.SPARSE
    )
    print(f"   Original size: {metrics_s.original_size} bytes")
    print(f"   Compressed size: {metrics_s.compressed_size} bytes")
    print(f"   Compression ratio: {metrics_s.compression_ratio:.2f}")
    print()
    
    # Demonstrate cross-modal fusion compression
    print("4. Cross-Modal Fusion Compression:")
    fused_compressed, fused_metrics = cross_modal_fusion_compress(
        visual_activations, text_activations, compressor
    )
    print(f"   Original size: {fused_metrics.original_size} bytes")
    print(f"   Compressed size: {fused_metrics.compressed_size} bytes")
    print(f"   Compression ratio: {fused_metrics.compression_ratio:.2f}")
    print()
    
    # Show compression statistics
    stats = compressor.get_compression_statistics()
    print("5. Compression Statistics:")
    print(f"   Total compressions performed: {stats['total_compressions']}")
    print(f"   Total memory saved: {stats['total_memory_saved']} bytes")
    print(f"   Average compression ratio: {stats['average_compression_ratio']:.2f}")
    print()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()