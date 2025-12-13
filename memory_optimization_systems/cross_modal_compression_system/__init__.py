"""
Cross-Modal Memory Compression System for Qwen3-VL
===================================================

This package provides efficient cross-modal memory compression between visual and textual modalities.
"""

from .cross_modal_compression import (
    CrossModalCompressor,
    CompressionMode,
    CompressionMetrics,
    adaptive_compression_selector,
    cross_modal_fusion_compress,
    cleanup_memory
)

__all__ = [
    'CrossModalCompressor',
    'CompressionMode', 
    'CompressionMetrics',
    'adaptive_compression_selector',
    'cross_modal_fusion_compress',
    'cleanup_memory'
]

__version__ = '1.0.0'
__author__ = 'Qwen3-VL Team'