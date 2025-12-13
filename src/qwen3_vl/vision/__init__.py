"""
Vision module for Qwen3-VL architecture.

This module provides components for processing visual information in the Qwen3-VL model.
"""

from .hierarchical_vision_processor import (
    HierarchicalVisionProcessor,
    MultiResolutionAnalyzer,
    ResolutionAdaptiveBlock,
    ResolutionAdaptiveFusion,
    HierarchicalFeatureExtractor,
    InputComplexityAssessor,
    MultiResolutionAttention
)


__all__ = [
    "HierarchicalVisionProcessor",
    "MultiResolutionAnalyzer",
    "ResolutionAdaptiveBlock",
    "ResolutionAdaptiveFusion",
    "HierarchicalFeatureExtractor",
    "InputComplexityAssessor",
    "MultiResolutionAttention",
    "Qwen3VLVisionTransformer",
    "Qwen3VLVisionModel",
    "Qwen3VLVisionAttention",
    "OptimizedVisionAttention"
]