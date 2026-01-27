"""
Multimodal Attention Module - Compatibility Wrapper

This module serves as a compatibility wrapper for multimodal attention optimizations,
delegating to the newer optimization modules.
"""

from .multimodal_attention_optimization import (
    apply_multimodal_attention_optimizations_to_model as apply_multimodal_attention_to_model
)

__all__ = [
    "apply_multimodal_attention_to_model"
]
