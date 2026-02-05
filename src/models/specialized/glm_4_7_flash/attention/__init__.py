"""
Attention module for GLM-4.7-Flash model.
Exports the specialized Flash Attention implementation.
"""

from .flash_attention import FlashAttention, FlashAttentionConfig, create_flash_attention_layer

__all__ = [
    "FlashAttention",
    "FlashAttentionConfig", 
    "create_flash_attention_layer"
]