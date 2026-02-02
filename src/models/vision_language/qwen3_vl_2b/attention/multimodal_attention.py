"""
Qwen3-VL-2B Multimodal Attention Implementation

This module implements multimodal attention mechanisms for the Qwen3-VL-2B model using the common implementation.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ....common.multimodal_attention import (
    AdaptiveMultimodalAttention as BaseAdaptiveMultimodalAttention,
)
from ....common.multimodal_attention import (
    EfficientMultimodalCrossAttention as BaseEfficientMultimodalCrossAttention,
)
from ....common.multimodal_attention import (
    ModalitySpecificAttention as BaseModalitySpecificAttention,
)
from ....common.multimodal_attention import (
    MultimodalAlignmentModule,
)
from ....common.multimodal_attention import (
    MultimodalFusionLayer as BaseMultimodalFusionLayer,
)
from ....common.multimodal_attention import (
    create_multimodal_attention,
)
from ..config import Qwen3VL2BConfig


class Qwen3VLMultimodalAttention(BaseEfficientMultimodalCrossAttention):
    """
    Qwen3-VL-2B specific multimodal attention implementation using the common base implementation.

    This implementation provides optimized attention computation for multimodal inputs
    with reduced memory usage and improved performance compared to standard attention mechanisms.
    It inherits from the common EfficientMultimodalCrossAttention implementation to ensure
    consistency across models.
    """

    def __init__(
        self,
        config: Qwen3VL2BConfig,
        layer_idx: Optional[int] = None,
    ):
        # Determine multimodal attention parameters from config
        d_model = config.hidden_size
        nhead = config.num_attention_heads
        modalities = getattr(config, "modalities", ["text", "image"])
        attention_dropout = getattr(config, "attention_dropout_prob", 0.0)
        bias = not getattr(config, "remove_bias_in_attention", False)
        is_causal = getattr(config, "is_causal", True)
        use_flash_attention = getattr(config, "use_flash_attention_2", True)
        use_sparse_attention = getattr(config, "use_sparse_attention", False)
        sparse_topk = getattr(config, "sparse_attention_topk", 32)

        # Initialize using the common EfficientMultimodalCrossAttention implementation
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities,
            dropout=attention_dropout,
            bias=bias,
            is_causal=is_causal,
            use_flash_attention=use_flash_attention,
            use_sparse_attention=use_sparse_attention,
            sparse_topk=sparse_topk,
        )
        self.config = config
        self.layer_idx = layer_idx


class Qwen3VL2BModalitySpecificAttention(BaseModalitySpecificAttention):
    """
    Qwen3-VL-2B specific modality-specific attention implementation using the common base implementation.

    This implementation provides optimized attention computation for specific modalities
    with reduced memory usage and improved performance compared to standard attention mechanisms.
    It inherits from the common ModalitySpecificAttention implementation to ensure consistency across models.
    """

    def __init__(
        self,
        config: Qwen3VL2BConfig,
        modality: str,
        layer_idx: Optional[int] = None,
    ):
        # Determine modality-specific attention parameters from config
        d_model = config.hidden_size
        nhead = config.num_attention_heads
        attention_dropout = getattr(config, "attention_dropout_prob", 0.0)
        bias = not getattr(config, "remove_bias_in_attention", False)
        is_causal = getattr(config, "is_causal", True)

        # Initialize using the common ModalitySpecificAttention implementation
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            modality=modality,
            dropout=attention_dropout,
            bias=bias,
            is_causal=is_causal,
        )
        self.config = config
        self.layer_idx = layer_idx


class Qwen3VL2BMultimodalFusionLayer(BaseMultimodalFusionLayer):
    """
    Qwen3-VL-2B specific multimodal fusion layer implementation using the common base implementation.

    This implementation provides optimized fusion of information from multiple modalities
    with reduced memory usage and improved performance compared to standard fusion mechanisms.
    It inherits from the common MultimodalFusionLayer implementation to ensure consistency across models.
    """

    def __init__(
        self,
        config: Qwen3VL2BConfig,
        layer_idx: Optional[int] = None,
    ):
        # Determine multimodal fusion parameters from config
        d_model = config.hidden_size
        nhead = config.num_attention_heads
        modalities = getattr(config, "modalities", ["text", "image"])
        attention_dropout = getattr(config, "attention_dropout_prob", 0.0)
        activation = getattr(config, "mlp_activation", "silu")  # Qwen3-VL-2B uses SiLU
        use_alignment = getattr(config, "use_cross_modal_alignment", True)
        alignment_method = getattr(
            config, "cross_modal_alignment_method", "qwen3_vl_specific"
        )

        # Initialize using the common MultimodalFusionLayer implementation
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities,
            dropout=attention_dropout,
            activation=activation,
            use_alignment=use_alignment,
            alignment_method=alignment_method,
        )
        self.config = config
        self.layer_idx = layer_idx


class Qwen3VL2BAdaptiveMultimodalAttention(BaseAdaptiveMultimodalAttention):
    """
    Qwen3-VL-2B specific adaptive multimodal attention implementation using the common base implementation.

    This implementation provides attention computation that adapts based on input characteristics
    with reduced memory usage and improved performance compared to standard attention mechanisms.
    It inherits from the common AdaptiveMultimodalAttention implementation to ensure consistency across models.
    """

    def __init__(
        self,
        config: Qwen3VL2BConfig,
        layer_idx: Optional[int] = None,
    ):
        # Determine adaptive multimodal attention parameters from config
        d_model = config.hidden_size
        nhead = config.num_attention_heads
        modalities = getattr(config, "modalities", ["text", "image"])
        attention_dropout = getattr(config, "attention_dropout_prob", 0.0)
        bias = not getattr(config, "remove_bias_in_attention", False)
        is_causal = getattr(config, "is_causal", True)
        use_efficient_attention = getattr(config, "use_flash_attention_2", True)
        adaptive_strategy = getattr(
            config, "adaptive_attention_strategy", "input_dependent"
        )

        # Initialize using the common AdaptiveMultimodalAttention implementation
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities,
            dropout=attention_dropout,
            bias=bias,
            is_causal=is_causal,
            use_efficient_attention=use_efficient_attention,
            adaptive_strategy=adaptive_strategy,
        )
        self.config = config
        self.layer_idx = layer_idx


def create_qwen3_vl_multimodal_attention(
    config: Qwen3VL2BConfig, layer_idx: Optional[int] = None
):
    """
    Factory function to create Qwen3-VL-2B multimodal attention implementation using the common implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)

    Returns:
        Qwen3VL2BMultimodalAttention: The Qwen3-VL-2B multimodal attention implementation
    """
    return Qwen3VL2BMultimodalAttention(config, layer_idx)


def create_qwen3_vl_modality_specific_attention(
    config: Qwen3VL2BConfig, modality: str, layer_idx: Optional[int] = None
):
    """
    Factory function to create Qwen3-VL-2B modality-specific attention implementation using the common implementation.

    Args:
        config: Model configuration
        modality: The modality for which to create attention ('text', 'image', 'audio', etc.)
        layer_idx: Index of the transformer layer (optional)

    Returns:
        Qwen3VL2BModalitySpecificAttention: The Qwen3-VL-2B modality-specific attention implementation
    """
    return Qwen3VL2BModalitySpecificAttention(config, modality, layer_idx)


def create_qwen3_vl_multimodal_fusion_layer(
    config: Qwen3VL2BConfig, layer_idx: Optional[int] = None
):
    """
    Factory function to create Qwen3-VL-2B multimodal fusion layer implementation using the common implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)

    Returns:
        Qwen3VL2BMultimodalFusionLayer: The Qwen3-VL-2B multimodal fusion layer implementation
    """
    return Qwen3VL2BMultimodalFusionLayer(config, layer_idx)


def create_qwen3_vl_adaptive_multimodal_attention(
    config: Qwen3VL2BConfig, layer_idx: Optional[int] = None
):
    """
    Factory function to create Qwen3-VL-2B adaptive multimodal attention implementation using the common implementation.

    Args:
        config: Model configuration
        layer_idx: Index of the transformer layer (optional)

    Returns:
        Qwen3VL2BAdaptiveMultimodalAttention: The Qwen3-VL-2B adaptive multimodal attention implementation
    """
    return Qwen3VL2BAdaptiveMultimodalAttention(config, layer_idx)


__all__ = [
    "Qwen3VL2BMultimodalAttention",
    "Qwen3VL2BModalitySpecificAttention",
    "Qwen3VL2BMultimodalFusionLayer",
    "Qwen3VL2BAdaptiveMultimodalAttention",
    "create_qwen3_vl_multimodal_attention",
    "create_qwen3_vl_modality_specific_attention",
    "create_qwen3_vl_multimodal_fusion_layer",
    "create_qwen3_vl_adaptive_multimodal_attention",
]
