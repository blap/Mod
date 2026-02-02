"""
Generic Cross-Modal Fusion Kernels for Vision-Language Models

This module implements generic cross-modal fusion kernels for multimodal operations in vision-language models.
Specific model implementations (like Qwen3-VL-2B) should extend these classes with their own
model-specific optimizations.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CrossModalFusionConfig:
    """Generic configuration for cross-modal fusion kernels."""

    hidden_size: int = 2048
    num_attention_heads: int = 16
    intermediate_size: int = 5504
    dropout: float = 0.1
    use_flash_attention: bool = True
    fusion_method: str = "concat"  # Options: "concat", "add", "multiply", "attention"
    temperature: float = 1.0
    max_vision_tokens: int = 1024
    max_language_tokens: int = 2048
    layer_norm_eps: float = 1e-6


class GenericCrossModalFusionKernel(nn.Module):
    """
    Generic implementation of cross-modal fusion kernel.
    Specific models should extend this class with their own optimizations.
    """

    def __init__(self, config: CrossModalFusionConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.fusion_method = config.fusion_method
        self.temperature = config.temperature

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f"Hidden size must be divisible by number of heads")

        # Generic fusion components
        self.vision_language_gate = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=False
        )
        self.language_vision_gate = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights according to generic specifications."""
        std = self.hidden_size**-0.5
        nn.init.normal_(self.vision_language_gate.weight, mean=0.0, std=std)
        nn.init.normal_(self.language_vision_gate.weight, mean=0.0, std=std)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for generic cross-modal fusion kernel.

        Args:
            vision_features: Vision features tensor of shape (batch, vision_seq_len, hidden_size)
            language_features: Language features tensor of shape (batch, lang_seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            Tuple of (fused_output, vision_output, language_output)
        """
        batch_size, vision_seq_len, hidden_size = vision_features.shape
        _, lang_seq_len, _ = language_features.shape

        # Pad sequences to same length if needed for fusion operations
        max_len = max(vision_seq_len, lang_seq_len)

        if vision_seq_len < max_len:
            vision_padded = F.pad(
                vision_features, (0, 0, 0, max_len - vision_seq_len), value=0
            )
        else:
            vision_padded = vision_features

        if lang_seq_len < max_len:
            language_padded = F.pad(
                language_features, (0, 0, 0, max_len - lang_seq_len), value=0
            )
        else:
            language_padded = language_features

        # Apply fusion based on method
        if self.fusion_method == "add":
            fused_output = vision_padded + language_padded
        elif self.fusion_method == "concat":
            concatenated = torch.cat([vision_padded, language_padded], dim=-1)
            # Project concatenated features back to original dimension
            fused_output = nn.Linear(concatenated.size(-1), self.hidden_size).to(
                concatenated.device
            )(concatenated)
        elif self.fusion_method == "multiply":
            # Element-wise multiplication
            min_len = min(vision_padded.size(1), language_padded.size(1))
            fused_output = (
                vision_padded[:, :min_len, :] * language_padded[:, :min_len, :]
            )
        elif self.fusion_method == "attention":
            # Attention-based fusion
            # Compute attention between vision and language features
            attention_scores = (
                torch.matmul(
                    F.normalize(vision_padded, dim=-1),
                    F.normalize(language_padded, dim=-1).transpose(-2, -1),
                )
                / self.temperature
            )

            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            attention_weights = F.softmax(
                attention_scores, dim=-1, dtype=torch.float32
            ).to(vision_features.dtype)
            fused_output = torch.matmul(attention_weights, language_padded)
        else:
            # Default to addition
            fused_output = vision_padded + language_padded

        # Apply gating mechanisms
        # Compute average representations for cross-gating
        vision_avg = vision_padded.mean(dim=1, keepdim=True)  # (batch, 1, hidden_size)
        language_avg = language_padded.mean(
            dim=1, keepdim=True
        )  # (batch, 1, hidden_size)

        # Create vision-to-language gate: process vision features to influence language
        vision_to_lang_combined = torch.cat(
            [vision_avg.expand_as(language_padded), language_padded], dim=-1
        )
        vision_language_gate = torch.sigmoid(
            self.vision_language_gate(vision_to_lang_combined)
        )

        # Create language-to-vision gate: process language features to influence vision
        language_to_vision_combined = torch.cat(
            [language_avg.expand_as(vision_padded), vision_padded], dim=-1
        )
        language_vision_gate = torch.sigmoid(
            self.language_vision_gate(language_to_vision_combined)
        )

        # Apply gates to the respective outputs (only to the original sequence lengths)
        vision_gate_part = language_vision_gate[:, :vision_seq_len, : self.hidden_size]
        language_gate_part = vision_language_gate[:, :lang_seq_len, : self.hidden_size]

        # Apply gating to create final outputs
        vision_output = vision_features + (
            fused_output[:, :vision_seq_len, :] * vision_gate_part
        )
        language_output = language_features + (
            fused_output[:, :lang_seq_len, :] * language_gate_part
        )

        # Truncate fused output to match original sequence lengths
        fused_output = fused_output[:, : max(vision_seq_len, lang_seq_len), :]

        return fused_output, vision_output, language_output


class GenericCrossModalFusionManager:
    """
    Generic manager for cross-modal fusion operations.
    Specific models should extend this class with their own optimizations.
    """

    def __init__(self, config: CrossModalFusionConfig):
        self.config = config
        self.fusion_methods = {}

        # Register default fusion methods
        self.register_fusion_method("add", config)
        self.register_fusion_method("concat", config)
        self.register_fusion_method("multiply", config)
        self.register_fusion_method("attention", config)

    def register_fusion_method(self, method_name: str, config: CrossModalFusionConfig):
        """
        Register a fusion method with its configuration.

        Args:
            method_name: Name of the fusion method
            config: Configuration for the fusion method
        """
        # Create a new config with the specific fusion method
        method_config = CrossModalFusionConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
            use_flash_attention=config.use_flash_attention,
            fusion_method=method_name,
            temperature=config.temperature,
            max_vision_tokens=config.max_vision_tokens,
            max_language_tokens=config.max_language_tokens,
            layer_norm_eps=config.layer_norm_eps,
        )

        self.fusion_methods[method_name] = GenericCrossModalFusionKernel(method_config)

    def get_fusion_kernel(
        self, method_name: str
    ) -> Optional[GenericCrossModalFusionKernel]:
        """
        Get a registered fusion kernel.

        Args:
            method_name: Name of the fusion method

        Returns:
            Fusion kernel if registered, None otherwise
        """
        return self.fusion_methods.get(method_name)

    def fuse_modalities(
        self,
        method_name: str,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fuse vision and language modalities using the specified method.

        Args:
            method_name: Name of the fusion method to use
            vision_features: Vision features tensor
            language_features: Language features tensor
            attention_mask: Optional attention mask

        Returns:
            Tuple of (fused_output, vision_output, language_output)
        """
        kernel = self.get_fusion_kernel(method_name)
        if kernel is None:
            raise ValueError(f"Fusion method '{method_name}' not registered")

        return kernel(vision_features, language_features, attention_mask)


def create_generic_cross_modal_fusion(
    config: CrossModalFusionConfig,
) -> GenericCrossModalFusionManager:
    """
    Create a generic cross-modal fusion manager.

    Args:
        config: CrossModalFusionConfig configuration

    Returns:
        GenericCrossModalFusionManager instance
    """
    return GenericCrossModalFusionManager(config)


def apply_generic_cross_modal_fusion_to_model(
    model: nn.Module, config: CrossModalFusionConfig
) -> nn.Module:
    """
    Apply generic cross-modal fusion optimizations to the model.

    Args:
        model: The model to optimize
        config: Configuration for the model

    Returns:
        Model with cross-modal fusion capabilities
    """
    logger.info("Applying generic cross-modal fusion optimizations to model...")

    # Create fusion manager
    fusion_manager = create_generic_cross_modal_fusion(config)

    # Add fusion manager to model
    model.cross_modal_fusion_manager = fusion_manager

    # Add fusion method to model
    def perform_cross_modal_fusion(
        self, vision_features, language_features, method="add", attention_mask=None
    ):
        """
        Perform cross-modal fusion between vision and language features.

        Args:
            vision_features: Vision features tensor
            language_features: Language features tensor
            method: Fusion method to use
            attention_mask: Optional attention mask

        Returns:
            Tuple of (fused_output, vision_output, language_output)
        """
        if not hasattr(self, "cross_modal_fusion_manager"):
            raise RuntimeError("Cross-modal fusion manager not available in model")

        return self.cross_modal_fusion_manager.fuse_modalities(
            method, vision_features, language_features, attention_mask
        )

    # Bind the method to the model
    model.perform_cross_modal_fusion = perform_cross_modal_fusion.__get__(
        model, model.__class__
    )

    logger.info("Generic cross-modal fusion optimizations applied successfully")
    return model


__all__ = [
    "CrossModalFusionConfig",
    "GenericCrossModalFusionKernel",
    "GenericCrossModalFusionManager",
    "create_generic_cross_modal_fusion",
    "apply_generic_cross_modal_fusion_to_model",
]
