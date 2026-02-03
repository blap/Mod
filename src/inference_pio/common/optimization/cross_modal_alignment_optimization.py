"""
Generic Cross-Modal Alignment Optimization System

This module implements generic cross-modal alignment techniques for multimodal models.
Specific model implementations (like Qwen3-VL-2B) should extend these classes with their own
model-specific optimizations.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CrossModalAlignmentConfig:
    """
    Generic configuration for cross-modal alignment optimization.
    """

    # Temperature for alignment computation (controls sharpness of alignment distribution)
    alignment_temperature: float = 0.5

    # Weight for alignment loss in total loss
    alignment_lambda: float = 0.1

    # Whether to use contrastive alignment loss
    use_contrastive_alignment: bool = True

    # Margin for contrastive loss
    contrastive_margin: float = 0.2

    # Whether to enable dynamic alignment based on input complexity
    enable_dynamic_alignment: bool = True

    # Frequency of alignment updates (every N steps)
    alignment_frequency: int = 10

    # Threshold for alignment quality (above which alignment is considered good enough)
    alignment_threshold: float = 0.8

    # Whether to use attention-based alignment
    use_attention_alignment: bool = True

    # Whether to use learned alignment projections
    use_learned_alignment: bool = True

    # Dimension for alignment projections
    alignment_projection_dim: int = 512

    # Whether to enable similarity-based alignment
    enable_similarity_alignment: bool = True

    # Method for similarity computation ('cosine', 'dot_product', 'euclidean')
    similarity_method: str = "cosine"

    # Hidden size for the model
    hidden_size: int = 2048

    # Number of attention heads
    num_attention_heads: int = 16

    # Number of hidden layers
    num_hidden_layers: int = 24

    # Intermediate size
    intermediate_size: int = 5504

    # Layer norm epsilon
    layer_norm_eps: float = 1e-6


class AlignmentMethod(Enum):
    """Enum for different alignment methods."""

    CONTRASTIVE = "contrastive"
    ATTENTION = "attention"
    LEARNED_PROJECTION = "learned_projection"
    SIMILARITY_BASED = "similarity_based"


class CrossModalAlignmentOptimizer(nn.Module):
    """
    Generic cross-modal alignment optimizer for aligning vision and language representations.
    Specific models should extend this class with their own optimizations.
    """

    def __init__(self, config: CrossModalAlignmentConfig):
        super().__init__()
        self.config = config

        # Generic alignment projection layers
        if config.use_learned_alignment:
            self.vision_projection = nn.Linear(
                config.hidden_size, config.alignment_projection_dim, bias=False
            )
            self.language_projection = nn.Linear(
                config.hidden_size, config.alignment_projection_dim, bias=False
            )

            # Initialize projection weights
            std = config.hidden_size**-0.5
            nn.init.normal_(self.vision_projection.weight, mean=0.0, std=std)
            nn.init.normal_(self.language_projection.weight, mean=0.0, std=std)

        # Attention-based alignment
        if config.use_attention_alignment:
            self.alignment_attention = nn.MultiheadAttention(
                embed_dim=config.alignment_projection_dim,
                num_heads=max(
                    1, config.alignment_projection_dim // 64
                ),  # Heuristic for number of heads
                dropout=0.1,
                batch_first=True,
            )

        # Contrastive alignment components
        if config.use_contrastive_alignment:
            self.contrastive_temperature = config.alignment_temperature
            self.contrastive_margin = config.contrastive_margin

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Align vision and language modalities.

        Args:
            vision_features: Vision features tensor of shape (batch, vision_seq_len, hidden_size)
            language_features: Language features tensor of shape (batch, lang_seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            Tuple of (aligned_features_tuple, alignment_loss)
        """
        batch_size, vision_seq_len, hidden_size = vision_features.shape
        _, lang_seq_len, _ = language_features.shape

        # Pad sequences to same length if needed for alignment operations
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

        # Apply learned projections for alignment
        if self.config.use_learned_alignment:
            aligned_vision = self.vision_projection(vision_padded)
            aligned_language = self.language_projection(language_padded)
        else:
            aligned_vision = vision_padded
            aligned_language = language_padded

        # Apply attention-based alignment
        if self.config.use_attention_alignment:
            # Use language features as query, vision features as key/value for vision-language attention
            aligned_vision_attn, _ = self.alignment_attention(
                aligned_language,
                aligned_vision,
                aligned_vision,
                attn_mask=attention_mask,
                need_weights=False,
            )

            # Use vision features as query, language features as key/value for language-vision attention
            aligned_language_attn, _ = self.alignment_attention(
                aligned_vision,
                aligned_language,
                aligned_language,
                attn_mask=attention_mask,
                need_weights=False,
            )

            # Update aligned features with attention outputs
            aligned_vision = aligned_vision + aligned_vision_attn
            aligned_language = aligned_language + aligned_language_attn

        # Calculate alignment loss
        alignment_loss = self._calculate_alignment_loss(
            vision_features, language_features, aligned_vision, aligned_language
        )

        # Return aligned features with original sequence lengths
        return (
            aligned_vision[:, :vision_seq_len, :],
            aligned_language[:, :lang_seq_len, :],
        ), alignment_loss

    def _calculate_alignment_loss(
        self,
        original_vision: torch.Tensor,
        original_language: torch.Tensor,
        aligned_vision: torch.Tensor,
        aligned_language: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate alignment loss between original and aligned features.

        Args:
            original_vision: Original vision features
            original_language: Original language features
            aligned_vision: Aligned vision features
            aligned_language: Aligned language features

        Returns:
            Alignment loss tensor
        """
        # Calculate similarity between original and aligned features
        if self.config.similarity_method == "cosine":
            vision_sim = F.cosine_similarity(
                original_vision, aligned_vision, dim=-1
            ).mean()
            language_sim = F.cosine_similarity(
                original_language, aligned_language, dim=-1
            ).mean()
        elif self.config.similarity_method == "dot_product":
            vision_sim = torch.sum(original_vision * aligned_vision, dim=-1).mean()
            language_sim = torch.sum(
                original_language * aligned_language, dim=-1
            ).mean()
        else:  # euclidean
            vision_sim = -torch.norm(original_vision - aligned_vision, dim=-1).mean()
            language_sim = -torch.norm(
                original_language - aligned_language, dim=-1
            ).mean()

        # Generic contrastive alignment loss
        contrastive_loss = 0.0
        if self.config.use_contrastive_alignment:
            # Calculate similarity between vision and language features
            # Use mean pooling to get single representations for each modality
            vision_repr = aligned_vision.mean(dim=1)  # Shape: (batch, hidden_size)
            language_repr = aligned_language.mean(dim=1)  # Shape: (batch, hidden_size)

            # Positive pairs (vision-language similarity)
            vision_lang_sim = F.cosine_similarity(
                vision_repr, language_repr, dim=-1
            ).mean()

            # Negative pairs (vision-other_language similarity)
            # Shift language representations to create negative pairs
            lang_shifted = torch.roll(language_repr, shifts=1, dims=0)
            vision_neg_lang_sim = F.cosine_similarity(
                vision_repr, lang_shifted, dim=-1
            ).mean()

            # Contrastive loss: maximize positive pairs, minimize negative pairs
            contrastive_loss = torch.clamp(
                self.config.contrastive_margin - vision_lang_sim + vision_neg_lang_sim,
                min=0,
            )

        # Generic regularization term
        # Encourage alignment to preserve semantic meaning
        semantic_reg_loss = F.mse_loss(aligned_vision, original_vision) + F.mse_loss(
            aligned_language, original_language
        )

        # Combine similarity, contrastive, and regularization losses
        total_loss = (
            (1 - vision_sim)
            + (1 - language_sim)
            + self.config.alignment_lambda * contrastive_loss
            + 0.01 * semantic_reg_loss
        )  # Small regularization term

        return total_loss


class CrossModalAlignmentManager:
    """
    Generic manager for cross-modal alignment operations.
    Specific models should extend this class with their own optimizations.
    """

    def __init__(self, config: CrossModalAlignmentConfig):
        self.config = config
        self.alignment_methods = {}

        # Register default alignment methods
        self.register_alignment_method("contrastive", config)
        self.register_alignment_method("attention", config)
        self.register_alignment_method("learned_projection", config)
        self.register_alignment_method("similarity_based", config)

    def register_alignment_method(
        self, method_name: str, config: CrossModalAlignmentConfig
    ):
        """
        Register an alignment method with its configuration.

        Args:
            method_name: Name of the alignment method
            config: Configuration for the alignment method
        """
        self.alignment_methods[method_name] = CrossModalAlignmentOptimizer(config)

    def get_alignment_optimizer(
        self, method_name: str
    ) -> Optional[CrossModalAlignmentOptimizer]:
        """
        Get a registered alignment optimizer.

        Args:
            method_name: Name of the alignment method

        Returns:
            Alignment optimizer if registered, None otherwise
        """
        return self.alignment_methods.get(method_name)

    def align_modalities(
        self,
        method_name: str,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Align vision and language modalities using the specified method.

        Args:
            method_name: Name of the alignment method to use
            vision_features: Vision features tensor
            language_features: Language features tensor
            attention_mask: Optional attention mask

        Returns:
            Tuple of (aligned_vision, aligned_language, alignment_loss)
        """
        optimizer = self.get_alignment_optimizer(method_name)
        if optimizer is None:
            raise ValueError(f"Alignment method '{method_name}' not registered")

        aligned_features, alignment_loss = optimizer(
            vision_features, language_features, attention_mask
        )

        return aligned_features[0], aligned_features[1], alignment_loss

    def evaluate_alignment_quality(
        self,
        original_vision: torch.Tensor,
        original_language: torch.Tensor,
        aligned_vision: torch.Tensor,
        aligned_language: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate the quality of cross-modal alignment.

        Args:
            original_vision: Original vision features
            original_language: Original language features
            aligned_vision: Aligned vision features
            aligned_language: Aligned language features

        Returns:
            Dictionary with alignment quality metrics
        """
        # Calculate similarity metrics
        vision_similarity = (
            F.cosine_similarity(original_vision, aligned_vision, dim=-1).mean().item()
        )
        language_similarity = (
            F.cosine_similarity(original_language, aligned_language, dim=-1)
            .mean()
            .item()
        )

        # Calculate cross-modal similarity
        vision_repr = aligned_vision.mean(dim=1)
        language_repr = aligned_language.mean(dim=1)
        cross_modal_similarity = (
            F.cosine_similarity(vision_repr, language_repr, dim=-1).mean().item()
        )

        # Calculate alignment improvement
        original_cross_modal_sim = (
            F.cosine_similarity(
                original_vision.mean(dim=1), original_language.mean(dim=1), dim=-1
            )
            .mean()
            .item()
        )

        alignment_improvement = cross_modal_similarity - original_cross_modal_sim

        return {
            "vision_similarity": vision_similarity,
            "language_similarity": language_similarity,
            "cross_modal_similarity": cross_modal_similarity,
            "alignment_improvement": alignment_improvement,
            "quality_score": (
                vision_similarity + language_similarity + cross_modal_similarity
            )
            / 3.0,
        }


def create_generic_cross_modal_alignment(
    config: CrossModalAlignmentConfig,
) -> CrossModalAlignmentManager:
    """
    Create a generic cross-modal alignment manager.

    Args:
        config: CrossModalAlignmentConfig configuration

    Returns:
        CrossModalAlignmentManager configured generically
    """
    return CrossModalAlignmentManager(config)


def apply_cross_modal_alignment_to_model(
    model: nn.Module, config: CrossModalAlignmentConfig
) -> nn.Module:
    """
    Apply generic cross-modal alignment optimizations to the model.

    Args:
        model: The model to optimize
        config: Configuration for the model

    Returns:
        Model with cross-modal alignment capabilities
    """
    logger.info("Applying generic cross-modal alignment optimizations to model...")

    # Create alignment manager
    alignment_manager = create_generic_cross_modal_alignment(config)

    # Add alignment manager to model
    model.cross_modal_alignment_manager = alignment_manager

    # Add alignment method to model
    def perform_cross_modal_alignment(
        self,
        vision_features,
        language_features,
        method="learned_projection",
        attention_mask=None,
    ):
        """
        Perform cross-modal alignment between vision and language features.

        Args:
            vision_features: Vision features tensor
            language_features: Language features tensor
            method: Alignment method to use
            attention_mask: Optional attention mask

        Returns:
            Tuple of (aligned_vision, aligned_language, alignment_loss)
        """
        if not hasattr(self, "cross_modal_alignment_manager"):
            raise RuntimeError("Cross-modal alignment manager not available in model")

        return self.cross_modal_alignment_manager.align_modalities(
            method, vision_features, language_features, attention_mask
        )

    # Bind the method to the model
    model.perform_cross_modal_alignment = perform_cross_modal_alignment.__get__(
        model, model.__class__
    )

    logger.info("Generic cross-modal alignment optimizations applied successfully")
    return model


def get_cross_modal_alignment_report(
    model: nn.Module, config: CrossModalAlignmentConfig
) -> Dict:
    """
    Get a report of cross-modal alignment applied to the model.

    Args:
        model: The model
        config: Configuration for the model

    Returns:
        Report dictionary
    """
    # Check if cross-modal alignment is enabled in the config
    alignment_enabled = getattr(
        config, "enable_cross_modal_alignment", False
    ) or hasattr(model, "cross_modal_alignment_manager")

    report = {
        "model_type": "Generic Multimodal Model",
        "optimization_type": "Cross-Modal Alignment",
        "alignment_methods_registered": [],
        "alignment_enabled": alignment_enabled,
        "alignment_config": {
            "alignment_temperature": getattr(config, "alignment_temperature", 0.5),
            "alignment_lambda": getattr(config, "alignment_lambda", 0.1),
            "use_contrastive_alignment": getattr(
                config, "use_contrastive_alignment", True
            ),
            "contrastive_margin": getattr(config, "contrastive_margin", 0.2),
            "enable_dynamic_alignment": getattr(
                config, "enable_dynamic_alignment", True
            ),
            "alignment_frequency": getattr(config, "alignment_frequency", 10),
            "alignment_threshold": getattr(config, "alignment_threshold", 0.8),
            "use_attention_alignment": getattr(config, "use_attention_alignment", True),
            "use_learned_alignment": getattr(config, "use_learned_alignment", True),
            "alignment_projection_dim": getattr(
                config, "alignment_projection_dim", getattr(config, "hidden_size", 2048)
            ),
            "enable_similarity_alignment": getattr(
                config, "enable_similarity_alignment", True
            ),
            "similarity_method": getattr(config, "similarity_method", "cosine"),
        },
    }

    if hasattr(model, "cross_modal_alignment_manager"):
        if hasattr(model.cross_modal_alignment_manager, "alignment_methods"):
            report["alignment_methods_registered"] = list(
                model.cross_modal_alignment_manager.alignment_methods.keys()
            )
        else:
            report["alignment_methods_registered"] = []

    return report


__all__ = [
    "CrossModalAlignmentConfig",
    "CrossModalAlignmentOptimizer",
    "CrossModalAlignmentManager",
    "create_generic_cross_modal_alignment",
    "apply_cross_modal_alignment_to_model",
    "get_cross_modal_alignment_report",
]
