"""
Cross-Modal Alignment Optimization for Qwen3-VL-2B Model - Self-Contained Version

This module implements optimized cross-modal alignment techniques specifically for the Qwen3-VL-2B model.
The system efficiently aligns vision and language representations to improve multimodal understanding
and generation capabilities.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class CrossModalAlignmentConfig:
    """
    Configuration for cross-modal alignment optimization.
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


class Qwen3VL2BCrossModalAlignmentOptimizer(nn.Module):
    """
    Qwen3-VL-2B specific cross-modal alignment optimizer.
    This optimizer is tailored to the specific architecture and parameters of Qwen3-VL-2B.
    """

    def __init__(self, config: CrossModalAlignmentConfig, layer_idx: int = 0):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Qwen3-VL-2B specific alignment components
        self.vision_language_gate = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )
        self.language_vision_gate = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )

        # Qwen3-VL-2B specific normalization
        self.vision_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.language_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Qwen3-VL-2B specific alignment attention (SwiGLU-like mechanism)
        self.alignment_up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.alignment_gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.alignment_down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

        # Initialize Qwen3-VL-2B specific weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize Qwen3-VL-2B specific weights."""
        std = self.hidden_size**-0.5
        nn.init.normal_(self.vision_language_gate.weight, mean=0.0, std=std)
        nn.init.normal_(self.language_vision_gate.weight, mean=0.0, std=std)

        std = self.config.intermediate_size**-0.5
        nn.init.normal_(self.alignment_up_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.alignment_gate_proj.weight, mean=0.0, std=std)

        std = (2 * self.layer_idx + 2) ** -0.5  # Use layer index for scaling
        nn.init.normal_(self.alignment_down_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Qwen3-VL-2B specific alignment of vision and language modalities.

        Args:
            vision_features: Vision features tensor of shape (batch, vision_seq_len, hidden_size)
            language_features: Language features tensor of shape (batch, lang_seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            Tuple of (aligned_features_tuple, alignment_loss)
        """
        batch_size, vision_seq_len, hidden_size = vision_features.shape
        _, lang_seq_len, _ = language_features.shape

        # Normalize inputs using Qwen3-VL-2B specific norms
        vision_norm = self.vision_norm(vision_features)
        language_norm = self.language_norm(language_features)

        # Pad sequences to same length if needed for alignment operations
        max_len = max(vision_seq_len, lang_seq_len)

        if vision_seq_len < max_len:
            vision_padded = F.pad(
                vision_norm, (0, 0, 0, max_len - vision_seq_len), value=0
            )
        else:
            vision_padded = vision_norm

        if lang_seq_len < max_len:
            language_padded = F.pad(
                language_norm, (0, 0, 0, max_len - lang_seq_len), value=0
            )
        else:
            language_padded = language_norm

        # Apply Qwen3-VL-2B specific SwiGLU-like alignment mechanism
        # Gate and up projections
        vision_gate = F.silu(self.alignment_gate_proj(vision_padded))
        vision_up = self.alignment_up_proj(vision_padded)
        vision_aligned = self.alignment_down_proj(
            vision_gate * vision_up
        )  # Element-wise multiplication for SwiGLU

        language_gate = F.silu(self.alignment_gate_proj(language_padded))
        language_up = self.alignment_up_proj(language_padded)
        language_aligned = self.alignment_down_proj(
            language_gate * language_up
        )  # Element-wise multiplication for SwiGLU

        # Apply Qwen3-VL-2B specific gating mechanisms
        # Process vision and language features separately for gating
        # First, pad to same length for cross-modal processing
        if vision_padded.size(1) < max_len:
            vision_padded_for_gate = F.pad(
                vision_padded, (0, 0, 0, max_len - vision_padded.size(1)), value=0
            )
        else:
            vision_padded_for_gate = vision_padded

        if language_padded.size(1) < max_len:
            language_padded_for_gate = F.pad(
                language_padded, (0, 0, 0, max_len - language_padded.size(1)), value=0
            )
        else:
            language_padded_for_gate = language_padded

        # Apply cross-modal gating using separate mechanisms
        # Create vision-to-language gate: process vision features to influence language
        vision_to_lang_avg = vision_padded_for_gate.mean(
            dim=1, keepdim=True
        )  # Average across sequence dimension
        vision_to_lang_combined = torch.cat(
            [
                vision_to_lang_avg.expand_as(language_padded_for_gate),
                language_padded_for_gate,
            ],
            dim=-1,
        )
        vision_language_gate = torch.sigmoid(
            self.vision_language_gate(vision_to_lang_combined)
        )

        # Create language-to-vision gate: process language features to influence vision
        language_to_vision_avg = language_padded_for_gate.mean(
            dim=1, keepdim=True
        )  # Average across sequence dimension
        language_to_vision_combined = torch.cat(
            [
                language_to_vision_avg.expand_as(vision_padded_for_gate),
                vision_padded_for_gate,
            ],
            dim=-1,
        )
        language_vision_gate = torch.sigmoid(
            self.language_vision_gate(language_to_vision_combined)
        )

        # Apply gates to the respective outputs (only to the aligned parts)
        # Ensure the gate tensors match the sequence lengths of the original features
        vision_gate_part = language_vision_gate[
            :, :vision_seq_len, : self.config.hidden_size
        ]
        language_gate_part = vision_language_gate[
            :, :lang_seq_len, : self.config.hidden_size
        ]

        # Make sure the gate parts match the original sequence lengths
        if vision_gate_part.size(1) != vision_seq_len:
            vision_gate_part = F.pad(
                vision_gate_part,
                (0, 0, 0, vision_seq_len - vision_gate_part.size(1)),
                value=0,
            )
        if language_gate_part.size(1) != lang_seq_len:
            language_gate_part = F.pad(
                language_gate_part,
                (0, 0, 0, lang_seq_len - language_gate_part.size(1)),
                value=0,
            )

        # Apply gating to aligned features
        aligned_vision = (
            vision_features + vision_aligned[:, :vision_seq_len, :] * vision_gate_part
        )
        aligned_language = (
            language_features
            + language_aligned[:, :lang_seq_len, :] * language_gate_part
        )

        # Calculate Qwen3-VL-2B specific alignment loss
        alignment_loss = self._calculate_qwen_alignment_loss(
            vision_features, language_features, aligned_vision, aligned_language
        )

        # Return aligned features with original sequence lengths
        return (
            aligned_vision[:, :vision_seq_len, :],
            aligned_language[:, :lang_seq_len, :],
        ), alignment_loss

    def _calculate_qwen_alignment_loss(
        self,
        original_vision: torch.Tensor,
        original_language: torch.Tensor,
        aligned_vision: torch.Tensor,
        aligned_language: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate Qwen3-VL-2B specific alignment loss.

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

        # Qwen3-VL-2B specific contrastive alignment loss
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

        # Qwen3-VL-2B specific regularization term
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
    Manager for cross-modal alignment operations.
    Handles selection and application of appropriate alignment methods.
    """

    def __init__(self, config: CrossModalAlignmentConfig):
        self.config = config
        self.alignment_methods = {}

        # Register default alignment methods
        self.register_alignment_method("contrastive", config)
        self.register_alignment_method("attention", config)
        self.register_alignment_method("learned_projection", config)
        self.register_alignment_method("similarity_based", config)
        self.register_alignment_method("qwen3_vl_specific", config)

    def register_alignment_method(
        self, method_name: str, config: CrossModalAlignmentConfig
    ):
        """
        Register an alignment method with its configuration.

        Args:
            method_name: Name of the alignment method
            config: Configuration for the alignment method
        """
        if method_name == "qwen3_vl_specific":
            self.alignment_methods[method_name] = Qwen3VL2BCrossModalAlignmentOptimizer(
                config=config,
                layer_idx=0,  # Default to layer 0, can be changed as needed
            )
        else:
            # Create a temporary config that has the required attributes
            temp_config = CrossModalAlignmentConfig(
                alignment_temperature=config.alignment_temperature,
                alignment_lambda=config.alignment_lambda,
                use_contrastive_alignment=config.use_contrastive_alignment,
                contrastive_margin=config.contrastive_margin,
                enable_dynamic_alignment=config.enable_dynamic_alignment,
                alignment_frequency=config.alignment_frequency,
                alignment_threshold=config.alignment_threshold,
                use_attention_alignment=config.use_attention_alignment,
                use_learned_alignment=config.use_learned_alignment,
                alignment_projection_dim=config.alignment_projection_dim,
                enable_similarity_alignment=config.enable_similarity_alignment,
                similarity_method=config.similarity_method,
            )
            # Set the hidden_size attribute from the original config
            temp_config.hidden_size = getattr(config, "hidden_size", 2048)
            temp_config.num_attention_heads = getattr(config, "num_attention_heads", 16)
            temp_config.num_hidden_layers = getattr(config, "num_hidden_layers", 24)
            temp_config.intermediate_size = getattr(config, "intermediate_size", 5504)
            temp_config.layer_norm_eps = getattr(config, "layer_norm_eps", 1e-06)

            self.alignment_methods[method_name] = Qwen3VL2BCrossModalAlignmentOptimizer(
                temp_config
            )

    def get_alignment_optimizer(
        self, method_name: str
    ) -> Optional[Qwen3VL2BCrossModalAlignmentOptimizer]:
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


def create_qwen3_vl_cross_modal_alignment(
    config: CrossModalAlignmentConfig,
) -> CrossModalAlignmentManager:
    """
    Create a cross-modal alignment manager specifically for Qwen3-VL-2B.

    Args:
        config: CrossModalAlignmentConfig configuration

    Returns:
        CrossModalAlignmentManager configured for Qwen3-VL-2B
    """
    return CrossModalAlignmentManager(config)


def apply_cross_modal_alignment_to_model(
    model: nn.Module, config: CrossModalAlignmentConfig
) -> nn.Module:
    """
    Apply cross-modal alignment optimizations to the model.

    Args:
        model: The model to optimize
        config: Configuration for the model

    Returns:
        Model with cross-modal alignment capabilities
    """
    logger.info("Applying cross-modal alignment optimizations to model...")

    # Create alignment manager
    alignment_manager = create_qwen3_vl_cross_modal_alignment(config)

    # Add alignment manager to model
    model.cross_modal_alignment_manager = alignment_manager

    # Add alignment optimizer (specific to Qwen3-VL-2B)
    model.cross_modal_alignment_optimizer = alignment_manager.get_alignment_optimizer(
        "qwen3_vl_specific"
    )

    # Add alignment method to model
    def perform_cross_modal_alignment(
        self,
        vision_features,
        language_features,
        method="qwen3_vl_specific",
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

    logger.info("Cross-modal alignment optimizations applied successfully")
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
        "model_type": "Qwen3-VL-2B",
        "optimization_type": "Cross-Modal Alignment",
        "alignment_methods_registered": [],
        "alignment_enabled": alignment_enabled,
        "alignment_config": {
            "alignment_temperature": getattr(
                config,
                "cross_modal_alignment_temperature",
                getattr(config, "rope_theta", 0.5),
            ),
            "alignment_lambda": getattr(config, "cross_modal_alignment_lambda", 0.1),
            "use_contrastive_alignment": getattr(
                config, "use_cross_modal_contrastive_alignment", True
            ),
            "contrastive_margin": getattr(
                config, "cross_modal_contrastive_margin", 0.2
            ),
            "enable_dynamic_alignment": getattr(
                config, "enable_dynamic_cross_modal_alignment", True
            ),
            "alignment_frequency": getattr(
                config, "cross_modal_alignment_frequency", 10
            ),
            "alignment_threshold": getattr(
                config, "cross_modal_alignment_threshold", 0.8
            ),
            "use_attention_alignment": getattr(
                config, "use_cross_modal_attention_alignment", True
            ),
            "use_learned_alignment": getattr(
                config, "use_cross_modal_learned_alignment", True
            ),
            "alignment_projection_dim": getattr(
                config,
                "cross_modal_alignment_projection_dim",
                getattr(config, "hidden_size", 2048),
            ),
            "enable_similarity_alignment": getattr(
                config, "enable_cross_modal_similarity_alignment", True
            ),
            "similarity_method": getattr(
                config, "cross_modal_similarity_method", "cosine"
            ),
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
    "Qwen3VL2BCrossModalAlignmentOptimizer",
    "CrossModalAlignmentManager",
    "create_qwen3_vl_cross_modal_alignment",
    "apply_cross_modal_alignment_to_model",
    "get_cross_modal_alignment_report",
]
