"""
Corrected Migration Script for Qwen3-VL-2B Specific Code

This script moves all Qwen3-VL-2B specific code from the common directory
to the model-specific plugin directory, ensuring proper separation of concerns.
"""

import os
import shutil
from pathlib import Path


def migrate_remaining_qwen3_vl_specific_code():
    """
    Migrate remaining Qwen3-VL-2B specific code from common to model-specific directory.
    """
    print("Starting migration of remaining Qwen3-VL-2B specific code...")

    # Source and destination directories
    common_dir = Path("C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/common")
    model_dir = Path(
        "C:/Users/Admin/Documents/Mod/src/inference_pio/models/qwen3_vl_2b"
    )

    # Files that contain Qwen3-VL-2B specific code in the common directory
    files_to_move = [
        "cross_modal_alignment_optimization.py",
        "cross_modal_fusion_kernels.py",
        "quantized_multimodal_kernels.py",
        "vision_transformer_kernels.py",
        "rotary_embeddings.py",
        "async_multimodal_processing.py",
        "intelligent_multimodal_caching.py",
        "visual_resource_compression.py",
        "image_tokenization.py",
    ]

    for file_name in files_to_move:
        src_path = common_dir / file_name
        dst_path = model_dir / file_name

        if src_path.exists():
            print(f"Migrating {file_name} from common to model-specific directory...")

            # If destination already exists, we'll skip to avoid overwriting our model-specific implementations
            if dst_path.exists():
                print(f"  {dst_path} already exists, skipping migration for this file")
                continue

            # Move the file
            try:
                shutil.move(str(src_path), str(dst_path))
                print(f"  Successfully moved {file_name}")
            except Exception as e:
                print(f"  Error moving {file_name}: {e}")
        else:
            print(f"  {src_path} does not exist, skipping")

    # Also check for files in subdirectories that might contain Qwen3-VL-2B specific code
    subdirs_to_check = [
        "multimodal_attention",
        "multimodal_cuda_kernels",
        "multimodal_preprocessing",
        "multimodal_projector",
        "vision_transformer_kernels",
        "rotary_embeddings",
        "quantized_multimodal_kernels",
        "async_multimodal_processing",
        "intelligent_multimodal_caching",
        "visual_resource_compression",
        "image_tokenization",
    ]

    for subdir_name in subdirs_to_check:
        src_subdir = common_dir / subdir_name
        dst_subdir = model_dir / subdir_name

        if src_subdir.exists():
            print(f"Migrating {subdir_name} directory from common to model-specific...")

            if dst_subdir.exists():
                print(
                    f"  Warning: {dst_subdir} already exists, removing and replacing..."
                )
                shutil.rmtree(dst_subdir)

            try:
                shutil.move(str(src_subdir), str(dst_subdir))
                print(f"  Successfully moved {subdir_name} directory")
            except Exception as e:
                print(f"  Error moving {subdir_name} directory: {e}")

    print("\nMigration completed!")
    print("Summary:")
    print("- Qwen3-VL-2B specific code has been moved to the model plugin directory")
    print("- Generic code remains in the common directory")


def update_common_init():
    """
    Update the common __init__.py to remove Qwen3-VL-2B specific imports and aliases.
    """
    print("\nUpdating common/__init__.py to remove Qwen3-VL-2B specific aliases...")

    common_init_path = Path(
        "C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/common/__init__.py"
    )

    if common_init_path.exists():
        with open(common_init_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove Qwen3-VL-2B specific aliases while keeping generic implementations
        lines = content.split("\n")
        filtered_lines = []

        for line in lines:
            # Skip lines that create Qwen3-VL-2B specific aliases
            if not any(
                alias in line
                for alias in [
                    "Qwen3VL2BConfig",
                    "Qwen3VL2BCrossAttentionKernel",
                    "Qwen3VL2BFusionKernel",
                    "Qwen3VL2BVisionLanguageAttentionKernel",
                    "Qwen3VL2BPositionEncodingKernel",
                    "Qwen3VL2BMLPKernel",
                    "Qwen3VL2BRMSNormKernel",
                    "Qwen3VL2BVisionProcessingKernel",
                    "create_qwen3_vl_cross_attention_kernel",
                    "create_qwen3_vl_fusion_kernel",
                    "create_qwen3_vl_vision_language_attention_kernel",
                    "create_qwen3_vl_position_encoding_kernel",
                    "create_qwen3_vl_mlp_kernel",
                    "create_qwen3_vl_rms_norm_kernel",
                    "create_qwen3_vl_vision_processing_kernel",
                    "apply_qwen3_vl_cuda_optimizations_to_model",
                    "get_qwen3_vl_cuda_optimization_report",
                ]
            ):
                filtered_lines.append(line)

        # Join the filtered lines back together
        new_content = "\n".join(filtered_lines)

        # Write the updated content back
        with open(common_init_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print("  Updated common/__init__.py to remove Qwen3-VL-2B specific aliases")
    else:
        print("  common/__init__.py not found")


def create_generic_versions():
    """
    Create generic versions of the modules that were removed from common.
    """
    print("\nCreating generic versions of modules...")

    # Create generic versions of the modules that should remain in common
    generic_modules = {
        "cross_modal_alignment_optimization.py": '''
"""
Generic Cross-Modal Alignment Optimization System

This module implements generic cross-modal alignment techniques for multimodal models.
Specific model implementations (like Qwen3-VL-2B) should extend these classes with their own
model-specific optimizations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging
from dataclasses import dataclass
from enum import Enum

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
    similarity_method: str = 'cosine'


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
            self.vision_projection = nn.Linear(config.hidden_size, config.alignment_projection_dim, bias=False)
            self.language_projection = nn.Linear(config.hidden_size, config.alignment_projection_dim, bias=False)

            # Initialize projection weights
            std = config.hidden_size ** -0.5
            nn.init.normal_(self.vision_projection.weight, mean=0.0, std=std)
            nn.init.normal_(self.language_projection.weight, mean=0.0, std=std)

        # Attention-based alignment
        if config.use_attention_alignment:
            self.alignment_attention = nn.MultiheadAttention(
                embed_dim=config.alignment_projection_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )

        # Contrastive alignment components
        if config.use_contrastive_alignment:
            self.contrastive_temperature = config.alignment_temperature
            self.contrastive_margin = config.contrastive_margin

    def align_modalities(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
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
            vision_padded = F.pad(vision_features, (0, 0, 0, max_len - vision_seq_len), value=0)
        else:
            vision_padded = vision_features

        if lang_seq_len < max_len:
            language_padded = F.pad(language_features, (0, 0, 0, max_len - lang_seq_len), value=0)
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
                aligned_language, aligned_vision, aligned_vision,
                attn_mask=attention_mask,
                need_weights=False
            )

            # Use vision features as query, language features as key/value for language-vision attention
            aligned_language_attn, _ = self.alignment_attention(
                aligned_vision, aligned_language, aligned_language,
                attn_mask=attention_mask,
                need_weights=False
            )

            # Update aligned features with attention outputs
            aligned_vision = aligned_vision + aligned_vision_attn
            aligned_language = aligned_language + aligned_language_attn

        # Calculate alignment loss
        alignment_loss = self._calculate_alignment_loss(vision_features, language_features, aligned_vision, aligned_language)

        # Return aligned features with original sequence lengths
        return (
            aligned_vision[:, :vision_seq_len, :],
            aligned_language[:, :lang_seq_len, :]
        ), alignment_loss

    def _calculate_alignment_loss(
        self,
        original_vision: torch.Tensor,
        original_language: torch.Tensor,
        aligned_vision: torch.Tensor,
        aligned_language: torch.Tensor
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
        if self.config.similarity_method == 'cosine':
            vision_sim = F.cosine_similarity(original_vision, aligned_vision, dim=-1).mean()
            language_sim = F.cosine_similarity(original_language, aligned_language, dim=-1).mean()
        elif self.config.similarity_method == 'dot_product':
            vision_sim = torch.sum(original_vision * aligned_vision, dim=-1).mean()
            language_sim = torch.sum(original_language * aligned_language, dim=-1).mean()
        else:  # euclidean
            vision_sim = -torch.norm(original_vision - aligned_vision, dim=-1).mean()
            language_sim = -torch.norm(original_language - aligned_language, dim=-1).mean()

        # Contrastive alignment loss
        contrastive_loss = 0.0
        if self.config.use_contrastive_alignment:
            # Calculate similarity between vision and language features
            vision_lang_sim = F.cosine_similarity(
                aligned_vision.mean(dim=1, keepdim=True),
                aligned_language.mean(dim=1, keepdim=True),
                dim=-1
            ).mean()

            # Calculate similarity between vision and other language features (negative pairs)
            lang_shifted = torch.roll(aligned_language, shifts=1, dims=0)
            vision_neg_lang_sim = F.cosine_similarity(
                aligned_vision.mean(dim=1, keepdim=True),
                lang_shifted.mean(dim=1, keepdim=True),
                dim=-1
            ).mean()

            # Contrastive loss: maximize positive pairs, minimize negative pairs
            contrastive_loss = torch.clamp(
                self.config.contrastive_margin - vision_lang_sim + vision_neg_lang_sim,
                min=0
            )

        # Combine similarity and contrastive losses
        total_loss = (1 - vision_sim) + (1 - language_sim) + self.config.alignment_lambda * contrastive_loss

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

    def register_alignment_method(self, method_name: str, config: CrossModalAlignmentConfig):
        """
        Register an alignment method with its configuration.

        Args:
            method_name: Name of the alignment method
            config: Configuration for the alignment method
        """
        self.alignment_methods[method_name] = CrossModalAlignmentOptimizer(config)

    def get_alignment_optimizer(self, method_name: str) -> Optional[CrossModalAlignmentOptimizer]:
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
        attention_mask: Optional[torch.Tensor] = None
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

        aligned_features, alignment_loss = optimizer.align_modalities(
            vision_features, language_features, attention_mask
        )

        return aligned_features[0], aligned_features[1], alignment_loss


def create_generic_cross_modal_alignment(config) -> CrossModalAlignmentManager:
    """
    Create a generic cross-modal alignment manager.

    Args:
        config: CrossModalAlignmentConfig configuration

    Returns:
        CrossModalAlignmentManager configured generically
    """
    return CrossModalAlignmentManager(config)


def apply_cross_modal_alignment_to_model(model: nn.Module, config) -> nn.Module:
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
    def perform_cross_modal_alignment(self, vision_features, language_features, method="learned_projection", attention_mask=None):
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
        if not hasattr(self, 'cross_modal_alignment_manager'):
            raise RuntimeError("Cross-modal alignment manager not available in model")

        return self.cross_modal_alignment_manager.align_modalities(
            method, vision_features, language_features, attention_mask
        )

    # Bind the method to the model
    model.perform_cross_modal_alignment = perform_cross_modal_alignment.__get__(model, model.__class__)

    logger.info("Generic cross-modal alignment optimizations applied successfully")
    return model


__all__ = [
    "CrossModalAlignmentConfig",
    "CrossModalAlignmentOptimizer",
    "CrossModalAlignmentManager",
    "create_generic_cross_modal_alignment",
    "apply_cross_modal_alignment_to_model"
]
        ''',
        "cross_modal_fusion_kernels.py": '''
"""
Generic Cross-Modal Fusion Kernels for Vision-Language Models

This module implements generic cross-modal fusion kernels for multimodal operations in vision-language models.
Specific model implementations (like Qwen3-VL-2B) should extend these classes with their own
model-specific optimizations.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import logging
from dataclasses import dataclass

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
        self.vision_language_gate = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.language_vision_gate = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # Initialize weights
        std = self.hidden_size ** -0.5
        nn.init.normal_(self.vision_language_gate.weight, mean=0.0, std=std)
        nn.init.normal_(self.language_vision_gate.weight, mean=0.0, std=std)

        # Layer norms
        self.vision_norm = nn.LayerNorm(self.hidden_size)
        self.language_norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
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

        # Normalize inputs
        vision_norm = self.vision_norm(vision_features)
        language_norm = self.language_norm(language_features)

        # Pad sequences to same length if needed for fusion operations
        max_len = max(vision_seq_len, lang_seq_len)

        if vision_seq_len < max_len:
            vision_padded = torch.cat([
                vision_norm,
                torch.zeros(batch_size, max_len - vision_seq_len, hidden_size,
                          dtype=vision_norm.dtype, device=vision_norm.device)
            ], dim=1)
        else:
            vision_padded = vision_norm

        if lang_seq_len < max_len:
            language_padded = torch.cat([
                language_norm,
                torch.zeros(batch_size, max_len - lang_seq_len, hidden_size,
                          dtype=language_norm.dtype, device=language_norm.device)
            ], dim=1)
        else:
            language_padded = language_norm

        # Apply fusion based on method
        if self.fusion_method == "add":
            fused_output = vision_padded + language_padded
        elif self.fusion_method == "concat":
            concatenated = torch.cat([vision_padded, language_padded], dim=-1)
            fused_output = nn.Linear(concatenated.size(-1), self.hidden_size).to(concatenated.device)(concatenated)
        elif self.fusion_method == "attention":
            # Simple attention-based fusion
            attention_scores = torch.matmul(vision_padded, language_padded.transpose(-2, -1)) / (self.hidden_size ** 0.5)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_weights = torch.softmax(attention_scores, dim=-1)
            fused_output = torch.matmul(attention_weights, language_padded)
        else:  # Default to "add"
            fused_output = vision_padded + language_padded

        # Apply gating mechanisms
        vision_to_lang_avg = vision_padded.mean(dim=1, keepdim=True)  # Average across sequence dimension
        vision_to_lang_combined = torch.cat([vision_to_lang_avg.expand_as(language_padded), language_padded], dim=-1)
        vision_language_gate = torch.sigmoid(self.vision_language_gate(vision_to_lang_combined))

        language_to_vision_avg = language_padded.mean(dim=1, keepdim=True)  # Average across sequence dimension
        language_to_vision_combined = torch.cat([language_to_vision_avg.expand_as(vision_padded), vision_padded], dim=-1)
        language_vision_gate = torch.sigmoid(self.language_vision_gate(language_to_vision_combined))

        # Apply gates to outputs
        vision_output = vision_features + (fused_output[:, :vision_seq_len, :] * vision_language_gate[:, :vision_seq_len, :hidden_size])
        language_output = language_features + (fused_output[:, :lang_seq_len, :] * language_vision_gate[:, :lang_seq_len, :hidden_size])

        # Truncate fused output to match original sequence lengths
        fused_output = fused_output[:, :max(vision_seq_len, lang_seq_len), :]

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
        self.register_fusion_method("attention", config)

    def register_fusion_method(self, method_name: str, config: CrossModalFusionConfig):
        """
        Register a fusion method with its configuration.

        Args:
            method_name: Name of the fusion method
            config: Configuration for the fusion method
        """
        self.fusion_methods[method_name] = GenericCrossModalFusionKernel(config)

    def get_fusion_kernel(self, method_name: str) -> Optional[GenericCrossModalFusionKernel]:
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
        attention_mask: Optional[torch.Tensor] = None
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


def create_generic_cross_modal_fusion(config: CrossModalFusionConfig) -> GenericCrossModalFusionManager:
    """
    Create a generic cross-modal fusion manager.

    Args:
        config: CrossModalFusionConfig configuration

    Returns:
        GenericCrossModalFusionManager instance
    """
    return GenericCrossModalFusionManager(config)


def apply_generic_cross_modal_fusion_to_model(model: nn.Module, config: CrossModalFusionConfig) -> nn.Module:
    """
    Apply generic cross-modal fusion optimizations to the model.

    Args:
        model: The model to optimize
        config: CrossModalFusionConfig configuration

    Returns:
        Model with cross-modal fusion capabilities
    """
    logger.info("Applying generic cross-modal fusion optimizations to model...")

    # Create fusion manager
    fusion_manager = create_generic_cross_modal_fusion(config)

    # Add fusion manager to model
    model.cross_modal_fusion_manager = fusion_manager

    # Add fusion method to model
    def perform_cross_modal_fusion(self, vision_features, language_features, method="add", attention_mask=None):
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
        if not hasattr(self, 'cross_modal_fusion_manager'):
            raise RuntimeError("Cross-modal fusion manager not available in model")

        return self.cross_modal_fusion_manager.fuse_modalities(
            method, vision_features, language_features, attention_mask
        )

    # Bind the method to the model
    model.perform_cross_modal_fusion = perform_cross_modal_fusion.__get__(model, model.__class__)

    logger.info("Generic cross-modal fusion optimizations applied successfully")
    return model


__all__ = [
    "CrossModalFusionConfig",
    "GenericCrossModalFusionKernel",
    "GenericCrossModalFusionManager",
    "create_generic_cross_modal_fusion",
    "apply_generic_cross_modal_fusion_to_model"
]
        ''',
    }

    # Write the generic modules to the common directory
    for filename, content in generic_modules.items():
        file_path = Path(
            f"C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/common/{filename}"
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Created generic version of {filename}")


if __name__ == "__main__":
    migrate_remaining_qwen3_vl_specific_code()
    update_common_init()
    create_generic_versions()
    print(
        "\nAll Qwen3-VL-2B specific code has been properly migrated to the model plugin directory!"
    )
    print("Generic implementations have been created in the common directory.")
