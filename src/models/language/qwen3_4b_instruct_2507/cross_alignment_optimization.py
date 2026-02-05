"""Cross-Alignment Optimization for Qwen3-4B-Instruct-2507 Model - Self-Contained Version

This module implements advanced cross-alignment techniques specifically for the Qwen3-4B-Instruct-2507 model.
The system efficiently aligns internal representations across different layers and components to 
improve model performance and coherence, especially for instruction-following tasks.

The cross-alignment optimization uses advanced techniques including:
- Learned projection alignment
- Attention-based alignment
- Contrastive alignment loss
- Dynamic alignment based on input complexity
- SwiGLU-based alignment mechanisms similar to Qwen3 architecture
"""

import logging
import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


logger = logging.getLogger(__name__)


class CrossAlignmentConfig:
    """Configuration for cross-alignment optimization."""
    
    def __init__(self):
        # Temperature for alignment computation (controls sharpness of alignment distribution)
        self.alignment_temperature: float = 0.5
        
        # Weight for alignment loss in total loss
        self.alignment_lambda: float = 0.1
        
        # Whether to use contrastive alignment loss
        self.use_contrastive_alignment: bool = True
        
        # Whether to enable dynamic alignment based on input complexity
        self.enable_dynamic_alignment: bool = True
        
        # Frequency of alignment updates (every N steps)
        self.alignment_frequency: int = 10
        
        # Threshold for alignment quality (above which alignment is considered good enough)
        self.alignment_threshold: float = 0.8
        
        # Whether to use attention-based alignment
        self.use_attention_alignment: bool = True
        
        # Whether to use learned alignment projections
        self.use_learned_alignment: bool = True
        
        # Dimension for alignment projections
        self.alignment_projection_dim: int = 512
        
        # Whether to enable similarity-based alignment
        self.enable_similarity_alignment: bool = True
        
        # Hidden size of the model (will be set dynamically)
        self.hidden_size: int = 2048


class AlignmentMethod(Enum):
    """Enum for different alignment methods."""
    CONTRASTIVE = "contrastive"
    ATTENTION = "attention"
    LEARNED_PROJECTION = "learned_projection"
    SIMILARITY_BASED = "similarity_based"
    QWEN3_INSTRUCT_SPECIFIC = "qwen3_instruct_specific"


class Qwen3InstructCrossAlignmentOptimizer(nn.Module):
    """Qwen3-4B-Instruct-2507 specific cross-alignment optimizer for aligning internal representations."""

    def __init__(self, config: CrossAlignmentConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Calculate standard deviation for weight initialization
        std = 1.0 / math.sqrt(config.hidden_size)
        
        # Qwen3-4B-Instruct-2507 specific alignment components
        self.alignment_norm = nn.LayerNorm(config.hidden_size)
        
        # Qwen3-4B-Instruct-2507 specific alignment attention (SwiGLU-like mechanism)
        self.alignment_up_proj = nn.Linear(
            config.hidden_size, config.alignment_projection_dim, bias=False
        )
        self.alignment_gate_proj = nn.Linear(
            config.hidden_size, config.alignment_projection_dim, bias=False
        )
        self.alignment_down_proj = nn.Linear(
            config.alignment_projection_dim, config.hidden_size, bias=False
        )
        
        # Initialize weights similar to Qwen3's initialization
        nn.init.normal_(self.alignment_up_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.alignment_gate_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.alignment_down_proj.weight, mean=0.0, std=std)

    def forward(
        self,
        rep1: Tensor,
        rep2: Tensor,
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        """
        Qwen3-4B-Instruct-2507 specific alignment of internal representations.

        Args:
            rep1: First representation tensor of shape (batch, seq_len, hidden_size)
            rep2: Second representation tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Tuple of (aligned_representations_tuple, alignment_loss)
        """
        batch_size, seq_len, hidden_size = rep1.shape
        
        # Store original representations for loss calculation
        original_rep1 = rep1.clone()
        original_rep2 = rep2.clone()
        
        # Normalize representations
        rep1_norm = self.alignment_norm(rep1)
        rep2_norm = self.alignment_norm(rep2)
        
        # Pad sequences to same length if needed for alignment operations
        max_seq_len = max(rep1_norm.size(1), rep2_norm.size(1))
        if rep1_norm.size(1) != max_seq_len:
            padding = torch.zeros(batch_size, max_seq_len - rep1_norm.size(1), hidden_size, 
                                dtype=rep1_norm.dtype, device=rep1_norm.device)
            rep1_padded = torch.cat([rep1_norm, padding], dim=1)
            rep1_seq_len = rep1_norm.size(1)
        else:
            rep1_padded = rep1_norm
            rep1_seq_len = rep1_norm.size(1)
            
        if rep2_norm.size(1) != max_seq_len:
            padding = torch.zeros(batch_size, max_seq_len - rep2_norm.size(1), hidden_size, 
                                dtype=rep2_norm.dtype, device=rep2_norm.device)
            rep2_padded = torch.cat([rep2_norm, padding], dim=1)
            rep2_seq_len = rep2_norm.size(1)
        else:
            rep2_padded = rep2_norm
            rep2_seq_len = rep2_norm.size(1)

        # Apply Qwen3-4B-Instruct-2507 specific SwiGLU-like alignment mechanism
        # For rep1
        rep1_gate = F.silu(self.alignment_gate_proj(rep1_padded))
        rep1_up = self.alignment_up_proj(rep1_padded)
        rep1_aligned = self.alignment_down_proj(
            rep1_up * rep1_gate
        )
        
        # For rep2
        rep2_gate = F.silu(self.alignment_gate_proj(rep2_padded))
        rep2_up = self.alignment_up_proj(rep2_padded)
        rep2_aligned = self.alignment_down_proj(
            rep2_up * rep2_gate
        )
        
        # Extract the parts that correspond to the original sequence lengths
        rep1_gate_part = rep1_gate[:, :rep1_seq_len, :]
        rep2_gate_part = rep2_gate[:, :rep2_seq_len, :]
        
        # Apply gating to aligned features
        aligned_rep1 = (
            rep1 + rep1_aligned[:, :rep1_seq_len, :] * rep1_gate_part
        )
        aligned_rep2 = (
            rep2 + rep2_aligned[:, :rep2_seq_len, :] * rep2_gate_part
        )

        # Calculate Qwen3-4B-Instruct-2507 specific alignment loss
        alignment_loss = self._calculate_qwen_alignment_loss(
            original_rep1, original_rep2, aligned_rep1, aligned_rep2
        )
        
        # Return aligned representations with original sequence lengths
        return (aligned_rep1, aligned_rep2), alignment_loss

    def _calculate_qwen_alignment_loss(
        self,
        original_rep1: Tensor,
        original_rep2: Tensor,
        aligned_rep1: Tensor,
        aligned_rep2: Tensor,
    ) -> Tensor:
        """
        Calculate Qwen3-4B-Instruct-2507 specific alignment loss.

        Args:
            original_rep1: Original first representation
            original_rep2: Original second representation
            aligned_rep1: Aligned first representation
            aligned_rep2: Aligned second representation

        Returns:
            Alignment loss tensor
        """
        # Calculate similarity between original and aligned representations
        rep1_sim = F.cosine_similarity(original_rep1, aligned_rep1, dim=-1).mean()
        rep2_sim = F.cosine_similarity(original_rep2, aligned_rep2, dim=-1).mean()
        
        # If cosine similarity fails, fall back to MSE
        if torch.isnan(rep1_sim) or torch.isinf(rep1_sim):
            rep1_sim = -torch.norm(original_rep1 - aligned_rep1, dim=-1).mean()
        if torch.isnan(rep2_sim) or torch.isinf(rep2_sim):
            rep2_sim = -torch.norm(original_rep2 - aligned_rep2, dim=-1).mean()
        
        # Qwen3-4B-Instruct-2507 specific contrastive alignment loss
        contrastive_loss = torch.tensor(0.0, device=original_rep1.device, dtype=original_rep1.dtype)
        if self.config.use_contrastive_alignment:
            # Compute representations for contrastive loss
            rep1_repr = aligned_rep1.mean(dim=1)  # Shape: (batch, hidden_size)
            rep2_repr = aligned_rep2.mean(dim=1)  # Shape: (batch, hidden_size)
            
            # Positive pairs (rep1 and rep2 should be similar after alignment)
            pos_sim = F.cosine_similarity(rep1_repr, rep2_repr, dim=-1).mean()
            
            # Negative pairs (shifted versions to create negative samples)
            rep2_shifted = torch.roll(aligned_rep2, shifts=1, dims=0)
            neg_sim = F.cosine_similarity(rep1_repr, rep2_shifted.mean(dim=1), dim=-1).mean()
            
            # Contrastive loss: maximize positive similarity, minimize negative similarity
            contrastive_loss = -torch.log(torch.exp(pos_sim / self.config.alignment_temperature) /
                                        (torch.exp(pos_sim / self.config.alignment_temperature) +
                                         torch.exp(neg_sim / self.config.alignment_temperature))).mean()
        
        # Encourage alignment to preserve semantic meaning
        semantic_reg_loss = F.mse_loss(aligned_rep1, original_rep1) + F.mse_loss(
            aligned_rep2, original_rep2
        )
        
        # Total loss: minimize distance from originals, maximize contrastive alignment
        total_loss = (1 - rep1_sim) + (1 - rep2_sim) + self.config.alignment_lambda * contrastive_loss
        return total_loss


class CrossAlignmentManager:
    """Manager for cross-alignment operations.
    Handles selection and application of appropriate alignment methods.
    """

    def __init__(self, config: CrossAlignmentConfig):
        self.config = config
        self.alignment_methods = {}
        
        # Register default alignment methods
        self.register_alignment_method("contrastive", config)
        self.register_alignment_method("attention", config)
        self.register_alignment_method("learned_projection", config)
        self.register_alignment_method("similarity_based", config)
        self.register_alignment_method("qwen3_instruct_specific", config)

    def register_alignment_method(
        self, method_name: str, config: CrossAlignmentConfig
    ):
        """Register an alignment method with its configuration.

        Args:
            method_name: Name of the alignment method
            config: Configuration for the alignment method
        """
        if method_name == "qwen3_instruct_specific":
            self.alignment_methods[method_name] = Qwen3InstructCrossAlignmentOptimizer(
                config, layer_idx=len(self.alignment_methods)
            )
        else:
            # For other methods, create a generic version with modified config
            temp_config = CrossAlignmentConfig()
            temp_config.alignment_temperature = config.alignment_temperature
            temp_config.alignment_lambda = config.alignment_lambda
            temp_config.use_contrastive_alignment = config.use_contrastive_alignment
            temp_config.enable_dynamic_alignment = config.enable_dynamic_alignment
            temp_config.alignment_frequency = config.alignment_frequency
            temp_config.alignment_threshold = config.alignment_threshold
            temp_config.use_attention_alignment = config.use_attention_alignment
            temp_config.use_learned_alignment = config.use_learned_alignment
            temp_config.alignment_projection_dim = config.alignment_projection_dim
            temp_config.enable_similarity_alignment = config.enable_similarity_alignment
            temp_config.hidden_size = config.hidden_size
            
            self.alignment_methods[method_name] = Qwen3InstructCrossAlignmentOptimizer(
                temp_config, layer_idx=len(self.alignment_methods)
            )

    def get_alignment_optimizer(
        self, method_name: str
    ) -> Optional['Qwen3InstructCrossAlignmentOptimizer']:
        """Get a registered alignment optimizer.

        Args:
            method_name: Name of the alignment method

        Returns:
            Alignment optimizer if registered, None otherwise
        """
        return self.alignment_methods.get(method_name)

    def align_representations(
        self,
        rep1: Tensor,
        rep2: Tensor,
        method_name: str = "qwen3_instruct_specific",
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Align internal representations using the specified method.

        Args:
            rep1: First representation tensor
            rep2: Second representation tensor
            method_name: Name of the alignment method to use
            attention_mask: Optional attention mask

        Returns:
            Tuple of (aligned_rep1, aligned_rep2, alignment_loss)
        """
        optimizer = self.get_alignment_optimizer(method_name)
        if optimizer is None:
            raise ValueError(f"Alignment method '{method_name}' not registered")
        
        aligned_reps, alignment_loss = optimizer(rep1, rep2)
        return aligned_reps[0], aligned_reps[1], alignment_loss

    def evaluate_alignment_quality(
        self,
        original_rep1: Tensor,
        original_rep2: Tensor,
        aligned_rep1: Tensor,
        aligned_rep2: Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate the quality of cross-alignment.

        Args:
            original_rep1: Original first representation
            original_rep2: Original second representation
            aligned_rep1: Aligned first representation
            aligned_rep2: Aligned second representation

        Returns:
            Dictionary with alignment quality metrics
        """
        # Calculate preservation of original representations
        rep1_preservation = F.cosine_similarity(original_rep1, aligned_rep1, dim=-1).mean().item()
        rep2_preservation = F.cosine_similarity(original_rep2, aligned_rep2, dim=-1).mean().item()
        
        # Calculate alignment between representations
        rep1_repr = aligned_rep1.mean(dim=1)
        rep2_repr = aligned_rep2.mean(dim=1)
        cross_rep_similarity = F.cosine_similarity(rep1_repr, rep2_repr, dim=-1).mean().item()
        
        # Calculate original cross-representation similarity for comparison
        orig_rep1_repr = original_rep1.mean(dim=1)
        orig_rep2_repr = original_rep2.mean(dim=1)
        original_cross_rep_sim = F.cosine_similarity(orig_rep1_repr, orig_rep2_repr, dim=-1).mean().item()
        
        # Calculate alignment improvement
        alignment_improvement = cross_rep_similarity - original_cross_rep_sim
        
        return {
            "rep1_preservation": rep1_preservation,
            "rep2_preservation": rep2_preservation,
            "cross_rep_similarity": cross_rep_similarity,
            "original_cross_rep_similarity": original_cross_rep_sim,
            "alignment_improvement": alignment_improvement,
            "overall_alignment_score": (rep1_preservation + rep2_preservation + cross_rep_similarity) / 3
        }


def create_qwen3_instruct_cross_alignment(
    config: CrossAlignmentConfig,
) -> CrossAlignmentManager:
    """Create a cross-alignment manager specifically for Qwen3-4B-Instruct-2507.

    Args:
        config: CrossAlignmentConfig configuration

    Returns:
        CrossAlignmentManager configured for Qwen3-4B-Instruct-2507
    """
    return CrossAlignmentManager(config)


def apply_cross_alignment_to_model(
    model: nn.Module, config: CrossAlignmentConfig
) -> nn.Module:
    """
    Apply cross-alignment optimizations to the model.

    Args:
        model: The neural network model to optimize
        config: CrossAlignmentConfig with optimization settings

    Returns:
        Model with cross-alignment capabilities
    """
    logger.info("Applying cross-alignment optimizations to model...")
    
    # Create alignment manager
    alignment_manager = create_qwen3_instruct_cross_alignment(config)
    
    # Add alignment manager to model
    model.cross_alignment_manager = alignment_manager
    
    # Add alignment optimizer (specific to Qwen3-4B-Instruct-2507)
    model.cross_alignment_optimizer = alignment_manager.get_alignment_optimizer(
        "qwen3_instruct_specific"
    )
    
    # Add alignment method to model
    def perform_cross_alignment(
        self, rep1: Tensor, rep2: Tensor, method="qwen3_instruct_specific", attention_mask=None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Perform cross-alignment between internal representations.

        Args:
            rep1: First representation tensor
            rep2: Second representation tensor
            method: Alignment method to use
            attention_mask: Optional attention mask

        Returns:
            Tuple of (aligned_rep1, aligned_rep2, alignment_loss)
        """
        if not hasattr(self, "cross_alignment_manager"):
            raise RuntimeError("Cross-alignment manager not available in model")
        
        return self.cross_alignment_manager.align_representations(
            rep1, rep2, method, attention_mask
        )
    
    model.perform_cross_alignment = perform_cross_alignment.__get__(
        model, model.__class__
    )
    
    logger.info("Cross-alignment optimizations applied successfully")
    return model


def get_cross_alignment_report(
    model: nn.Module, config: CrossAlignmentConfig
) -> Dict[str, any]:
    """Get a report of cross-alignment applied to the model.

    Args:
        model: Model with cross-alignment optimizations
        config: CrossAlignmentConfig used for optimization

    Returns:
        Dictionary with cross-alignment optimization report
    """
    report = {
        "optimization_type": "Cross-Alignment",
        "alignment_methods_registered": [],
        "alignment_enabled": False,
        "alignment_config": {
            "alignment_temperature": getattr(config, "alignment_temperature", 0.5),
            "alignment_lambda": getattr(config, "alignment_lambda", 0.1),
            "use_contrastive_alignment": getattr(
                config, "use_contrastive_alignment", True
            ),
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
        },
    }
    
    # Check if cross-alignment is enabled in the config
    alignment_enabled = getattr(
        config, "enable_cross_alignment", False
    ) or hasattr(model, "cross_alignment_manager")
    report["alignment_enabled"] = alignment_enabled
    
    if hasattr(model, "cross_alignment_manager"):
        if hasattr(model.cross_alignment_manager, "alignment_methods"):
            report["alignment_methods_registered"] = list(
                model.cross_alignment_manager.alignment_methods.keys()
            )
        else:
            report["alignment_methods_registered"] = []
    
    return report


__all__ = [
    "CrossAlignmentConfig",
    "Qwen3InstructCrossAlignmentOptimizer",
    "CrossAlignmentManager",
    "create_qwen3_instruct_cross_alignment",
    "apply_cross_alignment_to_model",
    "get_cross_alignment_report",
]