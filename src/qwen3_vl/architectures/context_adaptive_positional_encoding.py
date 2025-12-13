"""
Context-adaptive positional representations for Qwen3-VL architecture.
This module implements learned context-adaptive positional representations
that replace fixed positional encodings.
"""
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextAdaptivePositionalEncoding(nn.Module):
    """
    Learned context-adaptive positional representations that adapt based on the input.
    This replaces fixed positional encodings with learned representations that can
    adapt based on context.
    """
    def __init__(self, hidden_size: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Learned positional embeddings (replaces fixed encodings)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
        
        # Context adaptation network - learns to modify positional encodings based on input context
        self.context_adaptation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()  # Output gate to control how much to adapt the position encoding
        )
        
        # Context query network - learns to generate context-aware positional queries
        self.context_query = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize position embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the position embeddings."""
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, 
                hidden_states: torch.Tensor, 
                position_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for context-adaptive positional encoding.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            position_ids: Position IDs for each token (optional)
            attention_mask: Attention mask to identify valid tokens (optional)
            
        Returns:
            Tensor with context-adaptive positional encodings added
        """
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        
        # Ensure position IDs are within bounds
        position_ids = torch.clamp(position_ids, 0, self.max_seq_len - 1)
        
        # Get base positional embeddings
        base_pos_embeddings = self.position_embeddings(position_ids)
        
        # Compute context-aware adaptations
        # Use the input hidden states to determine how to adapt positional encodings
        context_signal = hidden_states  # Use the input as context
        
        # Apply context adaptation to modify positional embeddings
        # This learns to adjust positional encodings based on the input context
        adaptation_weights = self.context_adaptation(context_signal)
        
        # Adapt the base positional embeddings based on context
        adapted_pos_embeddings = base_pos_embeddings * adaptation_weights
        
        # Add the adapted positional embeddings to the hidden states
        output = hidden_states + adapted_pos_embeddings
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class VisionContextAdaptivePositionalEncoding(nn.Module):
    """
    Context-adaptive positional encoding for vision components.
    Specifically designed for vision transformer patches with 2D spatial awareness.
    """
    def __init__(self, hidden_size: int, num_patches_per_dim: int = 24, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patches_per_dim = num_patches_per_dim
        self.num_patches = num_patches_per_dim ** 2
        
        # Learned 2D positional embeddings for vision
        self.row_embeddings = nn.Embedding(num_patches_per_dim, hidden_size // 2)
        self.col_embeddings = nn.Embedding(num_patches_per_dim, hidden_size // 2)
        
        # Context adaptation network for vision
        self.context_adaptation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Spatial context encoder - learns spatial relationships
        self.spatial_context_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, hidden_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the vision positional embeddings."""
        nn.init.normal_(self.row_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.col_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                patch_positions: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass for vision context-adaptive positional encoding.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, num_patches, hidden_size)
            patch_positions: Tuple of (row_ids, col_ids) for each patch (optional)
            
        Returns:
            Tensor with context-adaptive positional encodings added
        """
        batch_size, num_patches, hidden_size = hidden_states.size()
        
        # Calculate 2D positions if not provided
        if patch_positions is None:
            # Calculate row and column indices based on the number of patches per dimension
            device = hidden_states.device
            grid_size = int(math.sqrt(num_patches))
            
            # Generate row and column indices
            row_indices = torch.arange(grid_size, device=device).unsqueeze(1).expand(-1, grid_size).flatten()
            col_indices = torch.arange(grid_size, device=device).unsqueeze(0).expand(grid_size, -1).flatten()
            
            row_ids = row_indices[:num_patches]
            col_ids = col_indices[:num_patches]
        else:
            row_ids, col_ids = patch_positions
        
        # Ensure position IDs are within bounds
        row_ids = torch.clamp(row_ids, 0, self.num_patches_per_dim - 1)
        col_ids = torch.clamp(col_ids, 0, self.num_patches_per_dim - 1)
        
        # Get 2D positional embeddings
        row_embeddings = self.row_embeddings(row_ids)
        col_embeddings = self.col_embeddings(col_ids)
        base_pos_embeddings = torch.cat([row_embeddings, col_embeddings], dim=-1)
        
        # Apply context adaptation based on input features
        adaptation_weights = self.context_adaptation(hidden_states)
        adapted_pos_embeddings = base_pos_embeddings * adaptation_weights
        
        # Add spatial context encoding
        spatial_context = self.spatial_context_encoder(adapted_pos_embeddings)
        
        # Combine everything
        output = hidden_states + adapted_pos_embeddings + spatial_context
        output = self.dropout(output)
        
        return output


class CrossModalContextAdaptivePositionalEncoding(nn.Module):
    """
    Cross-modal context-adaptive positional encoding that adapts based on both
    vision and language inputs.
    """
    def __init__(self, hidden_size: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Separate position embeddings for different modalities
        self.text_pos_embeddings = nn.Embedding(max_seq_len, hidden_size)
        self.vision_pos_embeddings = nn.Embedding(max_seq_len, hidden_size)
        
        # Cross-modal context fusion network
        self.cross_modal_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Modality-specific adaptation networks
        self.text_adaptation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
        
        self.vision_adaptation = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the cross-modal positional embeddings."""
        nn.init.normal_(self.text_pos_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.vision_pos_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self,
                text_hidden_states: torch.Tensor,
                vision_hidden_states: torch.Tensor,
                text_position_ids: Optional[torch.LongTensor] = None,
                vision_position_ids: Optional[torch.LongTensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal context-adaptive positional encoding.
        
        Args:
            text_hidden_states: Text tensor of shape (batch_size, text_seq_len, hidden_size)
            vision_hidden_states: Vision tensor of shape (batch_size, vision_seq_len, hidden_size)
            text_position_ids: Position IDs for text tokens (optional)
            vision_position_ids: Position IDs for vision patches (optional)
            
        Returns:
            Tuple of (adapted_text_states, adapted_vision_states)
        """
        batch_size, text_seq_len, _ = text_hidden_states.size()
        _, vision_seq_len, _ = vision_hidden_states.size()
        
        # Generate position IDs if not provided
        if text_position_ids is None:
            text_position_ids = torch.arange(text_seq_len, dtype=torch.long, device=text_hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        if vision_position_ids is None:
            vision_position_ids = torch.arange(vision_seq_len, dtype=torch.long, device=vision_hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        
        # Ensure position IDs are within bounds
        text_position_ids = torch.clamp(text_position_ids, 0, self.max_seq_len - 1)
        vision_position_ids = torch.clamp(vision_position_ids, 0, self.max_seq_len - 1)
        
        # Get base positional embeddings
        base_text_pos = self.text_pos_embeddings(text_position_ids)
        base_vision_pos = self.vision_pos_embeddings(vision_position_ids)
        
        # Compute cross-modal context
        # Average representations for cross-modal context
        text_context = text_hidden_states.mean(dim=1, keepdim=True)  # (batch, 1, hidden)
        vision_context = vision_hidden_states.mean(dim=1, keepdim=True)  # (batch, 1, hidden)
        
        # Create cross-modal context by combining the averaged representations
        # Repeat each context to match the sequence lengths of the respective modalities
        text_context_expanded = text_context.expand(-1, text_seq_len, -1)  # (batch, text_seq_len, hidden)
        vision_context_expanded = vision_context.expand(-1, vision_seq_len, -1)  # (batch, vision_seq_len, hidden)
        
        # Combine contexts for cross-modal fusion - create a representation that captures both modalities
        # For text tokens, combine text and vision context
        text_vision_context = vision_context.expand(-1, text_seq_len, -1)  # Repeat vision context for text seq
        text_combined_context = torch.cat([text_context_expanded, text_vision_context], dim=-1)
        
        # For vision tokens, combine vision and text context  
        vision_text_context = text_context.expand(-1, vision_seq_len, -1)  # Repeat text context for vision seq
        vision_combined_context = torch.cat([vision_context_expanded, vision_text_context], dim=-1)
        
        # Apply cross-modal fusion separately for each modality
        text_cross_modal_weights = self.cross_modal_fusion(text_combined_context)
        vision_cross_modal_weights = self.cross_modal_fusion(vision_combined_context)
        
        # Apply modality-specific adaptations
        text_adaptation = self.text_adaptation(text_hidden_states)
        vision_adaptation = self.vision_adaptation(vision_hidden_states)
        
        # Adapt positional embeddings with cross-modal context
        adapted_text_pos = base_text_pos * text_adaptation * text_cross_modal_weights
        adapted_vision_pos = base_vision_pos * vision_adaptation * vision_cross_modal_weights
        
        # Add positional encodings to hidden states
        text_output = text_hidden_states + adapted_text_pos
        vision_output = vision_hidden_states + adapted_vision_pos
        
        # Apply dropout
        text_output = self.dropout(text_output)
        vision_output = self.dropout(vision_output)
        
        return text_output, vision_output


# Integration utilities for Qwen3-VL architecture
class Qwen3VLContextAdaptivePositionalProcessor(nn.Module):
    """
    Main processor for integrating context-adaptive positional representations
    into the Qwen3-VL architecture.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Initialize the context-adaptive positional encodings
        self.text_positional_encoder = ContextAdaptivePositionalEncoding(
            hidden_size=self.hidden_size,
            max_seq_len=config.max_position_embeddings,
            dropout=0.1
        )
        
        self.vision_positional_encoder = VisionContextAdaptivePositionalEncoding(
            hidden_size=config.vision_hidden_size,
            num_patches_per_dim=config.vision_image_size // config.vision_patch_size,
            dropout=0.1
        )
        
        if hasattr(config, 'use_cross_modal_positional_encoding') and config.use_cross_modal_positional_encoding:
            self.cross_modal_positional_encoder = CrossModalContextAdaptivePositionalEncoding(
                hidden_size=self.hidden_size,
                max_seq_len=config.max_position_embeddings,
                dropout=0.1
            )
    
    def forward(self, 
                text_hidden_states: Optional[torch.Tensor] = None,
                vision_hidden_states: Optional[torch.Tensor] = None,
                text_position_ids: Optional[torch.LongTensor] = None,
                vision_position_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Process hidden states with context-adaptive positional encodings.
        
        Args:
            text_hidden_states: Text tensor of shape (batch_size, text_seq_len, hidden_size)
            vision_hidden_states: Vision tensor of shape (batch_size, vision_seq_len, vision_hidden_size)
            text_position_ids: Position IDs for text tokens (optional)
            vision_position_ids: Position IDs for vision patches (optional)
            attention_mask: Attention mask for text (optional)
            
        Returns:
            Tuple of (processed_text_states, processed_vision_states)
        """
        processed_text = None
        processed_vision = None
        
        if text_hidden_states is not None:
            processed_text = self.text_positional_encoder(
                hidden_states=text_hidden_states,
                position_ids=text_position_ids,
                attention_mask=attention_mask
            )
        
        if vision_hidden_states is not None:
            processed_vision = self.vision_positional_encoder(
                hidden_states=vision_hidden_states,
                patch_positions=vision_position_ids
            )
        
        # If both modalities are present and cross-modal encoding is enabled, apply it
        if (text_hidden_states is not None and 
            vision_hidden_states is not None and
            hasattr(self, 'cross_modal_positional_encoder')):
            
            processed_text, processed_vision = self.cross_modal_positional_encoder(
                text_hidden_states=processed_text,
                vision_hidden_states=processed_vision,
                text_position_ids=text_position_ids,
                vision_position_ids=vision_position_ids
            )
        
        return processed_text, processed_vision