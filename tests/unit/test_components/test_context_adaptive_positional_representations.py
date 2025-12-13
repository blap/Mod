"""
Test file for context-adaptive positional representations implementation.
This tests the learned context-adaptive positional representations functionality
that will replace fixed positional encodings in the Qwen3-VL architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from typing import Optional, Tuple
import math


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
            position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
        
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
        batch_size_text, text_seq_len, _ = text_hidden_states.size()
        batch_size_vision, vision_seq_len, _ = vision_hidden_states.size()
        
        # Generate position IDs if not provided
        if text_position_ids is None:
            text_position_ids = torch.arange(text_seq_len, dtype=torch.long, device=text_hidden_states.device).unsqueeze(0)
        if vision_position_ids is None:
            vision_position_ids = torch.arange(vision_seq_len, dtype=torch.long, device=vision_hidden_states.device).unsqueeze(0)
        
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


def test_context_adaptive_positional_encoding():
    """Test the context-adaptive positional encoding implementation."""
    hidden_size = 256
    seq_len = 64
    batch_size = 2
    
    # Create the context-adaptive positional encoding layer
    pos_encoder = ContextAdaptivePositionalEncoding(hidden_size, max_seq_len=128)
    
    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Test forward pass
    output = pos_encoder(hidden_states, position_ids)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, hidden_size), f"Expected output shape {(batch_size, seq_len, hidden_size)}, got {output.shape}"
    
    # Check that output is different from input (position encodings were added)
    assert not torch.allclose(output, hidden_states), "Output should be different from input after adding positional encodings"
    
    print("PASS: ContextAdaptivePositionalEncoding test passed")


def test_vision_context_adaptive_positional_encoding():
    """Test the vision context-adaptive positional encoding implementation."""
    hidden_size = 256
    num_patches = 256  # 16x16 grid
    batch_size = 2
    
    # Create the vision context-adaptive positional encoding layer
    vision_pos_encoder = VisionContextAdaptivePositionalEncoding(hidden_size, num_patches_per_dim=16)
    
    # Create test input
    hidden_states = torch.randn(batch_size, num_patches, hidden_size)
    
    # Test forward pass
    output = vision_pos_encoder(hidden_states)
    
    # Check output shape
    assert output.shape == (batch_size, num_patches, hidden_size), f"Expected output shape {(batch_size, num_patches, hidden_size)}, got {output.shape}"
    
    # Check that output is different from input
    assert not torch.allclose(output, hidden_states), "Output should be different from input after adding positional encodings"
    
    print("PASS: VisionContextAdaptivePositionalEncoding test passed")


def test_cross_modal_context_adaptive_positional_encoding():
    """Test the cross-modal context-adaptive positional encoding implementation."""
    hidden_size = 256
    text_seq_len = 32
    vision_seq_len = 64
    batch_size = 2
    
    # Create the cross-modal context-adaptive positional encoding layer
    cross_modal_pos_encoder = CrossModalContextAdaptivePositionalEncoding(hidden_size, max_seq_len=128)
    
    # Create test inputs
    text_hidden_states = torch.randn(batch_size, text_seq_len, hidden_size)
    vision_hidden_states = torch.randn(batch_size, vision_seq_len, hidden_size)
    
    # Test forward pass
    text_output, vision_output = cross_modal_pos_encoder(text_hidden_states, vision_hidden_states)
    
    # Check output shapes
    assert text_output.shape == (batch_size, text_seq_len, hidden_size), f"Expected text output shape {(batch_size, text_seq_len, hidden_size)}, got {text_output.shape}"
    assert vision_output.shape == (batch_size, vision_seq_len, hidden_size), f"Expected vision output shape {(batch_size, vision_seq_len, hidden_size)}, got {vision_output.shape}"
    
    # Check that outputs are different from inputs
    assert not torch.allclose(text_output, text_hidden_states), "Text output should be different from input after adding positional encodings"
    assert not torch.allclose(vision_output, vision_hidden_states), "Vision output should be different from input after adding positional encodings"
    
    print("PASS: CrossModalContextAdaptivePositionalEncoding test passed")


def test_context_adaptation_behavior():
    """Test that the context adaptation actually adapts based on input context."""
    hidden_size = 64
    seq_len = 16
    batch_size = 2
    
    pos_encoder = ContextAdaptivePositionalEncoding(hidden_size, max_seq_len=32)
    
    # Create two different input contexts
    context1 = torch.randn(batch_size, seq_len, hidden_size)
    context2 = torch.randn(batch_size, seq_len, hidden_size) * 2  # Different scale
    
    # Get outputs for both contexts
    output1 = pos_encoder(context1)
    output2 = pos_encoder(context2)
    
    # The outputs should be different due to context adaptation
    # Even with same position IDs, different contexts should lead to different positional encodings
    assert not torch.allclose(output1, output2), "Outputs should be different for different input contexts"
    
    print("PASS: Context adaptation behavior test passed")


def test_gradient_flow():
    """Test that gradients flow properly through the context-adaptive positional encoding."""
    hidden_size = 64
    seq_len = 16
    batch_size = 2
    
    pos_encoder = ContextAdaptivePositionalEncoding(hidden_size, max_seq_len=32)
    
    # Create input with gradient tracking
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Forward pass
    output = pos_encoder(hidden_states, position_ids)
    
    # Create a simple loss
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist for input and model parameters
    assert hidden_states.grad is not None, "Gradients should exist for input"
    assert pos_encoder.position_embeddings.weight.grad is not None, "Gradients should exist for position embeddings"
    assert pos_encoder.context_adaptation[0].weight.grad is not None, "Gradients should exist for context adaptation network"
    
    print("PASS: Gradient flow test passed")


def test_positional_encoding_uniqueness():
    """Test that different positions get different encodings."""
    hidden_size = 64
    seq_len = 16
    batch_size = 1
    
    pos_encoder = ContextAdaptivePositionalEncoding(hidden_size, max_seq_len=32)
    
    # Create identical input states but with different position IDs
    hidden_states = torch.ones(batch_size, seq_len, hidden_size)  # All ones to isolate positional effects
    position_ids = torch.arange(seq_len).unsqueeze(0)
    
    output = pos_encoder(hidden_states, position_ids)
    
    # Different positions should have different encodings even with identical inputs
    for i in range(seq_len - 1):
        for j in range(i + 1, seq_len):
            # Check if the encodings for different positions are different
            pos_i = output[0, i, :]
            pos_j = output[0, j, :]
            assert not torch.allclose(pos_i, pos_j), f"Position {i} and {j} should have different encodings"
    
    print("PASS: Positional encoding uniqueness test passed")


def run_all_tests():
    """Run all tests for context-adaptive positional representations."""
    print("Running tests for Context-Adaptive Positional Representations...")
    print("=" * 60)
    
    test_context_adaptive_positional_encoding()
    test_vision_context_adaptive_positional_encoding()
    test_cross_modal_context_adaptive_positional_encoding()
    test_context_adaptation_behavior()
    test_gradient_flow()
    test_positional_encoding_uniqueness()
    
    print("=" * 60)
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()