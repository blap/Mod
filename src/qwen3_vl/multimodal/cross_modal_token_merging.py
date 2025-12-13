"""Cross-modal token merging implementation for Qwen3-VL model."""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class CrossModalTokenMerger(nn.Module):
    """Cross-modal token merger that reduces computational overhead by merging tokens based on semantic similarity."""
    
    def __init__(self, config, merge_ratio: float = 0.8):
        super().__init__()
        self.merge_ratio = merge_ratio
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Projection layers for token merging
        self.text_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.vision_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Cross-modal attention mechanism
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_attention_heads,
            batch_first=True
        )
        
        # Gate mechanism to control merging
        self.merge_gate = nn.Linear(self.hidden_size * 2, 1)
        
        # Output projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
    def forward(self, 
                text_hidden_states: torch.Tensor, 
                vision_hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge tokens across modalities based on semantic similarity.
        
        Args:
            text_hidden_states: Text features [batch_size, text_seq_len, hidden_size]
            vision_hidden_states: Vision features [batch_size, vision_seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (merged_text_states, merged_vision_states)
        """
        batch_size, text_seq_len, hidden_size = text_hidden_states.shape
        _, vision_seq_len, _ = vision_hidden_states.shape
        
        # Project text and vision features
        text_proj = self.text_proj(text_hidden_states)  # [batch, text_seq, hidden]
        vision_proj = self.vision_proj(vision_hidden_states)  # [batch, vision_seq, hidden]
        
        # Compute cross-modal attention
        # Use text as query and vision as key/value
        text_to_vision_attn, _ = self.cross_attn(
            query=text_proj,
            key=vision_proj,
            value=vision_proj
        )
        
        # Use vision as query and text as key/value
        vision_to_text_attn, _ = self.cross_attn(
            query=vision_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Determine how many tokens to merge based on merge ratio
        num_text_to_merge = max(1, int(text_seq_len * (1 - self.merge_ratio)))
        num_vision_to_merge = max(1, int(vision_seq_len * (1 - self.merge_ratio)))
        
        # Select tokens with lowest attention scores for merging
        text_attn_scores = torch.norm(text_to_vision_attn, dim=-1)  # [batch, text_seq]
        vision_attn_scores = torch.norm(vision_to_text_attn, dim=-1)  # [batch, vision_seq]
        
        # Get indices of tokens with lowest attention scores (to merge)
        _, text_merge_indices = torch.topk(text_attn_scores, num_text_to_merge, dim=1, largest=False)
        _, vision_merge_indices = torch.topk(vision_attn_scores, num_vision_to_merge, dim=1, largest=False)
        
        # Create merged representations
        merged_text = self._merge_tokens(text_hidden_states, text_merge_indices)
        merged_vision = self._merge_tokens(vision_hidden_states, vision_merge_indices)
        
        # Apply output projection
        merged_text = self.out_proj(merged_text)
        merged_vision = self.out_proj(merged_vision)
        
        return merged_text, merged_vision
    
    def _merge_tokens(self, hidden_states: torch.Tensor, merge_indices: torch.Tensor) -> torch.Tensor:
        """
        Merge tokens at the specified indices to reduce sequence length.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            merge_indices: Indices of tokens to merge [batch_size, num_to_merge]
            
        Returns:
            Merged tensor [batch_size, new_seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create a mask to identify tokens to keep
        keep_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
        keep_mask.scatter_(1, merge_indices, False)
        
        # Keep tokens that are not marked for merging
        kept_tokens = hidden_states[keep_mask].view(batch_size, -1, hidden_size)
        
        # For merged tokens, create averaged representations
        merged_tokens = []
        for batch_idx in range(batch_size):
            merge_idx_batch = merge_indices[batch_idx]
            merge_token_batch = hidden_states[batch_idx, merge_idx_batch, :]  # [num_to_merge, hidden_size]
            merged_repr = torch.mean(merge_token_batch, dim=0, keepdim=True)  # [1, hidden_size]
            merged_tokens.append(merged_repr)
        
        merged_tokens_tensor = torch.stack(merged_tokens, dim=0)  # [batch_size, 1, hidden_size]
        
        # Combine kept tokens with merged representations
        combined = torch.cat([kept_tokens, merged_tokens_tensor], dim=1)
        
        return combined


class VisionCrossModalTokenMerger(CrossModalTokenMerger):
    """Cross-modal token merger specifically for vision components."""
    
    def __init__(self, config, merge_ratio: float = 0.7):
        super().__init__(config, merge_ratio)
        self.vision_hidden_size = getattr(config, 'vision_hidden_size', self.hidden_size)
        
        # Adjust for vision-specific dimensions
        self.vision_proj = nn.Linear(self.vision_hidden_size, self.hidden_size)
        self.vision_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=min(self.num_attention_heads, 8),  # Vision might use fewer heads
            batch_first=True
        )
        
    def forward(self, 
                text_hidden_states: torch.Tensor, 
                vision_hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Merge tokens across modalities with vision-specific optimizations.
        
        Args:
            text_hidden_states: Text features [batch_size, text_seq_len, hidden_size]
            vision_hidden_states: Vision features [batch_size, vision_seq_len, vision_hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (merged_text_states, merged_vision_states)
        """
        batch_size, text_seq_len, hidden_size = text_hidden_states.shape
        _, vision_seq_len, vision_hidden_size = vision_hidden_states.shape
        
        # Project text and vision features
        text_proj = self.text_proj(text_hidden_states)  # [batch, text_seq, hidden]
        vision_proj = self.vision_proj(vision_hidden_states)  # [batch, vision_seq, hidden]
        
        # Compute cross-modal attention with vision-specific optimizations
        text_to_vision_attn, _ = self.vision_cross_attn(
            query=text_proj,
            key=vision_proj,
            value=vision_proj
        )
        
        vision_to_text_attn, _ = self.vision_cross_attn(
            query=vision_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Determine how many tokens to merge based on merge ratio
        num_text_to_merge = max(1, int(text_seq_len * (1 - self.merge_ratio)))
        num_vision_to_merge = max(1, int(vision_seq_len * (1 - self.merge_ratio)))
        
        # Select tokens with lowest attention scores for merging
        text_attn_scores = torch.norm(text_to_vision_attn, dim=-1)  # [batch, text_seq]
        vision_attn_scores = torch.norm(vision_to_text_attn, dim=-1)  # [batch, vision_seq]
        
        # Get indices of tokens with lowest attention scores (to merge)
        _, text_merge_indices = torch.topk(text_attn_scores, num_text_to_merge, dim=1, largest=False)
        _, vision_merge_indices = torch.topk(vision_attn_scores, num_vision_to_merge, dim=1, largest=False)
        
        # Create merged representations
        merged_text = self._merge_tokens(text_hidden_states, text_merge_indices)
        merged_vision = self._merge_tokens_vision(vision_hidden_states, vision_merge_indices)
        
        # Apply output projection
        merged_text = self.out_proj(merged_text)
        merged_vision = self.out_proj(merged_vision)
        
        return merged_text, merged_vision
    
    def _merge_tokens_vision(self, hidden_states: torch.Tensor, merge_indices: torch.Tensor) -> torch.Tensor:
        """
        Merge vision tokens with specific optimizations for vision features.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            merge_indices: Indices of tokens to merge [batch_size, num_to_merge]
            
        Returns:
            Merged tensor [batch_size, new_seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Create a mask to identify tokens to keep
        keep_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=hidden_states.device)
        keep_mask.scatter_(1, merge_indices, False)
        
        # Keep tokens that are not marked for merging
        kept_tokens = hidden_states[keep_mask].view(batch_size, -1, hidden_size)
        
        # For merged tokens, create averaged representations
        merged_tokens = []
        for batch_idx in range(batch_size):
            merge_idx_batch = merge_indices[batch_idx]
            merge_token_batch = hidden_states[batch_idx, merge_idx_batch, :]  # [num_to_merge, hidden_size]
            merged_repr = torch.mean(merge_token_batch, dim=0, keepdim=True)  # [1, hidden_size]
            merged_tokens.append(merged_repr)
        
        merged_tokens_tensor = torch.stack(merged_tokens, dim=0)  # [batch_size, 1, hidden_size]
        
        # Combine kept tokens with merged representations
        combined = torch.cat([kept_tokens, merged_tokens_tensor], dim=1)
        
        return combined