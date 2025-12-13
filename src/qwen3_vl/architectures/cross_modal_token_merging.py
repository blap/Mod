"""
Cross-Modal Token Merging (CMTM) for Qwen3-VL model.
Implements token similarity computation and merging algorithms for vision-text fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class CrossModalTokenMerger(nn.Module):
    """
    Cross-Modal Token Merging module for reducing computation overhead
    by merging similar tokens across vision and language modalities.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Hidden dimensions
        self.language_hidden_size = config.hidden_size
        self.vision_hidden_size = config.vision_hidden_size
        
        # Create projection layers to align dimensions if needed
        self.vision_to_lang_proj = nn.Linear(self.vision_hidden_size, self.language_hidden_size)
        self.lang_to_vision_proj = nn.Linear(self.language_hidden_size, self.vision_hidden_size)
        
        # Token similarity computation
        self.similarity_temperature = getattr(config, 'cmtm_similarity_temperature', 0.1)
        self.merging_threshold = getattr(config, 'cmtm_merging_threshold', 0.7)
        self.max_merged_tokens = getattr(config, 'cmtm_max_merged_tokens', 0.5)  # Fraction of tokens to merge
        
        # Merging weight computation
        self.merging_network = nn.Sequential(
            nn.Linear(self.language_hidden_size * 2, self.language_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.language_hidden_size // 2, 2),  # Two weights for merging
            nn.Softmax(dim=-1)
        )
        
        # Cross-attention for token similarity
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.language_hidden_size,
            num_heads=min(8, self.language_hidden_size // 64),
            batch_first=True
        )

    def forward(
        self,
        language_tokens: torch.Tensor,
        vision_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Merge similar tokens across language and vision modalities.
        
        Args:
            language_tokens: [batch_size, lang_seq_len, language_hidden_size]
            vision_tokens: [batch_size, vision_seq_len, vision_hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (merged_language_tokens, merged_vision_tokens, merge_info)
        """
        batch_size, lang_seq_len, lang_dim = language_tokens.shape
        _, vision_seq_len, vision_dim = vision_tokens.shape
        
        # Project vision tokens to language dimension space
        vision_projected = self.vision_to_lang_proj(vision_tokens)  # [B, vision_seq_len, lang_dim]
        
        # Compute similarity between language and vision tokens
        similarity_matrix = self._compute_cross_modal_similarity(
            language_tokens, vision_projected
        )
        
        # Determine which tokens to merge based on similarity
        merge_indices = self._determine_merge_indices(similarity_matrix)
        
        # Perform token merging
        merged_lang_tokens, merged_vision_tokens = self._perform_token_merging(
            language_tokens, vision_projected, vision_tokens, merge_indices
        )
        
        # Create merge information for debugging/monitoring
        merge_info = {
            'similarity_matrix': similarity_matrix,
            'merge_indices': merge_indices,
            'merged_lang_count': merged_lang_tokens.shape[1],
            'merged_vision_count': merged_vision_tokens.shape[1]
        }
        
        return merged_lang_tokens, merged_vision_tokens, merge_info

    def _compute_cross_modal_similarity(self, lang_tokens, vision_tokens):
        """
        Compute similarity between language and vision tokens.
        """
        batch_size, lang_seq_len, lang_dim = lang_tokens.shape
        _, vision_seq_len, _ = vision_tokens.shape
        
        # Expand tokens for pairwise similarity computation
        lang_expanded = lang_tokens.unsqueeze(2).expand(-1, -1, vision_seq_len, -1)  # [B, L, V, D]
        vision_expanded = vision_tokens.unsqueeze(1).expand(-1, lang_seq_len, -1, -1)  # [B, L, V, D]
        
        # Compute cosine similarity
        lang_norm = F.normalize(lang_expanded, p=2, dim=-1)
        vision_norm = F.normalize(vision_expanded, p=2, dim=-1)
        
        similarity = torch.sum(lang_norm * vision_norm, dim=-1)  # [B, L, V]
        
        # Apply temperature scaling
        similarity = similarity / self.similarity_temperature
        
        return similarity

    def _determine_merge_indices(self, similarity_matrix):
        """
        Determine which token pairs to merge based on similarity scores.
        """
        batch_size, lang_seq_len, vision_seq_len = similarity_matrix.shape
        
        # Apply softmax to get attention-like weights
        attention_weights = F.softmax(similarity_matrix, dim=-1)  # [B, L, V]
        
        # Find top-k most similar pairs
        max_merge_count = int(min(lang_seq_len, vision_seq_len) * self.max_merged_tokens)
        
        # Flatten similarity matrix to find top pairs
        flat_similarity = similarity_matrix.view(batch_size, -1)  # [B, L*V]
        
        # Get top similar pairs
        top_values, top_indices = torch.topk(
            flat_similarity, 
            k=min(max_merge_count, flat_similarity.shape[-1]), 
            dim=-1
        )
        
        # Convert flat indices back to (lang_idx, vision_idx)
        lang_indices = top_indices // vision_seq_len
        vision_indices = top_indices % vision_seq_len
        
        # Only merge if similarity is above threshold
        valid_mask = top_values > self.merging_threshold
        
        return {
            'lang_indices': lang_indices,
            'vision_indices': vision_indices,
            'valid_mask': valid_mask,
            'similarity_scores': top_values
        }

    def _perform_token_merging(self, lang_tokens, vision_projected, original_vision_tokens, merge_indices):
        """
        Perform the actual token merging operation.
        """
        batch_size, lang_seq_len, lang_dim = lang_tokens.shape
        _, vision_seq_len, vision_dim = original_vision_tokens.shape
        
        # Get valid merge indices
        lang_idx = merge_indices['lang_indices']
        vision_idx = merge_indices['vision_indices']
        valid_mask = merge_indices['valid_mask']
        
        merged_lang_tokens = []
        merged_vision_tokens = []
        
        for b in range(batch_size):
            # Get valid indices for this batch
            valid_lang = lang_idx[b][valid_mask[b]]
            valid_vision = vision_idx[b][valid_mask[b]]
            
            # Separate tokens that will be merged vs kept
            merged_lang_portion = lang_tokens[b][valid_lang]  # Language tokens to merge
            kept_lang_mask = torch.ones(lang_seq_len, dtype=torch.bool, device=lang_tokens.device)
            kept_lang_mask[valid_lang] = False
            kept_lang_tokens = lang_tokens[b][kept_lang_mask]
            
            # Vision tokens to merge
            merged_vision_portion = original_vision_tokens[b][valid_vision]
            kept_vision_mask = torch.ones(vision_seq_len, dtype=torch.bool, device=original_vision_tokens.device)
            kept_vision_mask[valid_vision] = False
            kept_vision_tokens = original_vision_tokens[b][kept_vision_mask]
            
            # Merge similar tokens
            if len(valid_lang) > 0:
                # Project vision tokens to language space for merging
                vision_projected_batch = vision_projected[b][valid_vision]
                
                # Compute merging weights
                merged_portion = []
                for i in range(len(valid_lang)):
                    lang_token = merged_lang_portion[i]
                    vision_token = vision_projected_batch[i]
                    
                    # Concatenate language and vision tokens
                    concat_features = torch.cat([lang_token, vision_token], dim=-1).unsqueeze(0)  # [1, 2*D]
                    
                    # Compute merging weights
                    weights = self.merging_network(concat_features).squeeze(0)  # [2]
                    
                    # Merge tokens using computed weights
                    merged_token = weights[0] * lang_token + weights[1] * vision_token
                    merged_portion.append(merged_token)
                
                merged_portion = torch.stack(merged_portion, dim=0)
            else:
                merged_portion = torch.empty(0, lang_dim, device=lang_tokens.device)
            
            # Combine merged and kept tokens
            if merged_portion.shape[0] > 0:
                final_lang_tokens = torch.cat([kept_lang_tokens, merged_portion], dim=0)
            else:
                final_lang_tokens = kept_lang_tokens
                
            if merged_vision_portion.shape[0] > 0:
                final_vision_tokens = torch.cat([kept_vision_tokens, merged_vision_portion], dim=0)
            else:
                final_vision_tokens = kept_vision_tokens
                
            merged_lang_tokens.append(final_lang_tokens)
            merged_vision_tokens.append(final_vision_tokens)
        
        # Pad sequences to same length for batch processing
        max_lang_len = max([t.shape[0] for t in merged_lang_tokens])
        max_vision_len = max([t.shape[0] for t in merged_vision_tokens])
        
        # Pad language tokens
        padded_lang_tokens = []
        for tokens in merged_lang_tokens:
            if tokens.shape[0] < max_lang_len:
                padding = torch.zeros(max_lang_len - tokens.shape[0], lang_dim, device=tokens.device)
                tokens = torch.cat([tokens, padding], dim=0)
            padded_lang_tokens.append(tokens)
        
        # Pad vision tokens
        padded_vision_tokens = []
        for tokens in merged_vision_tokens:
            if tokens.shape[0] < max_vision_len:
                padding = torch.zeros(max_vision_len - tokens.shape[0], vision_dim, device=tokens.device)
                tokens = torch.cat([tokens, padding], dim=0)
            padded_vision_tokens.append(tokens)
        
        merged_lang_tokens = torch.stack(padded_lang_tokens, dim=0)
        merged_vision_tokens = torch.stack(padded_vision_tokens, dim=0)
        
        return merged_lang_tokens, merged_vision_tokens


class HierarchicalCrossModalFusion(nn.Module):
    """
    Hierarchical fusion mechanism that operates at multiple levels of granularity
    using the cross-modal token merging approach.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Multi-level token merger
        self.token_merger = CrossModalTokenMerger(config)
        
        # Hierarchical fusion layers
        self.hierarchy_levels = getattr(config, 'cmtm_hierarchy_levels', 3)
        self.fusion_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size) 
            for _ in range(self.hierarchy_levels)
        ])
        
        # Attention mechanism for hierarchical fusion
        self.hierarchical_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=min(8, config.hidden_size // 64),
            batch_first=True
        )
        
    def forward(
        self,
        language_tokens: torch.Tensor,
        vision_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Perform hierarchical cross-modal fusion with token merging.
        """
        batch_size, lang_seq_len, lang_dim = language_tokens.shape
        _, vision_seq_len, vision_dim = vision_tokens.shape
        
        # Apply token merging at different hierarchy levels
        merged_lang_tokens = language_tokens
        merged_vision_tokens = vision_tokens
        
        merge_info_history = []
        
        for level in range(self.hierarchy_levels):
            # Apply token merging at this level
            merged_lang_tokens, merged_vision_tokens, merge_info = self.token_merger(
                merged_lang_tokens, merged_vision_tokens, attention_mask
            )
            
            # Apply fusion layer
            merged_lang_tokens = self.fusion_layers[level](merged_lang_tokens)
            merged_vision_tokens = self.fusion_layers[level](merged_vision_tokens)
            
            # Apply cross-modal attention
            merged_lang_tokens, _ = self.hierarchical_attention(
                query=merged_lang_tokens,
                key=merged_vision_tokens,
                value=merged_vision_tokens
            )
            
            merge_info_history.append(merge_info)
        
        # Final fusion information
        fusion_info = {
            'initial_lang_tokens': lang_seq_len,
            'initial_vision_tokens': vision_seq_len,
            'final_lang_tokens': merged_lang_tokens.shape[1],
            'final_vision_tokens': merged_vision_tokens.shape[1],
            'merge_info_history': merge_info_history
        }
        
        return merged_lang_tokens, merged_vision_tokens, fusion_info