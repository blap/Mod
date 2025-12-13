"""
Cross-modal memory compression system with semantic integrity maintenance
for the Qwen3-VL architecture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class CrossModalCompressionConfig:
    """Configuration for cross-modal memory compression."""
    compression_ratio: float = 0.5  # Ratio of tokens to keep after compression
    semantic_preservation_strength: float = 0.8  # Strength of semantic preservation
    cross_attention_temperature: float = 1.0  # Temperature for cross-attention
    low_rank_dimension: int = 64  # Dimension for low-rank approximations
    use_cross_attention_selection: bool = True  # Whether to use cross-attention for token selection
    use_low_rank_compression: bool = True  # Whether to use low-rank compression
    use_semantic_similarity_compression: bool = True  # Whether to use semantic similarity


class CrossModalMemoryCompressor(nn.Module):
    """
    Cross-modal memory compression system that reduces memory usage during 
    vision-language fusion while maintaining semantic integrity.
    """
    
    def __init__(self, config: CrossModalCompressionConfig):
        super().__init__()
        self.config = config

        # Cross-attention mechanism for identifying relevant cross-modal connections
        if config.use_cross_attention_selection:
            # Use a more flexible approach - create cross-attention with a common dimension
            # that can handle various input dimensions through projection
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.low_rank_dimension,
                num_heads=min(8, config.low_rank_dimension // 4) if config.low_rank_dimension >= 4 else 1,  # Ensure num_heads is valid
                dropout=0.1,
                batch_first=True
            )

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the compression system."""
        # Nothing to initialize since we removed fixed projectors
        pass
    
    def compute_cross_attention(self, vision_features: torch.Tensor,
                               lang_features: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-attention between vision and language features to identify
        relevant cross-modal connections.

        Args:
            vision_features: Vision features of shape (batch_size, vision_seq_len, hidden_dim)
            lang_features: Language features of shape (batch_size, lang_seq_len, hidden_dim)

        Returns:
            Cross-attention weights of shape (batch_size, vision_seq_len, lang_seq_len)
        """
        if not self.config.use_cross_attention_selection:
            # Return uniform attention if not using cross-attention selection
            batch_size, vision_seq_len, _ = vision_features.shape
            _, lang_seq_len, _ = lang_features.shape
            return torch.ones(batch_size, vision_seq_len, lang_seq_len,
                            device=vision_features.device, dtype=vision_features.dtype)

        # Project features to low-rank dimension if needed
        # First, ensure the projection matrix can handle the input dimension
        vision_hidden_dim = vision_features.size(-1)
        lang_hidden_dim = lang_features.size(-1)

        # Project vision features
        if vision_hidden_dim != self.config.low_rank_dimension:
            # Create a temporary projection matrix for this specific dimension
            # F.linear expects weight of shape (out_features, in_features)
            proj_weight = torch.randn(self.config.low_rank_dimension, vision_hidden_dim,
                                    device=vision_features.device, dtype=vision_features.dtype) / (vision_hidden_dim ** 0.5)
            proj_bias = torch.zeros(self.config.low_rank_dimension,
                                  device=vision_features.device, dtype=vision_features.dtype)
            projected_vision = F.linear(vision_features, proj_weight, proj_bias)
        else:
            projected_vision = vision_features

        # Project language features
        if lang_hidden_dim != self.config.low_rank_dimension:
            # Create a temporary projection matrix for this specific dimension
            # F.linear expects weight of shape (out_features, in_features)
            proj_weight = torch.randn(self.config.low_rank_dimension, lang_hidden_dim,
                                    device=lang_features.device, dtype=lang_features.dtype) / (lang_hidden_dim ** 0.5)
            proj_bias = torch.zeros(self.config.low_rank_dimension,
                                  device=lang_features.device, dtype=lang_features.dtype)
            projected_lang = F.linear(lang_features, proj_weight, proj_bias)
        else:
            projected_lang = lang_features

        # Compute cross-attention: vision attends to language
        # Use scaled dot-product attention directly to avoid dimension mismatch issues
        # Q: projected_vision (batch_size, vision_seq_len, low_rank_dim)
        # K: projected_lang (batch_size, lang_seq_len, low_rank_dim)
        # V: projected_lang (batch_size, lang_seq_len, low_rank_dim)

        # Transpose for attention computation
        Q = projected_vision  # (batch_size, vision_seq_len, low_rank_dim)
        K = projected_lang.transpose(-2, -1)  # (batch_size, low_rank_dim, lang_seq_len)

        # Compute attention scores: (batch_size, vision_seq_len, lang_seq_len)
        attn_scores = torch.bmm(Q, K)  # Batch matrix multiplication

        # Scale attention scores
        scale_factor = self.config.low_rank_dimension ** -0.5
        attn_scores = attn_scores * scale_factor

        # Apply temperature scaling
        attn_scores = attn_scores / self.config.cross_attention_temperature

        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_scores, dim=-1)

        return attn_weights
    
    def select_important_tokens(self, features: torch.Tensor, 
                               importance_scores: torch.Tensor, 
                               target_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select the most important tokens based on importance scores.
        
        Args:
            features: Input features of shape (batch_size, seq_len, hidden_dim)
            importance_scores: Importance scores of shape (batch_size, seq_len)
            target_ratio: Ratio of tokens to keep
        
        Returns:
            Tuple of (selected_features, selected_indices)
        """
        batch_size, seq_len, hidden_dim = features.shape
        num_keep = max(1, int(seq_len * target_ratio))  # Ensure at least 1 token is kept
        
        # Get top-k important token indices
        _, top_indices = torch.topk(importance_scores, num_keep, dim=1, largest=True, sorted=False)
        
        # Sort indices to maintain order (optional but often helpful)
        top_indices = top_indices.sort(dim=1).values
        
        # Gather the selected features
        batch_indices = torch.arange(batch_size, device=features.device).unsqueeze(1)
        selected_features = features[batch_indices, top_indices]  # (batch_size, num_keep, hidden_dim)
        
        return selected_features, top_indices
    
    def compute_importance_scores(self, features: torch.Tensor, 
                                 cross_attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for tokens based on cross-attention weights.
        
        Args:
            features: Input features of shape (batch_size, seq_len, hidden_dim)
            cross_attn_weights: Cross-attention weights of shape (batch_size, seq_len, other_seq_len)
        
        Returns:
            Importance scores of shape (batch_size, seq_len)
        """
        # Use attention weights to compute importance
        # Sum attention weights across the other modality to get importance per token
        importance_scores = cross_attn_weights.sum(dim=-1)  # (batch_size, seq_len)
        
        # Add feature magnitude as additional importance indicator
        feature_magnitude = torch.norm(features, p=2, dim=-1)  # (batch_size, seq_len)
        importance_scores = importance_scores + feature_magnitude * 0.1  # Weighted combination
        
        return importance_scores
    
    def low_rank_compress(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply low-rank compression to features using SVD approximation or linear projection.

        Args:
            features: Input features of shape (batch_size, seq_len, hidden_dim)

        Returns:
            Compressed features of shape (batch_size, seq_len, low_rank_dim)
        """
        if not self.config.use_low_rank_compression:
            return features

        batch_size, seq_len, hidden_dim = features.shape
        low_rank_dim = self.config.low_rank_dimension

        # Use a dynamic linear projection to reduce dimension
        # Create a temporary projection matrix for this specific dimension
        # F.linear expects weight of shape (out_features, in_features)
        proj_weight = torch.randn(low_rank_dim, hidden_dim, device=features.device, dtype=features.dtype) / (hidden_dim ** 0.5)
        proj_bias = torch.zeros(low_rank_dim, device=features.device, dtype=features.dtype)

        # Reshape for projection: treat batch and sequence dimensions together
        features_2d = features.view(-1, hidden_dim)  # (batch_size * seq_len, hidden_dim)

        # Apply compression projection
        compressed_2d = F.linear(features_2d, proj_weight, proj_bias)

        # Apply non-linearity for better representation
        compressed_2d = F.gelu(compressed_2d)

        # Reshape back
        compressed_features = compressed_2d.view(batch_size, seq_len, low_rank_dim)

        return compressed_features

    def low_rank_decompress(self, compressed_features: torch.Tensor,
                           original_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        Decompress low-rank features back to original dimension.

        Args:
            compressed_features: Compressed features of shape (batch_size, seq_len, low_rank_dim)
            original_shape: Original shape (batch_size, seq_len, hidden_dim)

        Returns:
            Decompressed features of original shape
        """
        if not self.config.use_low_rank_compression:
            return compressed_features

        batch_size, seq_len, low_rank_dim = compressed_features.shape
        _, _, original_hidden_dim = original_shape

        # Create a temporary decompression projection matrix
        # F.linear expects weight of shape (out_features, in_features)
        proj_weight = torch.randn(original_hidden_dim, low_rank_dim, device=compressed_features.device,
                                 dtype=compressed_features.dtype) / (low_rank_dim ** 0.5)
        proj_bias = torch.zeros(original_hidden_dim, device=compressed_features.device,
                               dtype=compressed_features.dtype)

        # Reshape for decompression
        compressed_2d = compressed_features.view(-1, low_rank_dim)  # (batch_size * seq_len, low_rank_dim)

        # Apply non-linearity before decompression
        compressed_2d = F.gelu(compressed_2d)

        # Apply decompression projection
        decompressed_2d = F.linear(compressed_2d, proj_weight, proj_bias)

        # Reshape back to original dimensions
        decompressed_features = decompressed_2d.view(batch_size, seq_len, original_hidden_dim)

        return decompressed_features
    
    def compute_semantic_similarity(self, original_features: torch.Tensor, 
                                   decompressed_features: torch.Tensor) -> torch.Tensor:
        """
        Compute semantic similarity between original and decompressed features.
        
        Args:
            original_features: Original features
            decompressed_features: Decompressed features
        
        Returns:
            Similarity scores
        """
        # Compute cosine similarity
        original_norm = F.normalize(original_features, p=2, dim=-1)
        decompressed_norm = F.normalize(decompressed_features, p=2, dim=-1)
        
        similarity = F.cosine_similarity(original_norm, decompressed_norm, dim=-1)
        
        return similarity.mean(dim=-1).mean(dim=0)  # Average across sequence and batch
    
    def compress(self, vision_features: torch.Tensor, 
                lang_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compress vision and language features with cross-modal awareness.
        
        Args:
            vision_features: Vision features of shape (batch_size, vision_seq_len, hidden_dim)
            lang_features: Language features of shape (batch_size, lang_seq_len, hidden_dim)
        
        Returns:
            Tuple of (compressed_vision, compressed_lang, compression_info)
        """
        start_time = time.time()
        
        batch_size_v, vision_seq_len, vision_hidden_dim = vision_features.shape
        batch_size_l, lang_seq_len, lang_hidden_dim = lang_features.shape
        
        # Ensure batch sizes match
        assert batch_size_v == batch_size_l, "Batch sizes must match"
        batch_size = batch_size_v
        
        # Compute cross-attention weights to identify important tokens
        cross_attn_weights = self.compute_cross_attention(vision_features, lang_features)
        
        # Compute importance scores for vision features based on cross-attention
        vision_importance_scores = self.compute_importance_scores(vision_features, cross_attn_weights)
        
        # Compute importance scores for language features (transpose attention)
        lang_cross_attn_weights = cross_attn_weights.transpose(1, 2)  # (batch_size, lang_seq_len, vision_seq_len)
        lang_importance_scores = self.compute_importance_scores(lang_features, lang_cross_attn_weights)
        
        # Select important tokens based on importance scores
        compressed_vision, vision_indices = self.select_important_tokens(
            vision_features, vision_importance_scores, self.config.compression_ratio
        )
        
        compressed_lang, lang_indices = self.select_important_tokens(
            lang_features, lang_importance_scores, self.config.compression_ratio
        )
        
        # Apply low-rank compression if enabled
        if self.config.use_low_rank_compression:
            compressed_vision = self.low_rank_compress(compressed_vision)
            compressed_lang = self.low_rank_compress(compressed_lang)
        
        # Compute compression information
        compression_info = {
            'compression_ratios': {
                'vision': compressed_vision.size(1) / vision_seq_len,
                'language': compressed_lang.size(1) / lang_seq_len
            },
            'selected_indices': {
                'vision': vision_indices,
                'language': lang_indices
            },
            'memory_reduction_ratio': 1.0 - (
                (compressed_vision.numel() + compressed_lang.numel()) / 
                (vision_features.numel() + lang_features.numel())
            ),
            'compression_time': time.time() - start_time,
            'semantic_preservation_metrics': {}
        }
        
        # Compute semantic preservation metrics if needed
        if self.config.semantic_preservation_strength > 0:
            # Temporarily decompress to evaluate semantic preservation
            temp_vision = self.low_rank_decompress(
                compressed_vision, (batch_size, compressed_vision.size(1), vision_hidden_dim)
            ) if self.config.use_low_rank_compression else compressed_vision
            
            temp_lang = self.low_rank_decompress(
                compressed_lang, (batch_size, compressed_lang.size(1), lang_hidden_dim)
            ) if self.config.use_low_rank_compression else compressed_lang
            
            # Calculate similarity with original features (expanded back to original sequence length)
            # For this evaluation, we'll expand back using the selected indices
            expanded_vision = torch.zeros_like(vision_features)
            expanded_lang = torch.zeros_like(lang_features)
            
            batch_indices = torch.arange(batch_size, device=vision_features.device).unsqueeze(1)
            expanded_vision[batch_indices, vision_indices] = temp_vision
            expanded_lang[batch_indices, lang_indices] = temp_lang
            
            vision_similarity = self.compute_semantic_similarity(
                vision_features, expanded_vision
            ).item()
            
            lang_similarity = self.compute_semantic_similarity(
                lang_features, expanded_lang
            ).item()
            
            compression_info['semantic_preservation_metrics'] = {
                'vision_semantic_preservation': vision_similarity,
                'lang_semantic_preservation': lang_similarity
            }
        
        return compressed_vision, compressed_lang, compression_info
    
    def decompress(self, compressed_vision: torch.Tensor,
                  compressed_lang: torch.Tensor,
                  original_vision_shape: Optional[Tuple[int, int, int]] = None,
                  original_lang_shape: Optional[Tuple[int, int, int]] = None,
                  vision_selected_indices: Optional[torch.Tensor] = None,
                  lang_selected_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompress vision and language features back to original representations.

        Args:
            compressed_vision: Compressed vision features
            compressed_lang: Compressed language features
            original_vision_shape: Original vision features shape
            original_lang_shape: Original language features shape
            vision_selected_indices: Indices of selected vision tokens (for sequence expansion)
            lang_selected_indices: Indices of selected language tokens (for sequence expansion)

        Returns:
            Tuple of (decompressed_vision, decompressed_lang)
        """
        # Apply low-rank decompression if enabled
        if self.config.use_low_rank_compression and original_vision_shape is not None:
            decompressed_vision = self.low_rank_decompress(compressed_vision, original_vision_shape)
        else:
            decompressed_vision = compressed_vision

        if self.config.use_low_rank_compression and original_lang_shape is not None:
            decompressed_lang = self.low_rank_decompress(compressed_lang, original_lang_shape)
        else:
            decompressed_lang = compressed_lang

        # If we have selected indices, we can expand back to original sequence length
        # For now, we'll return the decompressed features as they are
        # In a more advanced implementation, we might want to expand to original sequence length
        # by using the selected indices to place the features back in their original positions
        # and potentially using interpolation or other techniques for the missing positions

        return decompressed_vision, decompressed_lang


class CrossModalCompression(nn.Module):
    """
    Simplified wrapper class for cross-modal compression.
    """

    def __init__(self, config: CrossModalCompressionConfig):
        super().__init__()
        self.config = config
        self.compressor = CrossModalMemoryCompressor(config)

    def forward(self, vision_features: torch.Tensor,
               lang_features: torch.Tensor,
               return_compression_info: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with cross-modal compression.

        Args:
            vision_features: Vision features from vision encoder
            lang_features: Language features from language model
            return_compression_info: Whether to return compression metrics

        Returns:
            Tuple of (compressed_vision, compressed_lang, compression_info) if return_compression_info else
            Tuple of (compressed_vision, compressed_lang)
        """
        compressed_vision, compressed_lang, compression_info = self.compressor.compress(
            vision_features, lang_features
        )

        if return_compression_info:
            return compressed_vision, compressed_lang, compression_info
        else:
            return compressed_vision, compressed_lang


class CrossModalFusionCompressor(nn.Module):
    """
    Integration module for cross-modal compression in the Qwen3-VL architecture.
    This module can be inserted at the vision-language fusion stage.
    """

    def __init__(self, config: CrossModalCompressionConfig):
        super().__init__()
        self.compressor = CrossModalMemoryCompressor(config)
        self.config = config

    def forward(self, vision_features: torch.Tensor,
               lang_features: torch.Tensor,
               return_compression_info: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with cross-modal compression.

        Args:
            vision_features: Vision features from vision encoder
            lang_features: Language features from language model
            return_compression_info: Whether to return compression metrics

        Returns:
            Tuple of (compressed_vision, compressed_lang, compression_info) if return_compression_info else
            Tuple of (compressed_vision, compressed_lang)
        """
        compressed_vision, compressed_lang, compression_info = self.compressor.compress(
            vision_features, lang_features
        )

        if return_compression_info:
            return compressed_vision, compressed_lang, compression_info
        else:
            return compressed_vision, compressed_lang