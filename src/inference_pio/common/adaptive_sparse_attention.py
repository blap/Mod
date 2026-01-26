"""
Adaptive Sparse Attention Implementation for Inference-PIO System

This module provides an advanced adaptive sparse attention implementation that dynamically adjusts
the attention pattern based on input characteristics. The implementation includes mechanisms to
adaptively determine sparsity patterns, reduce computational complexity while maintaining focus
on relevant tokens, and allow different patterns based on input type.

The system adapts to different input types by analyzing input characteristics and selecting
optimal attention patterns accordingly.
"""

import math
from typing import Optional, Tuple, Union, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveSparseAttention(nn.Module):
    """
    Advanced adaptive sparse attention implementation for the Inference-PIO system.

    This implementation dynamically adapts the attention pattern based on input characteristics
    to reduce computational complexity while maintaining focus on relevant tokens. It supports
    various adaptive sparse attention patterns and can switch between them based on input type.
    """

    def __init__(
        self,
        config: Any,  # Generic config that has attributes like hidden_size, num_attention_heads, etc.
        layer_idx: Optional[int] = None,
        adaptive_strategy: str = "input_dependent",  # Options: "input_dependent", "dynamic", "static"
        initial_sparse_pattern: str = "longformer",
        sparsity_ratio: float = 0.25,
        block_size: int = 64,
        local_window_size: int = 128,
        use_global_attention: bool = True,
        global_attention_indices: Optional[list] = None,
        attention_dropout: float = 0.0,
        bias: bool = True,
        temperature: float = 1.0,
        adaptation_threshold: float = 0.1,
    ):
        """
        Initialize the adaptive sparse attention module.

        Args:
            config: Model configuration object with attributes like hidden_size, num_attention_heads, etc.
            layer_idx: Index of the transformer layer (optional)
            adaptive_strategy: Strategy for adapting attention patterns ('input_dependent', 'dynamic', 'static')
            initial_sparse_pattern: Initial sparse attention pattern to use
            sparsity_ratio: Base ratio of tokens to attend to
            block_size: Block size for block sparse attention
            local_window_size: Window size for local attention
            use_global_attention: Whether to use global attention in sparse patterns
            global_attention_indices: Indices for global attention tokens
            attention_dropout: Dropout rate for attention
            bias: Whether to use bias in linear projections
            temperature: Temperature for attention scaling
            adaptation_threshold: Threshold for triggering adaptation
        """
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.adaptive_strategy = adaptive_strategy
        self.initial_sparse_pattern = initial_sparse_pattern
        self.sparsity_ratio = sparsity_ratio
        self.block_size = block_size
        self.local_window_size = local_window_size
        self.use_global_attention = use_global_attention
        self.global_attention_indices = global_attention_indices or [0]  # Default to first token as global
        self.temperature = temperature
        self.adaptation_threshold = adaptation_threshold

        # Extract attention parameters from config
        self.hidden_size = getattr(config, 'hidden_size', 512)
        self.num_attention_heads = getattr(config, 'num_attention_heads', 8)
        self.num_key_value_heads = getattr(
            config, 'num_key_value_heads', self.num_attention_heads
        )
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = getattr(config, 'max_position_embeddings', 2048)
        self.rope_theta = getattr(config, 'rope_theta', 10000.0)

        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )

        # Initialize projections
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=bias
        )

        # Store attention dropout
        self.attention_dropout = attention_dropout

        # Initialize adaptation components
        self._init_adaptation_components()

    def _init_adaptation_components(self):
        """
        Initialize components for adaptive attention pattern selection.
        """
        # Learnable parameters for adaptation
        self.pattern_selector = nn.Linear(self.hidden_size, 6)  # 6 different patterns
        self.sparsity_predictor = nn.Linear(self.hidden_size, 1)
        
        # Activation function for pattern selection
        self.pattern_activation = nn.Softmax(dim=-1)
        self.sparsity_activation = nn.Sigmoid()

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Repeat key and value tensors for grouped-query attention.
        """
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def _analyze_input_characteristics(self, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze input characteristics to determine optimal attention pattern.
        
        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Dictionary containing input characteristics
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Calculate various statistics about the input
        variance = torch.var(hidden_states, dim=-1).mean(dim=0)  # Variance per position
        entropy = -torch.sum(F.softmax(hidden_states, dim=-1) * F.log_softmax(hidden_states, dim=-1), dim=-1).mean(dim=0)  # Entropy per position
        magnitude = torch.norm(hidden_states, p=2, dim=-1).mean(dim=0)  # L2 norm per position
        
        # Detect sequence patterns
        position_variance = torch.var(hidden_states, dim=0).sum(dim=-1)  # How much each position varies across batch
        activation_density = (hidden_states.abs() > 0.1).float().mean()  # Density of activations
        
        return {
            'variance': variance.mean().item(),
            'entropy': entropy.mean().item(),
            'magnitude': magnitude.mean().item(),
            'position_variance': position_variance.mean().item(),
            'activation_density': activation_density.item(),
            'seq_len': seq_len,
            'batch_size': batch_size
        }

    def _select_adaptive_pattern(self, input_features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Select the optimal attention pattern based on input characteristics.
        
        Args:
            input_features: Dictionary containing input characteristics
            
        Returns:
            Tuple of (selected_pattern, adjusted_sparsity_ratio)
        """
        # Based on input characteristics, select the most appropriate pattern
        seq_len = input_features['seq_len']
        activation_density = input_features['activation_density']
        variance = input_features['variance']
        
        # Heuristic rules for pattern selection
        if seq_len > 1024:
            # For long sequences, prefer patterns that scale well
            if activation_density > 0.5:
                # Dense activation - use local attention
                selected_pattern = "local"
                adjusted_sparsity = min(0.3, self.sparsity_ratio)
            else:
                # Sparse activation - use random or strided attention
                selected_pattern = "strided"
                adjusted_sparsity = max(0.15, self.sparsity_ratio)
        elif seq_len > 256:
            # Medium sequences - balance between local and global
            if variance > 0.5:
                # High variance suggests important tokens scattered
                selected_pattern = "bigbird"
                adjusted_sparsity = self.sparsity_ratio
            else:
                # Low variance suggests local coherence
                selected_pattern = "local"
                adjusted_sparsity = min(0.4, self.sparsity_ratio)
        else:
            # Short sequences - can afford more connections
            selected_pattern = "longformer"
            adjusted_sparsity = min(0.5, self.sparsity_ratio)
        
        return selected_pattern, adjusted_sparsity

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass using adaptive sparse attention patterns.

        Args:
            hidden_states: Input hidden states of shape (batch_size, seq_len, hidden_size)
            attention_mask: Attention mask of shape (batch_size, 1, seq_len, seq_len)
            position_ids: Position IDs for rotary embeddings
            past_key_value: Past key-value states for caching
            output_attentions: Whether to output attention weights
            use_cache: Whether to use KV cache
            cache_position: Position tensor for caching

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.size()

        # Apply projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # (bsz, num_key_value_heads, q_len, head_dim)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)  # (bsz, num_key_value_heads, q_len, head_dim)

        # Repeat key and value states for grouped query attention
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Apply rotary embeddings if position_ids are provided
        if position_ids is not None:
            try:
                from .tensor_utils import apply_rotary_pos_emb, rotate_half
                from .rotary_embeddings import RotaryEmbedding
                # Initialize rotary embeddings if not already done
                if not hasattr(self, 'rotary_emb'):
                    self.rotary_emb = RotaryEmbedding(
                        dim=self.head_dim,
                        max_position_embeddings=self.max_position_embeddings,
                        base=self.rope_theta,
                    )
                cos, sin = self.rotary_emb(value_states, seq_len=q_len)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )
            except ImportError:
                # If rotary embeddings are not available, skip them
                pass

        # Determine attention pattern based on adaptive strategy
        if self.adaptive_strategy == "input_dependent":
            # Analyze input characteristics to select pattern
            input_features = self._analyze_input_characteristics(hidden_states)
            selected_pattern, adjusted_sparsity = self._select_adaptive_pattern(input_features)
        elif self.adaptive_strategy == "dynamic":
            # Use learnable pattern selector based on hidden states
            pattern_logits = self.pattern_selector(hidden_states.mean(dim=1))  # Average over sequence
            pattern_probs = self.pattern_activation(pattern_logits)
            selected_pattern_idx = torch.argmax(pattern_probs, dim=-1).item()
            
            # Map index to pattern name
            patterns = ["longformer", "bigbird", "block_sparse", "local", "random", "strided"]
            selected_pattern = patterns[selected_pattern_idx]
            
            # Predict sparsity ratio
            sparsity_pred = self.sparsity_activation(self.sparsity_predictor(hidden_states.mean(dim=1)))
            adjusted_sparsity = sparsity_pred.item() * 0.5 + 0.1  # Scale to [0.1, 0.6]
        else:
            # Static pattern
            selected_pattern = self.initial_sparse_pattern
            adjusted_sparsity = self.sparsity_ratio

        # Apply adaptive sparse attention pattern
        if selected_pattern == "longformer":
            attn_weights = self._apply_longformer_attention(
                query_states, key_states, value_states, attention_mask, adjusted_sparsity
            )
        elif selected_pattern == "bigbird":
            attn_weights = self._apply_bigbird_attention(
                query_states, key_states, value_states, attention_mask, adjusted_sparsity
            )
        elif selected_pattern == "block_sparse":
            attn_weights = self._apply_block_sparse_attention(
                query_states, key_states, value_states, attention_mask, adjusted_sparsity
            )
        elif selected_pattern == "local":
            attn_weights = self._apply_local_attention(
                query_states, key_states, value_states, attention_mask, adjusted_sparsity
            )
        elif selected_pattern == "random":
            attn_weights = self._apply_random_sparse_attention(
                query_states, key_states, value_states, attention_mask, adjusted_sparsity
            )
        elif selected_pattern == "strided":
            attn_weights = self._apply_strided_sparse_attention(
                query_states, key_states, value_states, attention_mask, adjusted_sparsity
            )
        else:
            # Default to local attention if pattern is not recognized
            attn_weights = self._apply_local_attention(
                query_states, key_states, value_states, attention_mask, adjusted_sparsity
            )

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)  # (bsz, num_heads, q_len, head_dim)

        # Reshape for output
        attn_output = (
            attn_output.transpose(1, 2)  # (bsz, q_len, num_heads, head_dim)
            .contiguous()
            .view(bsz, q_len, self.num_attention_heads * self.head_dim)  # (bsz, q_len, hidden_size)
        )
        attn_output = self.o_proj(attn_output)

        # Handle KV cache for inference
        if use_cache:
            if past_key_value is not None:
                # Concatenate with past keys and values
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

            past_key_value = (key_states, value_states)

        return (
            attn_output,
            attn_weights if output_attentions else None,
            past_key_value,
        )

    def _apply_longformer_attention(self, query_states, key_states, value_states, attention_mask, sparsity_ratio):
        """
        Apply Longformer-style sparse attention with adaptive parameters.
        """
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Create sparse attention mask based on local window and global attention
        sparse_mask = self._create_longformer_sparse_mask(q_len, sparsity_ratio, device=query_states.device)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)  # (bsz, num_heads, q_len, q_len)

        # Apply sparse mask to attention weights
        attn_weights = attn_weights.masked_fill(sparse_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # Apply dropout if specified
        if self.attention_dropout > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        return attn_weights

    def _create_longformer_sparse_mask(self, seq_len, sparsity_ratio, device):
        """
        Create sparse attention mask for Longformer pattern with adaptive parameters.
        """
        # Adjust local window size based on sparsity ratio
        adaptive_window_size = int(self.local_window_size * sparsity_ratio * 2)
        adaptive_window_size = max(8, min(adaptive_window_size, seq_len))  # Keep reasonable bounds

        # Create a mask with local window attention
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Add local window connections
        for i in range(seq_len):
            start_idx = max(0, i - adaptive_window_size // 2)
            end_idx = min(seq_len, i + adaptive_window_size // 2 + 1)
            mask[i, start_idx:end_idx] = True

        # Add global attention connections if enabled
        if self.use_global_attention and self.global_attention_indices:
            for global_idx in self.global_attention_indices:
                if 0 <= global_idx < seq_len:
                    # Global token attends to all other tokens
                    mask[global_idx, :] = True
                    # All tokens attend to global token
                    mask[:, global_idx] = True

        # Expand mask to match attention weights shape
        # Shape: (1, 1, seq_len, seq_len)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask

    def _apply_bigbird_attention(self, query_states, key_states, value_states, attention_mask, sparsity_ratio):
        """
        Apply BigBird-style sparse attention with adaptive parameters.
        """
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Create sparse attention mask based on BigBird pattern
        sparse_mask = self._create_bigbird_sparse_mask(q_len, sparsity_ratio, device=query_states.device)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)  # (bsz, num_heads, q_len, q_len)

        # Apply sparse mask to attention weights
        attn_weights = attn_weights.masked_fill(sparse_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # Apply dropout if specified
        if self.attention_dropout > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        return attn_weights

    def _create_bigbird_sparse_mask(self, seq_len, sparsity_ratio, device):
        """
        Create sparse attention mask for BigBird pattern with adaptive parameters.
        """
        # Adjust local window size based on sparsity ratio
        adaptive_window_size = int(self.local_window_size * sparsity_ratio)
        adaptive_window_size = max(4, min(adaptive_window_size, seq_len // 2))

        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Add window attention (local connections)
        for i in range(seq_len):
            start_idx = max(0, i - adaptive_window_size // 2)
            end_idx = min(seq_len, i + adaptive_window_size // 2 + 1)
            mask[i, start_idx:end_idx] = True

        # Add global attention connections
        if self.use_global_attention and self.global_attention_indices:
            for global_idx in self.global_attention_indices:
                if 0 <= global_idx < seq_len:
                    # Global token attends to all other tokens
                    mask[global_idx, :] = True
                    # All tokens attend to global token
                    mask[:, global_idx] = True

        # Add random attention connections based on sparsity ratio
        num_random_connections = int(seq_len * sparsity_ratio)
        if num_random_connections > 0:
            random_indices = torch.randperm(seq_len, device=device)[:num_random_connections]
            for idx in random_indices:
                # Randomly connect to other positions
                other_indices = torch.randperm(seq_len, device=device)[:max(1, adaptive_window_size//2)]
                mask[idx, other_indices] = True
                mask[other_indices, idx] = True

        # Expand mask to match attention weights shape
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask

    def _apply_block_sparse_attention(self, query_states, key_states, value_states, attention_mask, sparsity_ratio):
        """
        Apply block sparse attention with adaptive parameters.
        """
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Calculate number of blocks
        num_blocks = (q_len + self.block_size - 1) // self.block_size

        # Create sparse attention mask based on block pattern
        sparse_mask = self._create_block_sparse_mask(q_len, num_blocks, sparsity_ratio, device=query_states.device)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)  # (bsz, num_heads, q_len, q_len)

        # Apply sparse mask to attention weights
        attn_weights = attn_weights.masked_fill(sparse_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # Apply dropout if specified
        if self.attention_dropout > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        return attn_weights

    def _create_block_sparse_mask(self, seq_len, num_blocks, sparsity_ratio, device):
        """
        Create sparse attention mask for block sparse pattern with adaptive parameters.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Create block structure
        for block_row in range(num_blocks):
            for block_col in range(num_blocks):
                row_start = block_row * self.block_size
                row_end = min((block_row + 1) * self.block_size, seq_len)
                col_start = block_col * self.block_size
                col_end = min((block_col + 1) * self.block_size, seq_len)

                # Determine if this block should be active based on sparsity ratio
                # Higher sparsity ratio means more blocks are activated
                if torch.rand(1, device=device).item() < sparsity_ratio or block_row == block_col:
                    # Activate this block (allows attention within block or diagonal blocks)
                    mask[row_start:row_end, col_start:col_end] = True

        # Expand mask to match attention weights shape
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask

    def _apply_local_attention(self, query_states, key_states, value_states, attention_mask, sparsity_ratio):
        """
        Apply local attention with adaptive sliding window.
        """
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Create local attention mask with adaptive window size
        local_mask = self._create_local_attention_mask(q_len, sparsity_ratio, device=query_states.device)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)  # (bsz, num_heads, q_len, q_len)

        # Apply local mask to attention weights
        attn_weights = attn_weights.masked_fill(local_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # Apply dropout if specified
        if self.attention_dropout > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        return attn_weights

    def _create_local_attention_mask(self, seq_len, sparsity_ratio, device):
        """
        Create local attention mask with adaptive sliding window.
        """
        # Adjust window size based on sparsity ratio
        adaptive_window_size = int(self.local_window_size * sparsity_ratio * 2)
        adaptive_window_size = max(4, min(adaptive_window_size, seq_len))

        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Add local window connections
        for i in range(seq_len):
            start_idx = max(0, i - adaptive_window_size // 2)
            end_idx = min(seq_len, i + adaptive_window_size // 2 + 1)
            mask[i, start_idx:end_idx] = True

        # Expand mask to match attention weights shape
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask

    def _apply_random_sparse_attention(self, query_states, key_states, value_states, attention_mask, sparsity_ratio):
        """
        Apply random sparse attention with adaptive sparsity.
        """
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Create random sparse mask based on adaptive sparsity ratio
        sparse_mask = self._create_random_sparse_mask(q_len, sparsity_ratio, device=query_states.device)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)  # (bsz, num_heads, q_len, q_len)

        # Apply sparse mask to attention weights
        attn_weights = attn_weights.masked_fill(sparse_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # Apply dropout if specified
        if self.attention_dropout > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        return attn_weights

    def _create_random_sparse_mask(self, seq_len, sparsity_ratio, device):
        """
        Create random sparse attention mask with adaptive sparsity.
        """
        # Create a random mask with the specified adaptive sparsity ratio
        mask = torch.rand(seq_len, seq_len, device=device) < sparsity_ratio
        # Ensure each position attends to itself
        mask.diagonal(dim1=0, dim2=1).fill_(True)

        # Expand mask to match attention weights shape
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask

    def _apply_strided_sparse_attention(self, query_states, key_states, value_states, attention_mask, sparsity_ratio):
        """
        Apply strided sparse attention with adaptive parameters.
        """
        bsz, num_heads, q_len, head_dim = query_states.size()

        # Create strided sparse mask with adaptive parameters
        sparse_mask = self._create_strided_sparse_mask(q_len, sparsity_ratio, device=query_states.device)

        # Compute attention scores
        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(head_dim)  # (bsz, num_heads, q_len, q_len)

        # Apply sparse mask to attention weights
        attn_weights = attn_weights.masked_fill(sparse_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply softmax
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )

        # Apply dropout if specified
        if self.attention_dropout > 0.0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        return attn_weights

    def _create_strided_sparse_mask(self, seq_len, sparsity_ratio, device):
        """
        Create strided sparse attention mask with adaptive parameters.
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        # Create strided pattern - each position attends to positions at regular intervals
        # Adjust stride based on sparsity ratio
        adaptive_stride = max(1, int(1 / max(sparsity_ratio, 0.05)))  # Prevent division by zero

        for i in range(seq_len):
            # Attend to positions at regular intervals
            for j in range(i % adaptive_stride, seq_len, adaptive_stride):
                mask[i, j] = True
            # Also attend to local neighbors
            adaptive_local_size = int(self.local_window_size * sparsity_ratio / 2)
            adaptive_local_size = max(2, adaptive_local_size)
            start_idx = max(0, i - adaptive_local_size)
            end_idx = min(seq_len, i + adaptive_local_size + 1)
            mask[i, start_idx:end_idx] = True

        # Expand mask to match attention weights shape
        mask = mask.unsqueeze(0).unsqueeze(0).expand(1, 1, seq_len, seq_len)

        return mask


def create_adaptive_sparse_attention(
    config: Any,
    layer_idx: Optional[int] = None,
    adaptive_strategy: str = "input_dependent",
    initial_sparse_pattern: str = "longformer",
    sparsity_ratio: float = 0.25,
    block_size: int = 64,
    local_window_size: int = 128,
    use_global_attention: bool = True,
    global_attention_indices: Optional[list] = None,
    attention_dropout: float = 0.0,
    bias: bool = True,
    temperature: float = 1.0,
    adaptation_threshold: float = 0.1,
) -> AdaptiveSparseAttention:
    """
    Factory function to create adaptive sparse attention implementation.

    Args:
        config: Model configuration object
        layer_idx: Index of the transformer layer (optional)
        adaptive_strategy: Strategy for adapting attention patterns
        initial_sparse_pattern: Initial sparse attention pattern to use
        sparsity_ratio: Base ratio of tokens to attend to
        block_size: Block size for block sparse attention
        local_window_size: Window size for local attention
        use_global_attention: Whether to use global attention
        global_attention_indices: Indices for global attention tokens
        attention_dropout: Dropout rate for attention
        bias: Whether to use bias in linear projections
        temperature: Temperature for attention scaling
        adaptation_threshold: Threshold for triggering adaptation

    Returns:
        AdaptiveSparseAttention: The adaptive sparse attention implementation
    """
    return AdaptiveSparseAttention(
        config,
        layer_idx=layer_idx,
        adaptive_strategy=adaptive_strategy,
        initial_sparse_pattern=initial_sparse_pattern,
        sparsity_ratio=sparsity_ratio,
        block_size=block_size,
        local_window_size=local_window_size,
        use_global_attention=use_global_attention,
        global_attention_indices=global_attention_indices,
        attention_dropout=attention_dropout,
        bias=bias,
        temperature=temperature,
        adaptation_threshold=adaptation_threshold,
    )


def get_adaptive_sparse_attention_class():
    """
    Returns the AdaptiveSparseAttention class for dynamic instantiation.

    Returns:
        Type[AdaptiveSparseAttention]: The AdaptiveSparseAttention class
    """
    return AdaptiveSparseAttention


__all__ = [
    "AdaptiveSparseAttention",
    "create_adaptive_sparse_attention",
    "get_adaptive_sparse_attention_class"
]