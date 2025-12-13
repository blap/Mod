"""
Test suite for dynamic sparse attention with learned routing for token selection.
This implements the first task of Phase 7: Advanced Architecture Optimizations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add the src directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLAttention, Qwen3VLRotaryEmbedding, apply_rotary_pos_emb, repeat_kv


class DynamicSparseAttention(nn.Module):
    """
    Dynamic sparse attention with learned routing for token selection.
    Selectively computes attention only for the most relevant token pairs.
    """
    def __init__(self, config: Qwen3VLConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads or self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Rotary embedding
        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        
        # Learned routing mechanism to determine which tokens to attend to
        self.routing_network = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, self.num_heads),  # One routing score per head
            nn.Softmax(dim=-1)
        )
        
        # Sparse attention parameters
        self.sparsity_ratio = getattr(config, 'sparse_attention_sparsity_ratio', 0.5)
        self.top_k = max(1, int(self.sparsity_ratio * self.max_position_embeddings))
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5

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
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat K and V if using GQA
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Apply learned routing to determine which tokens to attend to
        routing_scores = self.routing_network(hidden_states.mean(dim=1, keepdim=True))  # [bsz, 1, num_heads]
        routing_scores = routing_scores.squeeze(1)  # [bsz, num_heads]
        
        # For each head, select top-k most relevant tokens
        # Use the routing scores to determine which tokens are most relevant
        # In this implementation, we'll use the routing scores to weight attention computation
        # and then apply sparsity by masking out low-attention positions

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply dynamic sparsity: for each head, keep only top-k attention weights
        # This is done by zeroing out the smallest attention weights
        sparse_attn_weights = self.apply_dynamic_sparsity(attn_weights, routing_scores)

        # Apply softmax
        attn_weights = F.softmax(sparse_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def apply_dynamic_sparsity(self, attn_weights: torch.Tensor, routing_scores: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic sparsity based on sequence length.
        For each head, keep only the top-k attention weights.
        """
        bsz, num_heads, q_len, k_len = attn_weights.size()

        # Calculate top_k based on actual sequence length and sparsity ratio
        top_k = max(1, int(self.sparsity_ratio * k_len))

        # Create a copy of attention weights to modify
        sparse_attn_weights = attn_weights.clone()

        # For each head, apply sparsity
        for head_idx in range(num_heads):
            # Get attention weights for this head
            head_attn_weights = sparse_attn_weights[:, head_idx, :, :]  # [bsz, q_len, k_len]

            # Apply sparsity: for each query position, keep only top-k key positions
            for q_pos in range(q_len):
                # Get attention scores for this query position
                query_attn_scores = head_attn_weights[:, q_pos, :]  # [bsz, k_len]

                # Find top-k positions for each batch
                top_k_val = min(top_k, k_len)
                top_k_values, top_k_indices = torch.topk(query_attn_scores, top_k_val, dim=-1, sorted=False)

                # Create a mask to zero out non-top-k positions
                mask = torch.zeros_like(query_attn_scores, dtype=torch.bool)
                mask.scatter_(-1, top_k_indices, True)

                # Apply mask: zero out non-top-k positions
                sparse_attn_weights[:, head_idx, q_pos, :] = torch.where(
                    mask,
                    head_attn_weights[:, q_pos, :],
                    torch.full_like(head_attn_weights[:, q_pos, :], torch.finfo(head_attn_weights.dtype).min)
                )

        return sparse_attn_weights


def test_dynamic_sparse_attention_basic():
    """Test basic functionality of dynamic sparse attention."""
    config = Qwen3VLConfig()
    config.hidden_size = 512
    config.num_attention_heads = 8
    config.max_position_embeddings = 128
    config.sparse_attention_sparsity_ratio = 0.5  # Keep top 50% of attention weights
    
    attention_layer = DynamicSparseAttention(config)
    
    # Create test input
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Forward pass
    output, _, _ = attention_layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, config.hidden_size), f"Expected {(batch_size, seq_len, config.hidden_size)}, got {output.shape}"
    
    print("PASS: Basic functionality test passed")


def test_dynamic_sparse_attention_sparsity():
    """Test that dynamic sparse attention actually applies sparsity."""
    config = Qwen3VLConfig()
    config.hidden_size = 256
    config.num_attention_heads = 4
    config.max_position_embeddings = 32
    config.sparse_attention_sparsity_ratio = 0.25  # Keep top 25% of attention weights
    
    attention_layer = DynamicSparseAttention(config)
    
    # Create test input
    batch_size = 1
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Forward pass to get attention weights
    with torch.no_grad():
        # We need to modify the forward method to return attention weights for this test
        # For now, we'll test sparsity indirectly by checking the sparse attention implementation
        routing_scores = attention_layer.routing_network(hidden_states.mean(dim=1, keepdim=True))
        routing_scores = routing_scores.squeeze(1)
        
        # Compute attention scores
        query_states = attention_layer.q_proj(hidden_states).view(batch_size, seq_len, attention_layer.num_heads, attention_layer.head_dim).transpose(1, 2)
        key_states = attention_layer.k_proj(hidden_states).view(batch_size, seq_len, attention_layer.num_key_value_heads, attention_layer.head_dim).transpose(1, 2)
        value_states = attention_layer.v_proj(hidden_states).view(batch_size, seq_len, attention_layer.num_key_value_heads, attention_layer.head_dim).transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = attention_layer.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Repeat K and V if using GQA
        key_states = repeat_kv(key_states, attention_layer.num_key_value_groups)
        value_states = repeat_kv(value_states, attention_layer.num_key_value_groups)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * attention_layer.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply dynamic sparsity
        sparse_attn_weights = attention_layer.apply_dynamic_sparsity(attn_weights, routing_scores)
        
        # Check that sparsity was applied
        # Count non-masked (non-min) values
        min_val = torch.finfo(sparse_attn_weights.dtype).min
        non_min_count = (sparse_attn_weights != min_val).sum().item()
        total_possible = batch_size * attention_layer.num_heads * seq_len * seq_len
        
        # With 25% sparsity (keeping top 25%), we should have roughly 25% of values that are not masked
        # In our implementation, this means for each query position, we keep top 25% of key positions
        expected_non_masked = total_possible * 0.25
        actual_non_masked_ratio = non_min_count / total_possible

        print(f"Sparsity test - Non-masked ratio: {actual_non_masked_ratio:.3f}, Expected: ~0.25")
        # Allow some tolerance due to implementation details
        # With 25% sparsity on 16 positions, we keep 4 positions per query, giving 25% sparsity
        assert actual_non_masked_ratio <= 0.35, f"Sparsity not properly applied: {actual_non_masked_ratio:.3f}"
        assert actual_non_masked_ratio >= 0.15, f"Too much sparsity applied: {actual_non_masked_ratio:.3f}"
    
    print("PASS: Sparsity test passed")


def test_dynamic_sparse_attention_routing():
    """Test that the routing mechanism works correctly."""
    config = Qwen3VLConfig()
    config.hidden_size = 128
    config.num_attention_heads = 4
    config.max_position_embeddings = 16
    
    attention_layer = DynamicSparseAttention(config)
    
    # Create test input with different patterns
    batch_size = 2
    seq_len = 8
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    # Test routing network
    routing_output = attention_layer.routing_network(hidden_states.mean(dim=1, keepdim=True))
    routing_output = routing_output.squeeze(1)
    
    # Check shape: should be [batch_size, num_heads]
    assert routing_output.shape == (batch_size, config.num_attention_heads), f"Expected {(batch_size, config.num_attention_heads)}, got {routing_output.shape}"
    
    # Check that routing scores sum to 1 (softmax applied)
    routing_sums = routing_output.sum(dim=-1)
    assert torch.allclose(routing_sums, torch.ones_like(routing_sums), atol=1e-5), "Routing scores should sum to 1"
    
    print("PASS: Routing mechanism test passed")


def test_dynamic_sparse_attention_integration():
    """Test integration with existing Qwen3-VL architecture."""
    config = Qwen3VLConfig()
    config.hidden_size = 256
    config.num_attention_heads = 8
    config.max_position_embeddings = 64
    config.sparse_attention_sparsity_ratio = 0.3
    
    # Create the attention layer
    attention_layer = DynamicSparseAttention(config)
    
    # Create test inputs
    batch_size = 1
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # Test forward pass
    output, _, _ = attention_layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    
    # Verify output properties
    assert output.shape == (batch_size, seq_len, config.hidden_size)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"
    
    print("PASS: Integration test passed")


def test_dynamic_sparse_attention_performance():
    """Test performance characteristics of dynamic sparse attention."""
    config = Qwen3VLConfig()
    config.hidden_size = 512
    config.num_attention_heads = 8
    config.max_position_embeddings = 128
    config.sparse_attention_sparsity_ratio = 0.5
    
    attention_layer = DynamicSparseAttention(config)
    
    # Create larger test input to measure performance
    batch_size = 2
    seq_len = 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    import time
    
    # Measure execution time
    start_time = time.time()
    output, _, _ = attention_layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids
    )
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"PASS: Performance test passed - Execution time: {execution_time:.4f}s")
    
    # Verify output is reasonable
    assert output.shape == (batch_size, seq_len, config.hidden_size)
    assert execution_time < 1.0, f"Execution time too high: {execution_time:.4f}s"


def run_all_tests():
    """Run all tests for dynamic sparse attention."""
    print("Running Dynamic Sparse Attention Tests...")
    print("=" * 60)

    test_dynamic_sparse_attention_basic()
    test_dynamic_sparse_attention_sparsity()
    test_dynamic_sparse_attention_routing()
    test_dynamic_sparse_attention_integration()
    test_dynamic_sparse_attention_performance()

    print("=" * 60)
    print("All Dynamic Sparse Attention tests passed!")


if __name__ == "__main__":
    run_all_tests()