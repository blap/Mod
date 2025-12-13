"""
Adaptive Sequence Packing for Qwen3-VL model.
Implements dynamic sequence packing algorithms to optimize memory access patterns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math


class SequencePacker(nn.Module):
    """
    Adaptive sequence packing that dynamically packs variable-length sequences
    to minimize padding and optimize memory usage.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.packing_algorithm = getattr(config, 'sequence_packing_algorithm', 'greedy')
        
        # Packing efficiency optimizer
        self.efficiency_optimizer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 2),  # [packing_efficiency, memory_reduction]
            nn.Sigmoid()
        )
        
        # Sequence similarity calculator for grouping
        self.similarity_calculator = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        sequences: List[torch.Tensor],
        attention_masks: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Pack sequences adaptively to minimize padding.
        
        Args:
            sequences: List of [seq_len_i, hidden_size] tensors with different lengths
            attention_masks: Optional list of attention masks for each sequence
            
        Returns:
            Tuple of (packed_sequences, packed_attention_mask, packing_info)
        """
        if not sequences:
            raise ValueError("At least one sequence must be provided")
        
        # Get sequence lengths
        seq_lengths = [seq.size(0) for seq in sequences]
        total_length = sum(seq_lengths)
        
        # Determine optimal packing strategy
        if self.packing_algorithm == 'greedy':
            packed_sequences, packed_masks, packing_info = self._greedy_pack(sequences, attention_masks)
        elif self.packing_algorithm == 'optimal':
            packed_sequences, packed_masks, packing_info = self._optimal_pack(sequences, attention_masks)
        else:
            packed_sequences, packed_masks, packing_info = self._simple_pack(sequences, attention_masks)
        
        # Calculate packing efficiency
        original_total = sum(seq.size(0) for seq in sequences)
        packed_total = packed_sequences.size(1)  # packed length
        
        efficiency = 1.0 - (packed_total / original_total) if original_total > 0 else 0.0
        packing_info['packing_efficiency'] = efficiency
        packing_info['memory_reduction'] = (original_total - packed_total) / original_total if original_total > 0 else 0.0
        
        return packed_sequences, packed_masks, packing_info

    def _simple_pack(
        self, 
        sequences: List[torch.Tensor], 
        attention_masks: Optional[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Simple packing by concatenating sequences."""
        batch_size = len(sequences)
        max_len = sum(seq.size(0) for seq in sequences)
        hidden_size = sequences[0].size(-1)
        
        # Create packed sequence tensor
        packed_seq = torch.zeros(batch_size, max_len, hidden_size, device=sequences[0].device)
        packed_mask = torch.zeros(batch_size, max_len, device=sequences[0].device)
        
        current_pos = 0
        for i, seq in enumerate(sequences):
            seq_len = seq.size(0)
            packed_seq[0, current_pos:current_pos + seq_len, :] = seq  # Only one packed sequence for now
            
            if attention_masks and i < len(attention_masks):
                mask = attention_masks[i]
                packed_mask[0, current_pos:current_pos + seq_len] = mask
            else:
                packed_mask[0, current_pos:current_pos + seq_len] = 1
            
            current_pos += seq_len
        
        packing_info = {
            'packing_strategy': 'simple',
            'num_sequences': len(sequences),
            'original_lengths': [seq.size(0) for seq in sequences],
            'packed_length': current_pos
        }
        
        return packed_seq, packed_mask, packing_info

    def _greedy_pack(
        self, 
        sequences: List[torch.Tensor], 
        attention_masks: Optional[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Greedy packing to minimize padding."""
        # For this implementation, we'll use a simplified approach
        # In practice, this would implement more sophisticated bin packing algorithms
        
        # Sort sequences by length (descending) for better packing
        sorted_indices = sorted(range(len(sequences)), key=lambda i: sequences[i].size(0), reverse=True)
        
        # Pack into fixed-size bins to minimize padding
        bin_size = getattr(self.config, 'sequence_packing_bin_size', 512)
        bins = []
        bin_contents = [[]]
        
        for idx in sorted_indices:
            seq = sequences[idx]
            seq_len = seq.size(0)
            
            # Find a bin that can fit this sequence
            placed = False
            for i, bin_content in enumerate(bin_contents):
                current_bin_size = sum(sequences[j].size(0) for j in bin_content)
                if current_bin_size + seq_len <= bin_size:
                    bin_content.append(idx)
                    placed = True
                    break
            
            if not placed:
                # Create new bin
                bin_contents.append([idx])
        
        # Create packed tensors
        max_bins = len(bin_contents)
        max_bin_length = max(sum(sequences[idx].size(0) for idx in bin_content) for bin_content in bin_contents)
        
        packed_seq = torch.zeros(max_bins, max_bin_length, sequences[0].size(-1), device=sequences[0].device)
        packed_mask = torch.zeros(max_bins, max_bin_length, device=sequences[0].device)
        
        for bin_idx, bin_content in enumerate(bin_contents):
            current_pos = 0
            for idx in bin_content:
                seq = sequences[idx]
                seq_len = seq.size(0)
                
                packed_seq[bin_idx, current_pos:current_pos + seq_len, :] = seq
                
                if attention_masks and idx < len(attention_masks):
                    mask = attention_masks[idx]
                    packed_mask[bin_idx, current_pos:current_pos + seq_len] = mask
                else:
                    packed_mask[bin_idx, current_pos:current_pos + seq_len] = 1
                
                current_pos += seq_len
        
        packing_info = {
            'packing_strategy': 'greedy',
            'num_bins': max_bins,
            'bin_size': bin_size,
            'num_sequences': len(sequences),
            'original_lengths': [sequences[i].size(0) for i in sorted_indices],
            'packed_bin_lengths': [sum(sequences[i].size(0) for i in bin_content) for bin_content in bin_contents]
        }
        
        return packed_seq, packed_mask, packing_info

    def _optimal_pack(
        self, 
        sequences: List[torch.Tensor], 
        attention_masks: Optional[List[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Placeholder for optimal packing algorithm."""
        # This would implement more sophisticated algorithms like first-fit-decreasing
        # For now, we'll use the greedy approach
        return self._greedy_pack(sequences, attention_masks)


class DynamicSequencePacker(nn.Module):
    """
    Dynamic sequence packer that adapts packing strategy based on input characteristics.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        
        # Strategy selector network
        self.strategy_selector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, 3),  # [simple, greedy, optimal]
            nn.Softmax(dim=-1)
        )
        
        # Length predictor for packing
        self.length_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Create packers for different strategies
        self.simple_packer = SequencePacker(config)
        config.sequence_packing_algorithm = 'greedy'
        self.greedy_packer = SequencePacker(config)
        config.sequence_packing_algorithm = 'optimal'
        self.optimal_packer = SequencePacker(config)

    def forward(
        self,
        sequences: List[torch.Tensor],
        attention_masks: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Dynamically select and apply the best packing strategy.
        """
        # Analyze sequences to determine best strategy
        strategy_weights = self._select_strategy(sequences)
        selected_strategy = torch.argmax(strategy_weights, dim=-1).item()
        
        # Apply selected packing strategy
        if selected_strategy == 0:  # Simple
            return self.simple_packer(sequences, attention_masks)
        elif selected_strategy == 1:  # Greedy
            return self.greedy_packer(sequences, attention_masks)
        else:  # Optimal (or fallback to greedy)
            return self.optimal_packer(sequences, attention_masks)

    def _select_strategy(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """Select the best packing strategy based on sequence characteristics."""
        # Calculate average sequence representation
        seq_reprs = []
        for seq in sequences:
            # Use mean pooling to get a representation of the sequence
            seq_repr = seq.mean(dim=0)  # [hidden_size]
            seq_reprs.append(seq_repr)
        
        # Average across all sequences
        avg_repr = torch.stack(seq_reprs).mean(dim=0)  # [hidden_size]
        
        # Get strategy weights
        strategy_weights = self.strategy_selector(avg_repr.unsqueeze(0))  # [1, 3]
        
        return strategy_weights


class MultimodalSequencePacker(nn.Module):
    """
    Sequence packer specifically designed for multimodal inputs.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.vision_hidden_size = config.vision_hidden_size
        
        # Separate packers for different modalities
        self.text_packer = DynamicSequencePacker(config)
        
        vision_config = type(config)()
        vision_config.hidden_size = config.vision_hidden_size
        vision_config.max_position_embeddings = (config.vision_image_size // config.vision_patch_size) ** 2
        self.vision_packer = DynamicSequencePacker(vision_config)
        
        # Cross-modal packing strategy selector
        self.cross_modal_selector = nn.Sequential(
            nn.Linear(self.hidden_size + self.vision_hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 2),  # [separate, joint]
            nn.Softmax(dim=-1)
        )
        
        # Multimodal fusion layer
        self.multimodal_fusion = nn.Linear(self.hidden_size + self.vision_hidden_size, self.hidden_size)

    def forward(
        self,
        text_sequences: Optional[List[torch.Tensor]] = None,
        vision_sequences: Optional[List[torch.Tensor]] = None,
        text_masks: Optional[List[torch.Tensor]] = None,
        vision_masks: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Pack multimodal sequences with adaptive strategies.
        """
        packing_info = {}
        
        if text_sequences is not None and vision_sequences is not None:
            # Both modalities present - multimodal packing
            text_packed, text_mask, text_info = self.text_packer(text_sequences, text_masks)
            vision_packed, vision_mask, vision_info = self.vision_packer(vision_sequences, vision_masks)
            
            # Determine how to combine modalities
            cross_modal_weights = self._get_cross_modal_weights(text_packed, vision_packed)
            combine_jointly = cross_modal_weights[0, 1] > 0.5  # If joint weight > 0.5
            
            if combine_jointly:
                # Combine modalities into a single packed sequence
                batch_size = max(text_packed.size(0), vision_packed.size(0))
                max_len = max(text_packed.size(1), vision_packed.size(1))
                
                # Pad to same dimensions if needed
                if text_packed.size(1) < max_len:
                    pad_len = max_len - text_packed.size(1)
                    text_packed = F.pad(text_packed, (0, 0, 0, pad_len), value=0)
                    text_mask = F.pad(text_mask, (0, pad_len), value=0)
                
                if vision_packed.size(1) < max_len:
                    pad_len = max_len - vision_packed.size(1)
                    vision_packed = F.pad(vision_packed, (0, 0, 0, pad_len), value=0)
                    vision_mask = F.pad(vision_mask, (0, pad_len), value=0)
                
                # Combine modalities
                combined_features = torch.cat([text_packed, vision_packed], dim=-1)
                fused_output = self.multimodal_fusion(combined_features)
                
                # Combine masks
                combined_mask = torch.maximum(text_mask, vision_mask)
                
                packing_info.update({
                    'modality_combination': 'joint',
                    'text_info': text_info,
                    'vision_info': vision_info
                })
                
                return fused_output, combined_mask, packing_info
            else:
                # Keep modalities separate
                packing_info.update({
                    'modality_combination': 'separate',
                    'text_packed': text_packed,
                    'vision_packed': vision_packed,
                    'text_info': text_info,
                    'vision_info': vision_info
                })
                
                # For return, we'll concatenate along batch dimension
                combined_output = torch.cat([text_packed, vision_packed], dim=0)
                combined_mask = torch.cat([text_mask, vision_mask], dim=0)
                
                return combined_output, combined_mask, packing_info
                
        elif text_sequences is not None:
            # Text-only packing
            return self.text_packer(text_sequences, text_masks)
        elif vision_sequences is not None:
            # Vision-only packing
            return self.vision_packer(vision_sequences, vision_masks)
        else:
            raise ValueError("At least one modality must be provided")

    def _get_cross_modal_weights(self, text_packed: torch.Tensor, vision_packed: torch.Tensor) -> torch.Tensor:
        """Get weights for determining cross-modal packing strategy."""
        # Use average representations to determine strategy
        text_repr = text_packed.mean(dim=1).mean(dim=0)  # [hidden_size]
        vision_repr = vision_packed.mean(dim=1).mean(dim=0)  # [vision_hidden_size]
        
        # Pad or project to same dimension if needed
        if text_repr.size(0) != self.hidden_size:
            text_repr = F.pad(text_repr, (0, self.hidden_size - text_repr.size(0)), value=0)
        if vision_repr.size(0) != self.vision_hidden_size:
            vision_repr = F.pad(vision_repr, (0, self.vision_hidden_size - vision_repr.size(0)), value=0)
        
        combined_repr = torch.cat([text_repr, vision_repr], dim=-1).unsqueeze(0)  # [1, total_size]
        
        weights = self.cross_modal_selector(combined_repr)  # [1, 2]
        return weights


class PackedAttention(nn.Module):
    """
    Attention mechanism designed to work with packed sequences.
    """
    def __init__(self, config, layer_idx: Optional[int] = None):
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

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=True)

        self.rotary_emb = Qwen3VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        sequence_lens: Optional[List[int]] = None  # Original sequence lengths for unpacking
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project queries, keys, and values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape to multi-head format
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embeddings
        if position_ids is None:
            position_ids = torch.arange(q_len, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle past key values for caching
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=-2)
            value_states = torch.cat([past_key_value[1], value_states], dim=-2)

        # Repeat keys and values for GQA (Grouped Query Attention) if applicable
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)