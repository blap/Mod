"""
Hierarchical Memory Compression for Qwen3-VL model.
Implements multi-level memory compression with adaptive strategies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math


class HierarchicalMemoryCompressor(nn.Module):
    """
    Hierarchical Memory Compression system with multiple compression strategies
    and adaptive selection based on usage patterns.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Hidden dimensions
        self.hidden_size = config.hidden_size
        self.vision_hidden_size = config.vision_hidden_size
        self.compression_levels = getattr(config, 'hierarchical_compression_levels', 3)
        
        # Multi-level compression components
        self.compression_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size // (2**i)) if i > 0 
            else nn.Identity()  # First level is no compression
            for i in range(self.compression_levels)
        ])
        
        # Decompression layers
        self.decompression_layers = nn.ModuleList([
            nn.Linear(self.hidden_size // (2**i), self.hidden_size) if i > 0 
            else nn.Identity()  # First level is no compression
            for i in range(self.compression_levels)
        ])
        
        # Compression strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 4, self.compression_levels),
            nn.Softmax(dim=-1)
        )
        
        # Frequency-based compression adaptors - adjust to match compressed dimensions
        self.frequency_adaptors = nn.ModuleList()
        for i in range(self.compression_levels):
            if i == 0:
                # No compression level - identity mapping
                self.frequency_adaptors.append(nn.Identity())
            else:
                # Match the compressed size from compression layer
                compressed_size = self.hidden_size // (2**i)
                self.frequency_adaptors.append(nn.Linear(compressed_size, compressed_size))
        
        # Memory access pattern analyzer
        self.access_pattern_analyzer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 2),  # [importance, reuse_frequency]
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        access_frequency: Optional[torch.Tensor] = None,
        importance_score: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compress memory states hierarchically based on access patterns.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            access_frequency: [batch_size, seq_len] - how often each token is accessed
            importance_score: [batch_size, seq_len] - importance of each token
            
        Returns:
            Tuple of (compressed_states, compression_info)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Analyze access patterns if not provided
        if access_frequency is None:
            access_frequency = torch.ones(batch_size, seq_len, device=hidden_states.device)
        
        if importance_score is None:
            importance_score = torch.ones(batch_size, seq_len, device=hidden_states.device)
        
        # Get compression strategy for each position
        strategy_weights = self._get_compression_strategy(
            hidden_states, access_frequency, importance_score
        )
        
        # Apply hierarchical compression
        compressed_states = []
        compression_info = {
            'compression_strategies': [],
            'compression_ratios': [],
            'access_frequencies': access_frequency,
            'importance_scores': importance_score
        }
        
        for pos in range(seq_len):
            pos_states = hidden_states[:, pos, :]  # [batch_size, hidden_size]
            pos_freq = access_frequency[:, pos]    # [batch_size]
            pos_importance = importance_score[:, pos]  # [batch_size]
            
            # Select compression strategy based on weights
            pos_strategy_weights = strategy_weights[:, pos, :]  # [batch_size, compression_levels]
            selected_strategy = torch.argmax(pos_strategy_weights, dim=-1)  # [batch_size]
            
            # Apply compression for each batch item
            pos_compressed = []
            pos_compression_ratios = []

            for b in range(batch_size):
                strategy_idx = selected_strategy[b].item()
                strategy_weight = pos_strategy_weights[b, strategy_idx]

                if strategy_idx == 0:
                    # No compression (level 0)
                    compressed = pos_states[b]
                    compression_ratio = 1.0
                else:
                    # Apply compression
                    compressed_input = pos_states[b]  # [hidden_size]

                    # Compress using the appropriate layer
                    compressed = self.compression_layers[strategy_idx](compressed_input)  # [reduced_size]

                    # Apply frequency adaptor (adjust dimensions)
                    compressed = self.frequency_adaptors[strategy_idx](compressed)  # [reduced_size]

                    # Decompress back to original size
                    compressed = self.decompression_layers[strategy_idx](compressed)  # [hidden_size]

                    compression_ratio = 1.0 / (2**strategy_idx)

                pos_compressed.append(compressed)
                pos_compression_ratios.append(compression_ratio)
            
            pos_compressed = torch.stack(pos_compressed, dim=0)  # [batch_size, hidden_size]
            compressed_states.append(pos_compressed)
            compression_info['compression_strategies'].append(selected_strategy)
            compression_info['compression_ratios'].append(pos_compression_ratios)
        
        compressed_states = torch.stack(compressed_states, dim=1)  # [batch_size, seq_len, hidden_size]
        
        return compressed_states, compression_info

    def _get_compression_strategy(
        self, 
        hidden_states: torch.Tensor, 
        access_frequency: torch.Tensor, 
        importance_score: torch.Tensor
    ) -> torch.Tensor:
        """
        Determine optimal compression strategy for each token.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Concatenate features for strategy selection
        freq_expanded = access_frequency.unsqueeze(-1).expand(-1, -1, hidden_size // 2)
        importance_expanded = importance_score.unsqueeze(-1).expand(-1, -1, hidden_size // 2)
        combined_features = torch.cat([
            hidden_states[:, :, :hidden_size // 2] * freq_expanded,
            hidden_states[:, :, hidden_size // 2:] * importance_expanded
        ], dim=-1)  # [batch_size, seq_len, hidden_size]

        # Adjust dimensions for strategy selector (reduce to expected input size)
        batch_size, seq_len, _ = combined_features.shape
        combined_features_reshaped = combined_features.view(-1, hidden_size)  # [batch_size*seq_len, hidden_size]

        # Get strategy weights
        strategy_weights = self.strategy_selector(combined_features_reshaped)  # [batch_size*seq_len, compression_levels]
        strategy_weights = strategy_weights.view(batch_size, seq_len, -1)  # [batch_size, seq_len, compression_levels]
        
        return strategy_weights

    def compress_sequence(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress an entire sequence using hierarchical compression.
        """
        batch_size, seq_len, hidden_size = sequence.shape
        
        # Analyze the sequence to determine compression parameters
        with torch.no_grad():
            # Compute importance scores based on attention patterns
            attention_scores = torch.matmul(sequence, sequence.transpose(-1, -2)) / math.sqrt(hidden_size)
            attention_scores = F.softmax(attention_scores, dim=-1)
            
            # Compute importance as sum of attention weights
            importance_scores = attention_scores.sum(dim=-1)  # [batch_size, seq_len]
            
            # Compute frequency as 1/position (simplistic model: earlier tokens more frequent)
            position_weights = 1.0 / (torch.arange(seq_len, device=sequence.device).float() + 1.0)
            frequency_scores = position_weights.unsqueeze(0).expand(batch_size, -1)
        
        # Apply hierarchical compression
        compressed_sequence, compression_info = self.forward(
            sequence, 
            access_frequency=frequency_scores,
            importance_score=importance_scores
        )
        
        return compressed_sequence, compression_info


class AdaptiveMemoryBank(nn.Module):
    """
    Adaptive memory bank that uses hierarchical compression based on access patterns.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Memory bank dimensions
        self.hidden_size = config.hidden_size
        self.max_memory_size = getattr(config, 'adaptive_memory_max_size', 4096)
        self.compression_threshold = getattr(config, 'adaptive_memory_compression_threshold', 0.5)
        
        # Hierarchical compressor
        self.hierarchical_compressor = HierarchicalMemoryCompressor(config)
        
        # Memory slots for different compression levels
        self.memory_slots = nn.ParameterDict({
            f'level_{i}': nn.Parameter(
                torch.randn(1, self.max_memory_size, self.hidden_size) * 0.02
            ) for i in range(self.hierarchical_compressor.compression_levels)
        })
        
        # Access counters for each memory slot
        self.access_counts = nn.ParameterDict({
            f'level_{i}': nn.Parameter(
                torch.zeros(1, self.max_memory_size), requires_grad=False
            ) for i in range(self.hierarchical_compressor.compression_levels)
        })
        
        # Memory management network
        self.memory_manager = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 3),  # [store, retrieve, compress]
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        operation: str = 'store_retrieve'  # 'store', 'retrieve', 'store_retrieve'
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Store and/or retrieve memory using hierarchical compression.
        """
        batch_size, seq_len, hidden_size = query.shape
        
        memory_info = {
            'compression_applied': False,
            'memory_utilization': {},
            'access_patterns': {}
        }
        
        if operation in ['store', 'store_retrieve'] and key is not None and value is not None:
            # Store key-value pairs in memory bank
            self._store_memory(key, value)
        
        if operation in ['retrieve', 'store_retrieve']:
            # Retrieve relevant memory
            retrieved_memory = self._retrieve_memory(query)
            return retrieved_memory, memory_info
        
        return query, memory_info

    def _store_memory(self, key: torch.Tensor, value: torch.Tensor):
        """
        Store key-value pairs in the memory bank with adaptive compression.
        """
        batch_size, seq_len, hidden_size = key.shape
        
        # Analyze importance and access patterns
        importance_scores = self._compute_importance_scores(key)
        
        # Compress and store based on importance
        for b in range(batch_size):
            for pos in range(seq_len):
                importance = importance_scores[b, pos].item()
                
                # Select appropriate compression level based on importance
                if importance > 0.7:
                    level = 0  # No compression for important tokens
                elif importance > 0.4:
                    level = 1  # Light compression
                else:
                    level = 2  # Heavy compression
                
                # Store in the appropriate level
                slot_idx = self._find_free_slot(level)
                if slot_idx is not None:
                    self.memory_slots[f'level_{level}'][0, slot_idx, :] = value[b, pos, :]
                    self.access_counts[f'level_{level}'][0, slot_idx] += 1

    def _retrieve_memory(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve relevant memory based on query.
        """
        batch_size, seq_len, hidden_size = query.shape
        
        # For each query position, find relevant stored memory
        retrieved_memory = torch.zeros_like(query)
        
        for b in range(batch_size):
            for pos in range(seq_len):
                query_vec = query[b, pos, :]  # [hidden_size]
                
                # Compute similarity with stored memory across all levels
                best_match = None
                best_similarity = -float('inf')
                
                for level in range(len(self.memory_slots)):
                    memory_level = self.memory_slots[f'level_{level}'][0]  # [max_memory_size, hidden_size]
                    access_counts = self.access_counts[f'level_{level}'][0]  # [max_memory_size]
                    
                    # Compute similarities only for used slots
                    used_mask = access_counts > 0
                    if used_mask.any():
                        used_memory = memory_level[used_mask]  # [used_count, hidden_size]
                        
                        # Compute cosine similarity
                        query_norm = F.normalize(query_vec.unsqueeze(0), p=2, dim=-1)
                        memory_norm = F.normalize(used_memory, p=2, dim=-1)
                        similarities = torch.sum(query_norm * memory_norm, dim=-1)  # [used_count]
                        
                        max_sim, max_idx = torch.max(similarities, dim=0)
                        if max_sim > best_similarity:
                            # Get the actual index in the full memory
                            actual_idx = torch.nonzero(used_mask)[max_idx].item()
                            best_match = memory_level[actual_idx]
                            best_similarity = max_sim
                
                if best_match is not None:
                    retrieved_memory[b, pos, :] = best_match
                else:
                    # If no match found, return the query as is
                    retrieved_memory[b, pos, :] = query_vec
        
        return retrieved_memory

    def _compute_importance_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute importance scores for hidden states.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Use a simple attention-based importance measure
        attention_scores = torch.matmul(hidden_states, hidden_states.transpose(-1, -2)) / math.sqrt(hidden_size)
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        # Importance is the sum of attention weights (how much this token attends to others)
        importance_scores = attention_scores.sum(dim=-1) / seq_len  # Normalize
        
        return importance_scores

    def _find_free_slot(self, level: int) -> Optional[int]:
        """
        Find a free slot in the specified memory level.
        """
        access_counts = self.access_counts[f'level_{level}'][0]  # [max_memory_size]
        
        # Find unused slots
        unused_indices = torch.nonzero(access_counts == 0).squeeze(-1)
        
        if len(unused_indices) > 0:
            return unused_indices[0].item()
        
        # If no unused slots, find least accessed slot
        min_access_idx = torch.argmin(access_counts)
        return min_access_idx.item()


class MemoryEfficientCrossAttention(nn.Module):
    """
    Cross-attention mechanism that uses hierarchical memory compression
    to reduce memory usage during attention computation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Hierarchical memory compression for keys and values
        self.memory_compressor = HierarchicalMemoryCompressor(config)
        
        # Adaptive memory bank
        self.memory_bank = AdaptiveMemoryBank(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with hierarchical memory compression.
        """
        batch_size, tgt_len, hidden_size = hidden_states.shape
        src_len = tgt_len if encoder_hidden_states is None else encoder_hidden_states.shape[1]
        
        # Get query from hidden states
        query = self.q_proj(hidden_states)  # [batch_size, tgt_len, hidden_size]
        
        # Get key and value from encoder hidden states or self
        if encoder_hidden_states is not None:
            key = self.k_proj(encoder_hidden_states)  # [batch_size, src_len, hidden_size]
            value = self.v_proj(encoder_hidden_states)  # [batch_size, src_len, hidden_size]
        else:
            key = self.k_proj(hidden_states)  # [batch_size, src_len, hidden_size]
            value = self.v_proj(hidden_states)  # [batch_size, src_len, hidden_size]
        
        # Apply hierarchical compression to key and value
        compressed_key, key_compression_info = self.memory_compressor(key)
        compressed_value, value_compression_info = self.memory_compressor(value)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        compressed_key = compressed_key.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        compressed_value = compressed_value.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(query, compressed_key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply attention to values
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, compressed_value)

        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None