"""
Memory Access Pattern Analysis for Optimization Opportunities
Used for Phase 2.9: Memory Pooling and Pre-allocation Techniques
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import gc
from collections import defaultdict, Counter


class MemoryAccessAnalyzer:
    """
    Analyzes memory access patterns to identify optimization opportunities
    """
    
    def __init__(self):
        self.access_patterns = []
        self.tensor_reuse_stats = defaultdict(int)
        self.temporal_locality = []
        self.spatial_locality = []
        
    def profile_tensor_access(self, tensor: torch.Tensor, operation: str = "unknown"):
        """Profile access to a tensor"""
        access_info = {
            'timestamp': time.time(),
            'operation': operation,
            'size': tensor.size(),
            'dtype': tensor.dtype,
            'numel': tensor.numel(),
            'element_size': tensor.element_size(),
            'total_size_bytes': tensor.numel() * tensor.element_size(),
            'tensor_id': id(tensor)
        }
        
        self.access_patterns.append(access_info)
        
        # Track tensor reuse
        self.tensor_reuse_stats[id(tensor)] += 1
        
        return access_info
    
    def analyze_reuse_patterns(self) -> Dict:
        """Analyze tensor reuse patterns"""
        reuse_counts = list(self.tensor_reuse_stats.values())
        
        if not reuse_counts:
            return {'error': 'No tensor reuse data available'}
        
        reuse_stats = {
            'total_tensors_accessed': len(reuse_counts),
            'mean_reuse_count': np.mean(reuse_counts),
            'max_reuse_count': np.max(reuse_counts),
            'min_reuse_count': np.min(reuse_counts),
            'reuse_distribution': Counter(reuse_counts),
            'highly_reused_tensors': sum(1 for count in reuse_counts if count > 5),  # Reused more than 5 times
            'one_time_use_tensors': sum(1 for count in reuse_counts if count == 1)
        }
        
        return reuse_stats
    
    def analyze_temporal_locality(self) -> Dict:
        """Analyze temporal locality of memory accesses"""
        if len(self.access_patterns) < 2:
            return {'error': 'Insufficient access pattern data'}
        
        # Calculate time between accesses to the same tensor
        tensor_access_times = defaultdict(list)
        for access in self.access_patterns:
            tensor_access_times[access['tensor_id']].append(access['timestamp'])
        
        time_gaps = []
        for tensor_id, timestamps in tensor_access_times.items():
            if len(timestamps) > 1:
                gaps = np.diff(sorted(timestamps))
                time_gaps.extend(gaps)
        
        if not time_gaps:
            return {'mean_time_gap_seconds': 0, 'temporal_locality_score': 0}
        
        mean_gap = np.mean(time_gaps)
        # Temporal locality score: smaller gaps = better locality
        temporal_locality_score = 1.0 / (1.0 + mean_gap) if mean_gap > 0 else 1.0
        
        locality_stats = {
            'mean_time_gap_seconds': mean_gap,
            'temporal_locality_score': temporal_locality_score,
            'total_time_gaps_measured': len(time_gaps)
        }
        
        return locality_stats
    
    def analyze_spatial_locality(self) -> Dict:
        """Analyze spatial locality based on tensor sizes and access patterns"""
        if not self.access_patterns:
            return {'error': 'No access pattern data available'}
        
        sizes = [access['total_size_bytes'] for access in self.access_patterns]
        
        size_stats = {
            'mean_tensor_size_bytes': np.mean(sizes),
            'median_tensor_size_bytes': np.median(sizes),
            'std_tensor_size_bytes': np.std(sizes),
            'min_tensor_size_bytes': np.min(sizes),
            'max_tensor_size_bytes': np.max(sizes),
            'total_memory_accessed_bytes': sum(sizes),
            'size_distribution': np.histogram(sizes, bins=20)
        }
        
        return size_stats
    
    def identify_optimization_opportunities(self) -> Dict:
        """Identify specific optimization opportunities based on access patterns"""
        reuse_stats = self.analyze_reuse_patterns()
        temporal_stats = self.analyze_temporal_locality()
        spatial_stats = self.analyze_spatial_locality()
        
        opportunities = {
            'reuse_optimization': False,
            'temporal_locality_improvement': False,
            'spatial_locality_improvement': False,
            'size_based_pooling': False,
            'recommendations': []
        }
        
        # Check for reuse optimization opportunities
        if reuse_stats.get('mean_reuse_count', 0) > 2:
            opportunities['reuse_optimization'] = True
            opportunities['recommendations'].append(
                "High tensor reuse detected - implement tensor caching/reuse mechanisms"
            )
        
        # Check for temporal locality improvement
        if temporal_stats.get('temporal_locality_score', 0) < 0.5:
            opportunities['temporal_locality_improvement'] = True
            opportunities['recommendations'].append(
                "Poor temporal locality - consider reordering operations to access data when it's hot in cache"
            )
        
        # Check for spatial locality improvement
        if spatial_stats.get('std_tensor_size_bytes', 0) / spatial_stats.get('mean_tensor_size_bytes', 1) > 2.0:
            opportunities['spatial_locality_improvement'] = True
            opportunities['recommendations'].append(
                "High variance in tensor sizes - consider standardizing tensor dimensions for better cache utilization"
            )
        
        # Check for size-based pooling opportunities
        if spatial_stats.get('total_memory_accessed_bytes', 0) > 1e8:  # 100MB
            opportunities['size_based_pooling'] = True
            opportunities['recommendations'].append(
                "High memory throughput detected - implement memory pooling for frequently used tensor sizes"
            )
        
        return opportunities


def simulate_transformer_memory_access_patterns():
    """Simulate memory access patterns in transformer operations"""
    print("Simulating transformer memory access patterns...")
    
    analyzer = MemoryAccessAnalyzer()
    
    # Simulate a transformer layer's memory access patterns
    batch_size, seq_len, hidden_dim = 1, 512, 4096
    
    # Attention mechanism
    print("  - Simulating attention mechanism...")
    
    # Input tensor
    input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
    analyzer.profile_tensor_access(input_tensor, "input_projection")
    
    # Query, Key, Value projections
    q = torch.randn(batch_size, seq_len, hidden_dim)
    k = torch.randn(batch_size, seq_len, hidden_dim)
    v = torch.randn(batch_size, seq_len, hidden_dim)
    
    analyzer.profile_tensor_access(q, "query_projection")
    analyzer.profile_tensor_access(k, "key_projection") 
    analyzer.profile_tensor_access(v, "value_projection")
    
    # Attention scores
    attn_scores = torch.bmm(q, k.transpose(-2, -1))
    analyzer.profile_tensor_access(attn_scores, "attention_scores")
    
    # Apply attention to values
    attn_output = torch.bmm(torch.softmax(attn_scores, dim=-1), v)
    analyzer.profile_tensor_access(attn_output, "attention_output")
    
    # Residual connection
    residual = input_tensor + attn_output
    analyzer.profile_tensor_access(residual, "residual_connection")
    
    # Layer norm
    ln_output = torch.layer_norm(residual, (hidden_dim,))
    analyzer.profile_tensor_access(ln_output, "layer_norm")
    
    # Feed-forward network
    print("  - Simulating feed-forward network...")
    
    # First linear layer
    ffn_intermediate = torch.randn(batch_size, seq_len, 11008)  # FFN intermediate size
    analyzer.profile_tensor_access(ffn_intermediate, "ffn_intermediate")
    
    # Second linear layer
    ffn_output = torch.randn(batch_size, seq_len, hidden_dim)
    analyzer.profile_tensor_access(ffn_output, "ffn_output")
    
    # Another residual connection
    final_output = ln_output + ffn_output
    analyzer.profile_tensor_access(final_output, "final_output")
    
    # Simulate multiple layers to get better access pattern statistics
    print("  - Simulating multiple transformer layers...")
    for layer_idx in range(5):  # Simulate 5 layers
        # Reuse some patterns from above
        temp_input = torch.randn(batch_size, seq_len, hidden_dim)
        temp_attn = torch.randn(batch_size, seq_len, hidden_dim)
        temp_ffn = torch.randn(batch_size, seq_len, hidden_dim)
        
        analyzer.profile_tensor_access(temp_input, f"layer_{layer_idx}_input")
        analyzer.profile_tensor_access(temp_attn, f"layer_{layer_idx}_attn")
        analyzer.profile_tensor_access(temp_ffn, f"layer_{layer_idx}_ffn")
    
    return analyzer


def analyze_vision_encoder_patterns():
    """Analyze memory access patterns specific to vision encoder operations"""
    print("Analyzing vision encoder memory access patterns...")
    
    analyzer = MemoryAccessAnalyzer()
    
    # Simulate vision encoder operations
    batch_size, channels, height, width = 1, 3, 224, 224  # Typical image input
    
    # Input image
    image = torch.randn(batch_size, channels, height, width)
    analyzer.profile_tensor_access(image, "vision_input")
    
    # Patch embedding (ViT-style)
    patch_size = 16
    num_patches = (height // patch_size) * (width // patch_size)  # 196 patches for 224x224
    patches = torch.randn(batch_size, num_patches, 768)  # Patch embeddings
    analyzer.profile_tensor_access(patches, "patch_embeddings")
    
    # Positional embeddings
    pos_embeddings = torch.randn(1, num_patches + 1, 768)  # +1 for class token
    analyzer.profile_tensor_access(pos_embeddings, "positional_embeddings")
    
    # Transformer layers in vision encoder
    for layer_idx in range(3):  # Simulate 3 vision transformer layers
        layer_input = torch.randn(batch_size, num_patches + 1, 768)
        layer_attn = torch.randn(batch_size, num_patches + 1, 768)
        layer_ffn = torch.randn(batch_size, num_patches + 1, 768)
        
        analyzer.profile_tensor_access(layer_input, f"vision_layer_{layer_idx}_input")
        analyzer.profile_tensor_access(layer_attn, f"vision_layer_{layer_idx}_attn")
        analyzer.profile_tensor_access(layer_ffn, f"vision_layer_{layer_idx}_ffn")
    
    return analyzer


def run_memory_access_analysis():
    """Run comprehensive memory access pattern analysis"""
    print("Running memory access pattern analysis...")
    
    # Analyze transformer patterns
    transformer_analyzer = simulate_transformer_memory_access_patterns()
    
    # Analyze vision encoder patterns
    vision_analyzer = analyze_vision_encoder_patterns()
    
    # Combine analyzers for overall analysis
    combined_analyzer = MemoryAccessAnalyzer()
    combined_analyzer.access_patterns = (
        transformer_analyzer.access_patterns + 
        vision_analyzer.access_patterns
    )
    
    # Update tensor reuse stats
    for tid, count in transformer_analyzer.tensor_reuse_stats.items():
        combined_analyzer.tensor_reuse_stats[tid] = count
    for tid, count in vision_analyzer.tensor_reuse_stats.items():
        combined_analyzer.tensor_reuse_stats[tid] = count
    
    # Analyze patterns
    reuse_stats = combined_analyzer.analyze_reuse_patterns()
    temporal_stats = combined_analyzer.analyze_temporal_locality()
    spatial_stats = combined_analyzer.analyze_spatial_locality()
    optimization_opps = combined_analyzer.identify_optimization_opportunities()
    
    print("\n=== Memory Access Analysis Results ===")
    print(f"Total tensors accessed: {reuse_stats.get('total_tensors_accessed', 0)}")
    print(f"Mean reuse count: {reuse_stats.get('mean_reuse_count', 0):.2f}")
    print(f"Highly reused tensors (>5x): {reuse_stats.get('highly_reused_tensors', 0)}")
    print(f"One-time use tensors: {reuse_stats.get('one_time_use_tensors', 0)}")
    print(f"Temporal locality score: {temporal_stats.get('temporal_locality_score', 0):.4f}")
    print(f"Mean tensor size: {spatial_stats.get('mean_tensor_size_bytes', 0) / 1024 / 1024:.2f} MB")
    print(f"Total memory accessed: {spatial_stats.get('total_memory_accessed_bytes', 0) / 1024 / 1024:.2f} MB")
    
    print("\n=== Optimization Opportunities ===")
    for rec in optimization_opps.get('recommendations', []):
        print(f"  - {rec}")
    
    return combined_analyzer


if __name__ == "__main__":
    print("Starting memory access pattern analysis...")
    
    analyzer = run_memory_access_analysis()
    
    print("\nMemory access pattern analysis completed!")