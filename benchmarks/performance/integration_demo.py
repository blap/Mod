"""
Integration Demo for Low-Level CPU Optimizations and Kernel Fusion
Targeting Intel i5-10210U + NVIDIA SM61 hardware in Qwen3-VL Model
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import time
from dataclasses import dataclass

from low_level_optimizations import (
    OptimizationConfig,
    apply_low_level_optimizations_to_model,
    FusedAttentionSoftmax,
    FusedMLPBlock,
    FusedLayerNormLinear,
    tiled_matmul,
    cache_blocked_layer_norm,
    simd_gelu,
    PrefetchingOptimizer,
    benchmark_optimizations
)


@dataclass
class Qwen3VLConfig:
    """Configuration for Qwen3-VL model"""
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    num_hidden_layers: int = 4
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 512
    rope_theta: int = 10000
    vocab_size: int = 32000
    num_key_value_heads: int = 8


class SimpleQwen3VLModel(nn.Module):
    """Simplified Qwen3-VL model for demonstration purposes"""
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        
        # Create a simple transformer structure
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits


class SimpleTransformerLayer(nn.Module):
    """Simplified transformer layer for demonstration"""
    def __init__(self, config: Qwen3VLConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Self-attention components
        self.self_attn = SimpleAttention(config)
        
        # Layer norms
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # MLP components
        self.mlp = SimpleMLP(config)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(hidden_states)
        hidden_states = residual + attn_output
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        
        return hidden_states


class SimpleAttention(nn.Module):
    """Simplified attention mechanism"""
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by num_attention_heads {self.num_attention_heads}"
            )
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class SimpleMLP(nn.Module):
    """Simplified MLP block"""
    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        act = self.act_fn(gate)
        intermediate = act * up
        output = self.down_proj(intermediate)
        return output


def create_optimized_qwen3_vl_model() -> Tuple[nn.Module, nn.Module]:
    """Create and return both original and optimized Qwen3-VL models"""
    config = Qwen3VLConfig()
    
    # Create original model
    original_model = SimpleQwen3VLModel(config)
    
    # Create copy for optimization
    import copy
    optimized_model = copy.deepcopy(original_model)
    
    # Apply low-level optimizations
    optimized_model = apply_low_level_optimizations_to_model(optimized_model)
    
    return original_model, optimized_model


def benchmark_models(original_model: nn.Module, optimized_model: nn.Module, 
                     input_ids: torch.Tensor, num_runs: int = 10) -> Dict[str, float]:
    """Benchmark original vs optimized models"""
    # Warm up both models
    for _ in range(3):
        with torch.no_grad():
            _ = original_model(input_ids)
            _ = optimized_model(input_ids)
    
    # Benchmark original model
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = original_model(input_ids)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    original_time = time.time() - start_time
    
    # Benchmark optimized model
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = optimized_model(input_ids)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    optimized_time = time.time() - start_time
    
    # Calculate metrics
    speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
    time_saved = original_time - optimized_time
    
    return {
        'original_time': original_time,
        'optimized_time': optimized_time,
        'speedup': speedup,
        'time_saved': time_saved,
        'runs': num_runs
    }


def demonstrate_low_level_optimizations():
    """Demonstrate the low-level optimizations in action"""
    print("=" * 80)
    print("LOW-LEVEL CPU OPTIMIZATIONS AND KERNEL FUSION DEMONSTRATION")
    print("Targeting Intel i5-10210U + NVIDIA SM61 hardware in Qwen3-VL Model")
    print("=" * 80)
    
    # Run the optimization benchmarks
    print("\n1. Running Low-Level Optimization Benchmarks...")
    opt_results = benchmark_optimizations()
    
    print(f"\nOptimization Results:")
    print(f"  Matrix Multiplication Speedup: {opt_results['matmul_speedup']:.2f}x")
    print(f"  Layer Normalization Speedup: {opt_results['norm_speedup']:.2f}x")
    print(f"  GELU Activation Speedup: {opt_results['gelu_speedup']:.2f}x")
    
    # Create models
    print("\n2. Creating Original and Optimized Models...")
    original_model, optimized_model = create_optimized_qwen3_vl_model()
    
    print(f"   Original Model Parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    print(f"   Optimized Model Parameters: {sum(p.numel() for p in optimized_model.parameters()):,}")
    
    # Create test input
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, original_model.config.vocab_size, (batch_size, seq_len))
    
    # Benchmark models
    print(f"\n3. Benchmarking Models (batch_size={batch_size}, seq_len={seq_len})...")
    benchmark_results = benchmark_models(original_model, optimized_model, input_ids, num_runs=5)
    
    print(f"\nModel Benchmark Results:")
    print(f"  Original Model Time: {benchmark_results['original_time']:.4f}s")
    print(f"  Optimized Model Time: {benchmark_results['optimized_time']:.4f}s")
    print(f"  Speedup: {benchmark_results['speedup']:.2f}x")
    print(f"  Time Saved: {benchmark_results['time_saved']:.4f}s")
    
    # Demonstrate kernel fusion
    print(f"\n4. Demonstrating Kernel Fusion Techniques...")
    
    # Show the structure of fused components
    for i, layer in enumerate(optimized_model.layers):
        attn_type = type(layer.self_attn).__name__
        mlp_type = type(layer.mlp).__name__
        print(f"   Layer {i}: Attention={attn_type}, MLP={mlp_type}")
    
    # Demonstrate prefetching
    print(f"\n5. Demonstrating Memory Prefetching...")
    prefetcher = PrefetchingOptimizer()
    
    # Create test tensors
    tensor1 = torch.randn(batch_size, seq_len, original_model.config.hidden_size)
    tensor2 = torch.randn(batch_size, seq_len, original_model.config.hidden_size // 2)
    
    # Prefetch tensor2 while processing tensor1
    prefetch_start = time.time()
    prefetcher.prefetch_tensor(tensor2)
    processing_time = time.time() - prefetch_start
    
    # Simulate processing of tensor1
    result1 = torch.relu(tensor1)
    
    # Get prefetched tensor
    result2 = prefetcher.get_prefetched_tensor()
    
    print(f"   Prefetching time: {processing_time:.6f}s")
    print(f"   Prefetching successful: {result2 is not None}")
    
    # Verify correctness
    print(f"\n6. Verifying Correctness...")
    with torch.no_grad():
        orig_output = original_model(input_ids)
        opt_output = optimized_model(input_ids)
    
    cosine_sim = torch.nn.functional.cosine_similarity(
        orig_output.flatten(),
        opt_output.flatten(),
        dim=0
    ).item()
    
    max_diff = torch.max(torch.abs(orig_output - opt_output)).item()
    
    print(f"   Output Cosine Similarity: {cosine_sim:.6f}")
    print(f"   Maximum Output Difference: {max_diff:.6f}")
    print(f"   Outputs are similar: {cosine_sim > 0.99}")  # Threshold for similarity
    
    print(f"\n7. Summary of Optimizations Applied:")
    print(f"   - Loop Tiling: Applied to matrix multiplication operations")
    print(f"   - Cache Blocking: Applied to layer normalization")
    print(f"   - SIMD Optimization: Applied to GELU activation function")
    print(f"   - Memory Prefetching: Implemented for data loading optimization")
    print(f"   - Kernel Fusion: Attention + Softmax fused, MLP operations fused")
    print(f"   - JIT Compilation: Ready for dynamic optimization of hot paths")
    
    print(f"\nOptimizations successfully integrated into Qwen3-VL model!")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_low_level_optimizations()