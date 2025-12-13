#!/usr/bin/env python
"""
Demonstration of the Consolidated Attention Mechanism System.

This script demonstrates the consolidated attention mechanism system
with various attention implementations, hardware-aware selection,
and performance monitoring.
"""

import torch
import torch.nn as nn
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import the attention modules from our implementation
from src.models.consolidated_attention_mechanism import (
    AttentionType,
    StandardAttentionModule,
    FlashAttentionModule,
    SparseAttentionModule,
    MemoryEfficientAttentionModule,
    AttentionManager,
    MultiModelAttentionAdapter,
    create_consolidated_attention_module,
    HardwareAwareAttentionSelector
)
from src.models.model_registry import ModelSpec


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    """Configuration for the demo."""
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    attention_dropout_prob: float = 0.1
    sparsity_factor: float = 0.2
    chunk_size: int = 256
    use_memory_efficient_attention: bool = False
    use_sparse_attention: bool = False
    use_flash_attention_2: bool = False


def benchmark_attention_mechanism(attention_module, name: str, num_iterations: int = 10):
    """Benchmark an attention mechanism."""
    logger.info(f"Benchmarking {name}...")
    
    # Create sample input
    batch_size = 2
    seq_len = 64  # Using smaller sequence length for faster benchmarking
    hidden_size = attention_module.config.hidden_size
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    
    # Warmup
    for _ in range(3):
        _ = attention_module(
            hidden_states=hidden_states,
            position_ids=position_ids,
            output_attentions=False
        )
    
    # Benchmark forward pass
    start_time = time.time()
    for _ in range(num_iterations):
        output, attn_weights, past_key_value = attention_module(
            hidden_states=hidden_states,
            position_ids=position_ids,
            output_attentions=True
        )
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    memory_usage = attention_module.get_memory_usage()
    
    logger.info(f"{name} - Average time per forward pass: {avg_time:.6f}s")
    logger.info(f"{name} - Memory usage: {memory_usage}")
    logger.info(f"{name} - Output shape: {output.shape}")
    
    return avg_time, memory_usage


def demonstrate_attention_variants():
    """Demonstrate different attention variants."""
    logger.info("=== Demonstrating Different Attention Variants ===")
    
    config = DemoConfig()
    
    # Create different attention modules
    attention_modules = {
        "Standard": StandardAttentionModule(config),
        "Memory Efficient": MemoryEfficientAttentionModule(config),
        "Sparse": SparseAttentionModule(config),
    }
    
    # Add Flash Attention if CUDA is available
    if torch.cuda.is_available():
        attention_modules["Flash"] = FlashAttentionModule(config)
    
    # Benchmark each attention variant
    results = {}
    for name, module in attention_modules.items():
        avg_time, memory_usage = benchmark_attention_mechanism(module, name)
        results[name] = {"time": avg_time, "memory": memory_usage}
    
    # Print comparison
    logger.info("\n=== Performance Comparison ===")
    for name, result in results.items():
        logger.info(f"{name}: {result['time']:.6f}s, Memory: {result['memory']}")
    
    return results


def demonstrate_attention_manager():
    """Demonstrate the AttentionManager functionality."""
    logger.info("\n=== Demonstrating Attention Manager ===")
    
    config = DemoConfig()
    manager = AttentionManager(config)
    
    logger.info("Available attention types:")
    for att_type in manager.attention_implementations.keys():
        logger.info(f"  - {att_type.value}")
    
    # Select standard attention
    std_attention = manager.select_attention_module(AttentionType.STANDARD)
    logger.info(f"Selected attention type: {manager.active_attention_type}")
    logger.info(f"Active attention module: {type(std_attention).__name__}")
    
    # Show active attention info
    info = manager.get_active_attention_info()
    logger.info(f"Active attention info: {info}")
    
    # Benchmark different attention types
    logger.info("Benchmarking attention type selection...")
    benchmark_results = manager.benchmark_attention_types(
        torch.randn(2, 32, config.hidden_size)
    )
    
    for att_type, result in benchmark_results.items():
        if result.get('success', False):
            logger.info(f"{att_type.value}: {result['time_seconds']:.6f}s, {result['memory_bytes']} bytes")
        else:
            logger.info(f"{att_type.value}: Failed - {result.get('error', 'Unknown error')}")


def demonstrate_hardware_aware_selection():
    """Demonstrate hardware-aware attention selection."""
    logger.info("\n=== Demonstrating Hardware-Aware Selection ===")

    config = DemoConfig()
    selector = HardwareAwareAttentionSelector()

    # Get available memory (simulated)
    available_memory_gb = 16.0  # Simulate 16GB available memory

    selected_type = selector.select_attention_type(config, available_memory_gb)
    logger.info(f"Selected attention type based on hardware: {selected_type.value}")


def demonstrate_multi_model_adapter():
    """Demonstrate the MultiModelAttentionAdapter."""
    logger.info("\n=== Demonstrating Multi-Model Adapter ===")
    
    config = DemoConfig()
    
    # Create a mock model spec
    model_spec = ModelSpec(
        name="demo_model",
        model_class=nn.Module,
        config_class=DemoConfig,
        adapter_class=None,
        supported_dtypes=["float16", "float32"],
        required_memory_gb=4.0,
        max_sequence_length=2048,
        description="Demo model for attention adapter",
        model_type="language"
    )
    
    # Create adapter
    adapter = MultiModelAttentionAdapter(config, model_spec)
    logger.info(f"Created adapter for model: {model_spec.name}")
    logger.info(f"Adapter attention type: {adapter.attention_module.__class__.__name__}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    
    output, attn_weights, past_key_value = adapter(
        hidden_states=hidden_states,
        position_ids=position_ids,
        output_attentions=True
    )
    
    logger.info(f"Adapter forward pass successful: output shape {output.shape}")
    
    # Get performance stats
    stats = adapter.get_performance_stats()
    logger.info(f"Performance stats: {stats}")


def demonstrate_factory_function():
    """Demonstrate the factory function."""
    logger.info("\n=== Demonstrating Factory Function ===")
    
    config = DemoConfig()
    
    # Create attention module using factory
    attention_module = create_consolidated_attention_module(config)
    logger.info(f"Created attention module using factory: {type(attention_module).__name__}")
    
    # Test forward pass
    batch_size = 1
    seq_len = 16
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    
    output, attn_weights, past_key_value = attention_module(
        hidden_states=hidden_states,
        position_ids=position_ids,
        output_attentions=True
    )
    
    logger.info(f"Factory-created module forward pass successful: output shape {output.shape}")


def main():
    """Main demonstration function."""
    logger.info("Starting Consolidated Attention Mechanism Demonstration")
    
    # Demonstrate different attention variants
    attention_results = demonstrate_attention_variants()
    
    # Demonstrate attention manager
    demonstrate_attention_manager()
    
    # Demonstrate hardware-aware selection
    demonstrate_hardware_aware_selection()
    
    # Demonstrate multi-model adapter
    demonstrate_multi_model_adapter()
    
    # Demonstrate factory function
    demonstrate_factory_function()
    
    logger.info("\n=== Summary ===")
    logger.info("The consolidated attention mechanism system provides:")
    logger.info("1. Multiple attention implementations (standard, sparse, memory-efficient, flash)")
    logger.info("2. Hardware-aware attention selection")
    logger.info("3. Performance monitoring and benchmarking")
    logger.info("4. Multi-model framework integration")
    logger.info("5. Factory functions for easy instantiation")
    logger.info("6. Memory usage tracking and optimization")
    
    logger.info("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()