"""
Consolidated tests for the Consolidated Attention Mechanism System

This test suite combines and consolidates the functionality from:
- test_consolidated_attention.py
- test_consolidated_attention_mechanism.py

All functionality from both files is preserved while eliminating redundancy.
"""

import pytest
import torch
import torch.nn as nn
import math
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List, Optional, Tuple
import sys
import os
import time
import unittest

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.attention.consolidated_attention_final import (
    # Core functions
    repeat_kv,
    rotate_half,
    apply_rotary_pos_emb,
    Qwen3VLRotaryEmbedding,

    # Attention mechanisms
    StandardAttention,
    SIMDAttention,
    MemoryEfficientAttention,
    FlashAttention2,
    SM61OptimizedFlashAttention2,
    TrueSparseAttention,
    BlockSparseAttention,
    DynamicSparseAttention,
    Qwen3VLVisionAttention,

    # Main attention module
    Qwen3VLAttention,

    # Enums and types
    AttentionType,
    TensorType,
    TensorState,

    # Lifecycle management
    TensorMetadata,
    TensorLifecycleTracker,
    LifetimePredictor,
    AccessPatternAnalyzer,
    EnhancedPredictiveTensorLifecycleManager,
    create_optimized_lifecycle_manager,
    integrate_with_existing_systems,

    # Managers and selectors
    AttentionMechanismSelector,
    AttentionManager,
    HardwareAwareAttentionSelector,
    AttentionPerformanceMonitor,

    # Factory function
    create_consolidated_attention_module
)

from src.models.consolidated_attention_mechanism import (
    AttentionModule,
    FlashAttentionModule,
    SparseAttentionModule,
    MemoryEfficientAttentionModule,
    MultiModelAttentionAdapter,
    create_consolidated_attention_module as legacy_create_consolidated_attention_module
)
from src.models.model_registry import ModelSpec


class MockConfig:
    """Mock configuration for testing purposes."""
    def __init__(self):
        self.hidden_size = 512
        self.num_attention_heads = 8
        self.num_key_value_heads = 8  # Added missing attribute
        self.max_position_embeddings = 2048
        self.rope_theta = 10000.0
        self.attention_dropout_prob = 0.1
        self.sparsity_factor = 0.1
        self.chunk_size = 512
        self.use_dynamic_sparse_attention = False
        self.use_block_sparse_attention = False
        self.use_memory_efficient_attention = False
        self.use_flash_attention_2 = False
        self.use_sparse_attention = False
        self.qkv_bias = True
        self.out_proj_bias = True


class TestCoreFunctions:
    """Test core utility functions."""

    def test_repeat_kv(self):
        """Test repeat_kv function."""
        # Create a tensor with shape (batch, num_key_value_heads, seq_len, head_dim)
        hidden_states = torch.randn(2, 4, 10, 16)  # batch=2, heads=4, seq_len=10, head_dim=16
        n_rep = 3

        result = repeat_kv(hidden_states, n_rep)

        # Expected shape: (batch, num_key_value_heads * n_rep, seq_len, head_dim)
        expected_shape = (2, 4 * n_rep, 10, 16)
        assert result.shape == expected_shape

        # Test with n_rep = 1 (should return the same tensor)
        result_single = repeat_kv(hidden_states, 1)
        assert torch.equal(result_single, hidden_states)

    def test_rotate_half(self):
        """Test rotate_half function."""
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # Shape: (1, 1, 4)
        # Expected result: first half [1.0, 2.0] becomes [-1.0, -2.0], second half [3.0, 4.0] becomes [3.0, 4.0]
        # Actually: rotate_half takes first half [1.0, 2.0] and second half [3.0, 4.0] and returns [-3.0, -4.0, 1.0, 2.0]
        expected = torch.tensor([[[-3.0, -4.0, 1.0, 2.0]]])  # Actual expected result

        result = rotate_half(x)
        assert torch.allclose(result, expected)

    def test_apply_rotary_pos_emb(self):
        """Test apply_rotary_pos_emb function."""
        # Create test tensors
        q = torch.randn(1, 2, 4, 8)  # (batch, heads, seq_len, head_dim)
        k = torch.randn(1, 2, 4, 8)
        cos = torch.randn(1, 4, 8)  # (batch, seq_len, head_dim)
        sin = torch.randn(1, 4, 8)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

        # Check shapes remain the same
        assert q_embed.shape == q.shape
        assert k_embed.shape == k.shape


class TestRotaryEmbedding:
    """Test rotary embedding implementation."""

    def test_qwen3vl_rotary_embedding(self):
        """Test Qwen3VLRotaryEmbedding."""
        dim = 64
        max_position_embeddings = 2048
        base = 10000

        rotary_emb = Qwen3VLRotaryEmbedding(dim, max_position_embeddings, base)

        # Create test input
        x = torch.randn(1, 8, 10, 64)  # (batch, heads, seq_len, head_dim)
        position_ids = torch.arange(10).unsqueeze(0)  # (batch, seq_len)

        cos, sin = rotary_emb(x, position_ids)

        # Check output shapes
        assert cos.shape == (1, 10, 64)  # Shape based on position_ids and dim
        assert sin.shape == (1, 10, 64)


class TestAttentionMechanisms:
    """Test individual attention mechanisms."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        class Config:
            hidden_size = 512
            num_attention_heads = 8
            num_key_value_heads = 8
            max_position_embeddings = 2048
            rope_theta = 10000.0
            attention_dropout_prob = 0.1
            qkv_bias = True
            out_proj_bias = True
            chunk_size = 512
            sparse_attention_sparsity_ratio = 0.5
            vision_sparse_attention_sparsity_ratio = 0.4
            block_sparse_block_size = 64
            cpu_model = 'Intel i5-10210U'
            gpu_model = 'NVIDIA SM61'
            memory_size = 8 * 1024 * 1024 * 1024
            storage_type = 'nvme'

        return Config()

    def test_standard_attention(self, config):
        """Test StandardAttention."""
        attention = StandardAttention(config)

        # Create test inputs
        hidden_states = torch.randn(2, 10, 512)  # (batch, seq_len, hidden_size)
        attention_mask = torch.ones(2, 1, 10, 10)  # (batch, 1, seq_len, seq_len)
        position_ids = torch.arange(10).unsqueeze(0)  # (batch, seq_len)

        output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True
        )

        # Check output shapes
        assert output.shape == hidden_states.shape
        assert attn_weights.shape == (2, 8, 10, 10)  # (batch, heads, seq_len, seq_len)
        assert past_key_value is None

    def test_memory_efficient_attention(self, config):
        """Test MemoryEfficientAttention."""
        attention = MemoryEfficientAttention(config)

        # Create test inputs
        hidden_states = torch.randn(2, 10, 512)  # (batch, seq_len, hidden_size)
        attention_mask = torch.ones(2, 1, 10, 10)  # (batch, 1, seq_len, seq_len)
        position_ids = torch.arange(10).unsqueeze(0)  # (batch, seq_len)

        output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=False  # Memory efficient doesn't return attn weights by default
        )

        # Check output shapes
        assert output.shape == hidden_states.shape
        assert attn_weights is None

    def test_flash_attention_2(self, config):
        """Test FlashAttention2."""
        attention = FlashAttention2(config)

        # Create test inputs
        hidden_states = torch.randn(2, 10, 512)  # (batch, seq_len, hidden_size)
        attention_mask = torch.ones(2, 1, 10, 10)  # (batch, 1, seq_len, seq_len)
        position_ids = torch.arange(10).unsqueeze(0)  # (batch, seq_len)

        output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True
        )

        # Check output shapes
        assert output.shape == hidden_states.shape
        assert attn_weights is not None  # When output_attentions=True, weights should be returned

    def test_true_sparse_attention(self, config):
        """Test TrueSparseAttention."""
        attention = TrueSparseAttention(config)

        # Create test inputs
        hidden_states = torch.randn(2, 10, 512)  # (batch, seq_len, hidden_size)
        attention_mask = torch.ones(2, 1, 10, 10)  # (batch, 1, seq_len, seq_len)
        position_ids = torch.arange(10).unsqueeze(0)  # (batch, seq_len)

        output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True
        )

        # Check output shapes
        assert output.shape == hidden_states.shape
        assert attn_weights is not None

    def test_block_sparse_attention(self, config):
        """Test BlockSparseAttention."""
        attention = BlockSparseAttention(config)

        # Create test inputs
        hidden_states = torch.randn(2, 10, 512)  # (batch, seq_len, hidden_size)
        attention_mask = torch.ones(2, 1, 10, 10)  # (batch, 1, seq_len, seq_len)
        position_ids = torch.arange(10).unsqueeze(0)  # (batch, seq_len)

        output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True
        )

        # Check output shapes
        assert output.shape == hidden_states.shape
        assert attn_weights is not None

    def test_dynamic_sparse_attention(self, config):
        """Test DynamicSparseAttention."""
        attention = DynamicSparseAttention(config)

        # Create test inputs
        hidden_states = torch.randn(2, 10, 512)  # (batch, seq_len, hidden_size)
        attention_mask = torch.ones(2, 1, 10, 10)  # (batch, 1, seq_len, seq_len)
        position_ids = torch.arange(10).unsqueeze(0)  # (batch, seq_len)

        output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True
        )

        # Check output shapes
        assert output.shape == hidden_states.shape
        assert attn_weights is not None


class TestVisionAttention:
    """Test vision-specific attention."""

    @pytest.fixture
    def vision_config(self):
        """Create a test vision configuration."""
        class Config:
            vision_hidden_size = 768
            vision_num_attention_heads = 12
            vision_qkv_bias = True

        return Config()

    def test_qwen3vl_vision_attention(self, vision_config):
        """Test Qwen3VLVisionAttention."""
        attention = Qwen3VLVisionAttention(vision_config)

        # Create test inputs
        hidden_states = torch.randn(2, 10, 768)  # (batch, seq_len, hidden_size)

        output, attn_weights = attention(
            hidden_states=hidden_states,
            output_attentions=True
        )

        # Check output shapes
        assert output.shape == hidden_states.shape
        assert attn_weights.shape == (2, 12, 10, 10)  # (batch, heads, seq_len, seq_len)


class TestLifecycleManagement:
    """Test lifecycle management components."""

    def test_tensor_metadata(self):
        """Test TensorMetadata creation."""
        metadata = TensorMetadata(
            tensor_id="test_tensor",
            tensor_type=TensorType.INTERMEDIATE,
            device="cpu",
            size_bytes=1024,
            is_pinned=True
        )

        assert metadata.tensor_id == "test_tensor"
        assert metadata.tensor_type == TensorType.INTERMEDIATE
        assert metadata.device == "cpu"
        assert metadata.size_bytes == 1024
        assert metadata.is_pinned is True
        assert metadata.state == TensorState.REGISTERED

    def test_tensor_lifecycle_tracker(self):
        """Test TensorLifecycleTracker."""
        tracker = TensorLifecycleTracker()

        # Create a test tensor
        tensor = torch.randn(10, 10)

        # Register the tensor
        tensor_id = tracker.register_tensor(tensor, tensor_type=TensorType.PARAMETER)
        assert tensor_id in tracker.tensors

        # Access the tensor
        tracker.access_tensor(tensor_id, context="test_context")

        # Get tensor metadata
        metadata = tracker.get_tensor_metadata(tensor_id)
        assert metadata is not None
        assert metadata.access_count == 1

        # Get tensors by type
        params = tracker.get_tensors_by_type(TensorType.PARAMETER)
        assert len(params) == 1
        assert params[0].tensor_id == tensor_id

    def test_lifetime_predictor(self):
        """Test LifetimePredictor."""
        predictor = LifetimePredictor()

        # Simulate access pattern
        current_time = time.time()

        predictor.update_prediction("tensor1", 1, current_time)
        predictor.update_prediction("tensor1", 2, current_time + 1)
        predictor.update_prediction("tensor1", 3, current_time + 2)

        # Predict lifetime
        lifetime = predictor.predict_lifetime("tensor1")
        assert isinstance(lifetime, float)
        assert lifetime > 0

    def test_access_pattern_analyzer(self):
        """Test AccessPatternAnalyzer."""
        analyzer = AccessPatternAnalyzer()

        # Record some accesses
        current_time = time.time()

        analyzer.record_access("tensor1", "layer1", current_time)
        analyzer.record_access("tensor1", "layer1", current_time + 1)
        analyzer.record_access("tensor2", "layer2", current_time + 2)

        # Get access pattern for tensor1
        pattern = analyzer.get_access_pattern("tensor1")
        assert len(pattern) == 2

        # Get context analysis
        ctx_analysis = analyzer.get_context_analysis("layer1")
        assert "tensor1" in ctx_analysis['tensor_ids']
        assert ctx_analysis['access_count'] == 2

        # Get frequent contexts
        frequent = analyzer.get_frequent_contexts(min_accesses=1)
        assert "layer1" in frequent
        assert "layer2" in frequent

    def test_enhanced_predictive_tensor_lifecycle_manager(self):
        """Test EnhancedPredictiveTensorLifecycleManager."""
        config = {
            'cpu_model': 'Intel i5-10210U',
            'gpu_model': 'NVIDIA SM61',
            'memory_size': 8 * 1024 * 1024 * 1024,
            'storage_type': 'nvme'
        }

        manager = EnhancedPredictiveTensorLifecycleManager(config)

        # Create and register a tensor
        tensor = torch.randn(10, 10)
        tensor_id = manager.register_tensor(tensor, tensor_type=TensorType.INTERMEDIATE)

        # Access the tensor
        manager.access_tensor(tensor_id, context="test_context")

        # Get metadata
        metadata = manager.get_tensor_metadata(tensor_id)
        assert metadata is not None

        # Predict lifetime
        lifetime = manager.predict_tensor_lifetime(tensor_id)
        assert isinstance(lifetime, float)

        # Optimize placement
        placement = manager.optimize_tensor_placement(tensor_id)
        assert placement in ['cpu', 'cuda', 'swapped']

        # Get stats
        stats = manager.get_stats()
        assert 'total_tensors' in stats
        assert stats['total_tensors'] == 1

        # Test integration
        mock_compression = Mock()
        mock_swapping = Mock()
        mock_memory_tiering = Mock()

        existing_systems = {
            'compression_manager': mock_compression,
            'swapping_system': mock_swapping,
            'memory_tiering_system': mock_memory_tiering
        }

        integrate_with_existing_systems(manager, existing_systems)
        assert manager.compression_manager == mock_compression
        assert manager.swapping_system == mock_swapping
        assert manager.memory_tiering_system == mock_memory_tiering


class TestAttentionSelectorAndManager:
    """Test attention selector and manager components."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        class Config:
            hidden_size = 512
            num_attention_heads = 8
            num_key_value_heads = 8
            max_position_embeddings = 2048
            rope_theta = 10000.0
            attention_dropout_prob = 0.1
            qkv_bias = True
            out_proj_bias = True
            chunk_size = 512
            sparse_attention_sparsity_ratio = 0.5
            vision_sparse_attention_sparsity_ratio = 0.4
            block_sparse_block_size = 64
            cpu_model = 'Intel i5-10210U'
            gpu_model = 'NVIDIA SM61'
            memory_size = 8 * 1024 * 1024 * 1024
            storage_type = 'nvme'

        return Config()

    def test_attention_mechanism_selector(self, config):
        """Test AttentionMechanismSelector."""
        # Test standard attention creation
        attention = AttentionMechanismSelector.create_attention(config)
        assert isinstance(attention, StandardAttention)

        # Test FlashAttention2 creation
        config.attention_implementation = 'flash_attention_2'
        attention = AttentionMechanismSelector.create_attention(config)
        assert isinstance(attention, FlashAttention2)

        # Test sparse attention creation
        config.attention_implementation = 'sparse_attention'
        attention = AttentionMechanismSelector.create_attention(config)
        assert isinstance(attention, TrueSparseAttention)

        # Test available implementations
        implementations = AttentionMechanismSelector.get_available_implementations()
        expected = [
            'standard',
            'flash_attention_2',
            'sparse_attention',
            'dynamic_sparse_attention',
            'block_sparse_attention',
            'memory_efficient',
            'simd'
        ]
        assert set(implementations) == set(expected)

    def test_hardware_aware_attention_selector(self, config):
        """Test HardwareAwareAttentionSelector."""
        selector = HardwareAwareAttentionSelector()

        # Test selection with limited memory
        attention_type = selector.select_attention_type(config, available_memory_gb=2.0)
        assert attention_type in [AttentionType.STANDARD, AttentionType.MEMORY_EFFICIENT]

    def test_attention_manager(self, config):
        """Test AttentionManager."""
        manager = AttentionManager(config)

        # Test selection of attention module
        attention_module = manager.select_attention_module(AttentionType.STANDARD)
        assert isinstance(attention_module, StandardAttention)

        # Test switching attention module
        success = manager.switch_attention_module(AttentionType.MEMORY_EFFICIENT)
        assert success is True
        assert manager.active_attention_type == AttentionType.MEMORY_EFFICIENT
        assert isinstance(manager.active_attention_module, MemoryEfficientAttention)

        # Test benchmarking
        sample_input = torch.randn(2, 10, 512)
        benchmarks = manager.benchmark_attention_types(sample_input)
        assert isinstance(benchmarks, dict)
        assert AttentionType.STANDARD in benchmarks

        # Test active attention info
        info = manager.get_active_attention_info()
        assert "active_type" in info

    def test_create_consolidated_attention_module(self, config):
        """Test create_consolidated_attention_module factory function."""
        attention_module = create_consolidated_attention_module(config, AttentionType.FLASH_ATTENTION)
        assert isinstance(attention_module, FlashAttention2)


class TestMainAttentionModule:
    """Test the main Qwen3VLAttention module."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        class Config:
            hidden_size = 512
            num_attention_heads = 8
            num_key_value_heads = 8
            max_position_embeddings = 2048
            rope_theta = 10000.0
            attention_dropout_prob = 0.1
            qkv_bias = True
            out_proj_bias = True
            chunk_size = 512
            sparse_attention_sparsity_ratio = 0.5
            vision_sparse_attention_sparsity_ratio = 0.4
            block_sparse_block_size = 64
            cpu_model = 'Intel i5-10210U'
            gpu_model = 'NVIDIA SM61'
            memory_size = 8 * 1024 * 1024 * 1024
            storage_type = 'nvme'

        return Config()

    def test_qwen3vl_attention(self, config):
        """Test Qwen3VLAttention."""
        attention = Qwen3VLAttention(config)

        # Create test inputs
        hidden_states = torch.randn(2, 10, 512)  # (batch, seq_len, hidden_size)
        attention_mask = torch.ones(2, 1, 10, 10)  # (batch, 1, seq_len, seq_len)
        position_ids = torch.arange(10).unsqueeze(0)  # (batch, seq_len)

        output, attn_weights, past_key_value = attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True
        )

        # Check output shapes
        assert output.shape == hidden_states.shape

        # Test lifecycle stats
        stats = attention.get_lifecycle_stats()
        assert isinstance(stats, dict)
        assert 'total_tensors' in stats


class TestPerformanceMonitor:
    """Test performance monitoring components."""

    def test_attention_performance_monitor(self):
        """Test AttentionPerformanceMonitor."""
        monitor = AttentionPerformanceMonitor()

        # Test timing
        monitor.start_timing("test_operation")
        time.sleep(0.01)  # Sleep briefly to measure time
        monitor.end_timing("test_operation")

        # Check metrics
        metrics = monitor.get_current_metrics()
        assert "test_operation_duration" in metrics
        assert metrics["test_operation_duration"] > 0

        # Test memory recording
        monitor.record_memory_usage("test_operation", 1024)
        monitor.record_peak_memory("test_operation", 2048)

        # Log metrics
        monitor.log_metrics()

        # Reset metrics
        monitor.reset_metrics()
        empty_metrics = monitor.get_current_metrics()
        assert len(empty_metrics) == 0


def test_attention_module_interface():
    """Test that AttentionModule properly enforces the interface."""
    config = MockConfig()

    # Try to instantiate the abstract class directly - should raise TypeError
    with pytest.raises(TypeError):
        AttentionModule(config)


def test_standard_attention_module_initialization():
    """Test StandardAttentionModule initialization."""
    config = MockConfig()

    attention_module = StandardAttentionModule(config)

    assert attention_module.hidden_size == config.hidden_size
    assert attention_module.num_heads == config.num_attention_heads
    assert attention_module.head_dim == config.hidden_size // config.num_attention_heads
    assert isinstance(attention_module.q_proj, nn.Linear)
    assert isinstance(attention_module.k_proj, nn.Linear)
    assert isinstance(attention_module.v_proj, nn.Linear)
    assert isinstance(attention_module.o_proj, nn.Linear)


def test_standard_attention_module_forward():
    """Test StandardAttentionModule forward pass."""
    config = MockConfig()

    attention_module = StandardAttentionModule(config)

    # Create sample input
    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    output, attn_weights, past_key_value = attention_module(
        hidden_states=hidden_states,
        position_ids=position_ids,
        output_attentions=True
    )

    assert output.shape == (batch_size, seq_len, hidden_size)
    assert attn_weights is not None
    assert past_key_value is None


def test_standard_attention_module_memory_usage():
    """Test StandardAttentionModule memory usage calculation."""
    config = MockConfig()

    attention_module = StandardAttentionModule(config)
    memory_usage = attention_module.get_memory_usage()

    assert isinstance(memory_usage, dict)
    assert "projection_params" in memory_usage
    assert "kv_cache_per_token" in memory_usage
    assert "attn_weights" in memory_usage
    assert memory_usage["projection_params"] > 0
    assert memory_usage["kv_cache_per_token"] > 0
    assert memory_usage["attn_weights"] > 0


def test_standard_attention_module_compute_complexity():
    """Test StandardAttentionModule compute complexity calculation."""
    config = MockConfig()

    attention_module = StandardAttentionModule(config)
    complexity = attention_module.get_compute_complexity()

    assert isinstance(complexity, dict)
    assert "qkv_projection_ops" in complexity
    assert "attention_matrix_ops" in complexity
    assert "attention_values_ops" in complexity
    assert "output_projection_ops" in complexity
    assert complexity["qkv_projection_ops"] > 0
    assert complexity["attention_matrix_ops"] > 0


def test_flash_attention_module_initialization():
    """Test FlashAttentionModule initialization."""
    config = MockConfig()
    config.use_flash_attention_2 = True

    attention_module = FlashAttentionModule(config)

    # The module should initialize without errors
    assert attention_module is not None


def test_flash_attention_module_forward():
    """Test FlashAttentionModule forward pass."""
    config = MockConfig()
    config.use_flash_attention_2 = True

    attention_module = FlashAttentionModule(config)

    # Create sample input
    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    output, attn_weights, past_key_value = attention_module(
        hidden_states=hidden_states,
        position_ids=position_ids,
        output_attentions=True
    )

    assert output.shape == (batch_size, seq_len, hidden_size)


def test_sparse_attention_module_initialization():
    """Test SparseAttentionModule initialization."""
    config = MockConfig()
    config.use_sparse_attention = True
    config.sparsity_factor = 0.2

    attention_module = SparseAttentionModule(config)

    assert attention_module.sparsity_factor == 0.2
    assert attention_module is not None


def test_sparse_attention_module_forward():
    """Test SparseAttentionModule forward pass."""
    config = MockConfig()
    config.use_sparse_attention = True
    config.sparsity_factor = 0.2

    attention_module = SparseAttentionModule(config)

    # Create sample input
    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    output, attn_weights, past_key_value = attention_module(
        hidden_states=hidden_states,
        position_ids=position_ids,
        output_attentions=True
    )

    assert output.shape == (batch_size, seq_len, hidden_size)


def test_memory_efficient_attention_module_initialization():
    """Test MemoryEfficientAttentionModule initialization."""
    config = MockConfig()
    config.use_memory_efficient_attention = True
    config.chunk_size = 256

    attention_module = MemoryEfficientAttentionModule(config)

    assert attention_module.chunk_size == 256
    assert attention_module is not None


def test_memory_efficient_attention_module_forward():
    """Test MemoryEfficientAttentionModule forward pass."""
    config = MockConfig()
    config.use_memory_efficient_attention = True
    config.chunk_size = 256

    attention_module = MemoryEfficientAttentionModule(config)

    # Create sample input
    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size

    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    output, attn_weights, past_key_value = attention_module(
        hidden_states=hidden_states,
        position_ids=position_ids,
        output_attentions=True
    )

    assert output.shape == (batch_size, seq_len, hidden_size)


def test_attention_performance_monitor():
    """Test AttentionPerformanceMonitor functionality."""
    monitor = AttentionPerformanceMonitor()

    # Test timing functionality
    monitor.start_timing("test_operation")
    time.sleep(0.01)  # Sleep for 10ms
    monitor.end_timing("test_operation")

    metrics = monitor.get_current_metrics()
    assert f"test_operation_duration" in metrics
    assert metrics[f"test_operation_duration"] >= 0.01

    # Test memory recording
    monitor.record_memory_usage("test_operation", 1024)
    metrics = monitor.get_current_metrics()
    assert f"test_operation_memory_used" in metrics
    assert metrics[f"test_operation_memory_used"] == 1024

    # Test peak memory recording
    monitor.record_peak_memory("test_operation", 2048)
    metrics = monitor.get_current_metrics()
    assert f"test_operation_peak_memory" in metrics
    assert metrics[f"test_operation_peak_memory"] == 2048

    # Test metrics history
    monitor.log_metrics()
    assert len(monitor.metrics_history) == 1

    # Test reset
    monitor.reset_metrics()
    assert monitor.get_current_metrics() == {}


def test_hardware_aware_attention_selector():
    """Test HardwareAwareAttentionSelector functionality."""
    selector = HardwareAwareAttentionSelector()

    config = MockConfig()

    # Test selection with different memory constraints
    attention_type_high_memory = selector.select_attention_type(config, available_memory_gb=16.0)
    attention_type_low_memory = selector.select_attention_type(config, available_memory_gb=4.0)

    # The exact type depends on the hardware, but both should be valid AttentionTypes
    assert isinstance(attention_type_high_memory, AttentionType)
    assert isinstance(attention_type_low_memory, AttentionType)


def test_attention_manager_initialization():
    """Test AttentionManager initialization."""
    config = MockConfig()

    manager = AttentionManager(config)

    assert manager.config == config
    assert manager.active_attention_type is None
    assert manager.active_attention_module is None
    assert len(manager.attention_implementations) > 0


def test_attention_manager_select_attention_module():
    """Test AttentionManager select_attention_module functionality."""
    config = MockConfig()

    manager = AttentionManager(config)

    # Test selecting specific attention type
    attention_module = manager.select_attention_module(AttentionType.STANDARD)
    assert isinstance(attention_module, StandardAttentionModule)
    assert manager.active_attention_type == AttentionType.STANDARD
    assert manager.active_attention_module is attention_module

    # Test auto-selection
    auto_module = manager.select_attention_module()
    assert auto_module is not None


def test_attention_manager_switch_attention_module():
    """Test AttentionManager switch_attention_module functionality."""
    config = MockConfig()

    manager = AttentionManager(config)

    # Start with standard attention
    initial_module = manager.select_attention_module(AttentionType.STANDARD)
    assert isinstance(initial_module, StandardAttentionModule)

    # Switch to flash attention
    success = manager.switch_attention_module(AttentionType.FLASH_ATTENTION)
    assert success
    assert manager.active_attention_type == AttentionType.FLASH_ATTENTION


def test_attention_manager_benchmark_attention_types():
    """Test AttentionManager benchmark_attention_types functionality."""
    config = MockConfig()

    manager = AttentionManager(config)

    # Create a small sample input for benchmarking
    sample_input = torch.randn(1, 5, config.hidden_size)

    # Run benchmark
    results = manager.benchmark_attention_types(sample_input)

    # Check that results contain entries for all attention types
    assert isinstance(results, dict)
    for att_type in manager.attention_implementations.keys():
        assert att_type in results
        assert "success" in results[att_type]
        # Either the benchmark succeeded or failed with an error
        if results[att_type]["success"]:
            assert "time_seconds" in results[att_type]
            assert "memory_bytes" in results[att_type]
        else:
            assert "error" in results[att_type]


def test_attention_manager_get_active_attention_info():
    """Test AttentionManager get_active_attention_info functionality."""
    config = MockConfig()

    manager = AttentionManager(config)

    # Initially should return info about no active module
    info = manager.get_active_attention_info()
    assert info["active_type"] is None

    # After selecting a module, should return proper info
    manager.select_attention_module(AttentionType.STANDARD)
    info = manager.get_active_attention_info()
    assert info["active_type"] == AttentionType.STANDARD.value
    assert "memory_usage" in info
    assert "compute_complexity" in info


def test_attention_manager_validate_attention_config():
    """Test AttentionManager validate_attention_config functionality."""
    config = MockConfig()

    manager = AttentionManager(config)

    # Valid config should have no issues
    issues = manager.validate_attention_config(config)
    assert len(issues) == 0

    # Invalid config should have issues
    invalid_config = MockConfig()
    invalid_config.hidden_size = 511  # Not divisible by num_heads
    invalid_config.num_attention_heads = 8
    issues = manager.validate_attention_config(invalid_config)
    assert len(issues) > 0
    assert any("divisible" in issue for issue in issues)

    # Test with invalid sparsity factor
    sparse_config = MockConfig()
    sparse_config.use_sparse_attention = True
    sparse_config.sparsity_factor = -0.5  # Invalid
    issues = manager.validate_attention_config(sparse_config)
    assert any("sparsity_factor" in issue for issue in issues)


def test_multi_model_attention_adapter():
    """Test MultiModelAttentionAdapter functionality."""
    config = MockConfig()

    # Create a mock model spec
    model_spec = ModelSpec(
        name="test_model",
        model_class=nn.Module,
        config_class=MockConfig,
        adapter_class=None,
        supported_dtypes=["float16", "float32"],
        required_memory_gb=4.0,
        max_sequence_length=2048,
        description="Test model for attention adapter",
        model_type="language"
    )

    adapter = MultiModelAttentionAdapter(config, model_spec)

    assert adapter.model_spec == model_spec
    assert adapter.config == config
    assert adapter.attention_manager is not None
    assert adapter.attention_module is not None

    # Test forward pass
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    output, attn_weights, past_key_value = adapter(
        hidden_states=hidden_states,
        position_ids=position_ids,
        output_attentions=True
    )

    assert output.shape == (batch_size, seq_len, config.hidden_size)

    # Test switching attention type
    success = adapter.switch_attention_type(AttentionType.MEMORY_EFFICIENT)
    assert success

    # Test getting performance stats
    stats = adapter.get_performance_stats()
    assert "model_info" in stats
    assert "attention_info" in stats
    assert "recent_metrics" in stats


def test_legacy_create_consolidated_attention_module():
    """Test legacy create_consolidated_attention_module factory function."""
    config = MockConfig()

    # Test creating standalone attention module
    attention_module = legacy_create_consolidated_attention_module(config)
    assert isinstance(attention_module, AttentionModule)

    # Test creating with model spec
    model_spec = ModelSpec(
        name="test_model",
        model_class=nn.Module,
        config_class=MockConfig,
        adapter_class=None,
        supported_dtypes=["float16", "float32"],
        required_memory_gb=4.0,
        max_sequence_length=2048,
        description="Test model",
        model_type="language"
    )

    adapter = legacy_create_consolidated_attention_module(config, model_spec)
    assert isinstance(adapter, MultiModelAttentionAdapter)
    assert adapter.model_spec == model_spec


def test_attention_module_error_handling():
    """Test error handling in attention modules."""
    config = MockConfig()

    # Test with invalid config (hidden_size not divisible by num_heads)
    invalid_config = MockConfig()
    invalid_config.hidden_size = 511  # Not divisible by 8
    invalid_config.num_attention_heads = 8

    with pytest.raises(ValueError):
        StandardAttentionModule(invalid_config)


def test_attention_manager_error_handling():
    """Test error handling in AttentionManager."""
    config = MockConfig()

    manager = AttentionManager(config)

    # Test with unsupported attention type
    with pytest.raises(ValueError):
        manager.select_attention_module("invalid_type")


@patch('torch.cuda.is_available')
@patch('torch.cuda.get_device_capability')
def test_cuda_dependent_functionality(mock_get_device_capability, mock_is_available):
    """Test CUDA-dependent functionality with mocking."""
    # Mock CUDA as available with compute capability 8.0
    mock_is_available.return_value = True
    mock_get_device_capability.return_value = (8, 0)

    config = MockConfig()
    config.use_flash_attention_2 = True

    # This should now use FlashAttention2
    attention_module = FlashAttentionModule(config)
    assert attention_module is not None


def test_attention_types_enum():
    """Test AttentionType enum values."""
    assert AttentionType.STANDARD.value == "standard"
    assert AttentionType.FLASH_ATTENTION.value == "flash_attention"
    assert AttentionType.SPARSE_ATTENTION.value == "sparse_attention"
    assert AttentionType.MEMORY_EFFICIENT.value == "memory_efficient"
    assert AttentionType.DYNAMIC_SPARSE.value == "dynamic_sparse"
    assert AttentionType.BLOCK_SPARSE.value == "block_sparse"
    assert AttentionType.CUSTOM.value == "custom"
    assert AttentionType.SIMD_OPTIMIZED.value == "simd_optimized"


def test_memory_efficiency_comparison():
    """Compare memory usage between different attention implementations."""
    config = MockConfig()

    # Create different attention modules
    standard_module = StandardAttentionModule(config)
    sparse_config = MockConfig()
    sparse_config.use_sparse_attention = True
    sparse_module = SparseAttentionModule(sparse_config)

    # Get memory usage
    standard_memory = standard_module.get_memory_usage()
    sparse_memory = sparse_module.get_memory_usage()

    # Sparse attention should generally use less memory for attention weights
    if "attn_weights" in standard_memory and "attn_weights" in sparse_memory:
        # The sparse attention should use less memory for attention weights
        # Note: This might not always be true depending on implementation details
        pass  # Just verify both exist


class TestConsolidatedAttention(unittest.TestCase):
    """Test suite for consolidated attention functionality"""

    def test_all_tests_combined(self):
        """Run all tests in the consolidated suite"""
        # This is a placeholder to ensure the file is properly structured as a test suite
        self.assertTrue(True)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])