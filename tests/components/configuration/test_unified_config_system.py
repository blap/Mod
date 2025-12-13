"""Comprehensive test for the unified configuration system."""

import unittest
import tempfile
import os
from pathlib import Path
import json
import yaml
from unittest.mock import patch
import torch
from copy import deepcopy


from dataclasses import dataclass
from src.qwen3_vl.core.config import (
    BaseConfig, MemoryConfig, CPUConfig, GPUConfig, PowerManagementConfig,
    OptimizationConfig, Qwen3VLConfig
)


class TestBaseConfig(unittest.TestCase):
    """Test the base configuration class."""
    
    def test_to_dict_from_dict(self):
        """Test serialization and deserialization of base config."""
        @dataclass
        class TestConfig(BaseConfig):
            test_field: int = 42
            another_field: str = "test"
        
        config = TestConfig(test_field=100, another_field="hello")
        config_dict = config.to_dict()
        
        # Verify the dictionary contains the expected values
        self.assertEqual(config_dict['test_field'], 100)
        self.assertEqual(config_dict['another_field'], "hello")
        
        # Deserialize back to config
        restored_config = TestConfig.from_dict(config_dict)
        self.assertEqual(restored_config.test_field, 100)
        self.assertEqual(restored_config.another_field, "hello")
    
    def test_save_load_file(self):
        """Test saving and loading config from file."""
        @dataclass
        class TestConfig(BaseConfig):
            field1: int = 42
            field2: str = "test"
        
        config = TestConfig(field1=100, field2="hello")
        
        # Test JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_json = f.name
        
        try:
            config.save_to_file(temp_json)
            loaded_config = TestConfig.from_file(temp_json)
            self.assertEqual(loaded_config.field1, 100)
            self.assertEqual(loaded_config.field2, "hello")
        finally:
            os.unlink(temp_json)
        
        # Test YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_yaml = f.name
        
        try:
            config.save_to_file(temp_yaml)
            loaded_config = TestConfig.from_file(temp_yaml)
            self.assertEqual(loaded_config.field1, 100)
            self.assertEqual(loaded_config.field2, "hello")
        finally:
            os.unlink(temp_yaml)


class TestMemoryConfig(unittest.TestCase):
    """Test memory configuration."""
    
    def test_memory_config_initialization(self):
        """Test memory configuration initialization."""
        config = MemoryConfig()
        
        self.assertEqual(config.memory_pool_size, 2 * 1024 * 1024 * 1024)
        self.assertEqual(config.memory_pool_dtype, "float16")
        self.assertTrue(config.enable_memory_tiering)
        self.assertEqual(config.compression_level, "medium")
        self.assertTrue(config.enable_memory_swapping)
        self.assertTrue(config.enable_memory_defragmentation)
    
    def test_memory_config_custom_values(self):
        """Test memory configuration with custom values."""
        config = MemoryConfig(
            memory_pool_size=1024 * 1024 * 1024,  # 1GB
            memory_pool_dtype="float32",
            enable_memory_tiering=False,
            gpu_memory_size=8 * 1024 * 1024 * 1024,  # 8GB
            compression_level="high",
            swap_threshold=0.9
        )
        
        self.assertEqual(config.memory_pool_size, 1024 * 1024 * 1024)
        self.assertEqual(config.memory_pool_dtype, "float32")
        self.assertFalse(config.enable_memory_tiering)
        self.assertEqual(config.gpu_memory_size, 8 * 1024 * 1024 * 1024)
        self.assertEqual(config.compression_level, "high")
        self.assertEqual(config.swap_threshold, 0.9)


class TestCPUConfig(unittest.TestCase):
    """Test CPU configuration."""
    
    def test_cpu_config_initialization(self):
        """Test CPU configuration initialization."""
        config = CPUConfig()
        
        self.assertEqual(config.num_threads, 4)
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(config.l1_cache_size, 32 * 1024)
        self.assertEqual(config.l2_cache_size, 256 * 1024)
        self.assertEqual(config.l3_cache_size, 6 * 1024 * 1024)
        self.assertTrue(config.enable_cpu_optimizations)
        self.assertTrue(config.use_hyperthreading)
        self.assertTrue(config.enable_simd_optimizations)
        self.assertEqual(config.simd_instruction_set, "avx2")
    
    def test_cpu_config_custom_values(self):
        """Test CPU configuration with custom values."""
        config = CPUConfig(
            num_threads=8,
            num_workers=6,
            l3_cache_size=12 * 1024 * 1024,  # 12MB
            simd_instruction_set="sse",
            enable_cpu_optimizations=False,
            memory_threshold=0.7
        )
        
        self.assertEqual(config.num_threads, 8)
        self.assertEqual(config.num_workers, 6)
        self.assertEqual(config.l3_cache_size, 12 * 1024 * 1024)
        self.assertEqual(config.simd_instruction_set, "sse")
        self.assertFalse(config.enable_cpu_optimizations)
        self.assertEqual(config.memory_threshold, 0.7)


class TestGPUConfig(unittest.TestCase):
    """Test GPU configuration."""
    
    def test_gpu_config_initialization(self):
        """Test GPU configuration initialization."""
        config = GPUConfig()
        
        self.assertEqual(config.gpu_compute_capability, (6, 1))
        self.assertEqual(config.max_threads_per_block, 1024)
        self.assertEqual(config.shared_memory_per_block, 48 * 1024)
        self.assertEqual(config.memory_bandwidth_gbps, 320.0)
        self.assertTrue(config.enable_gpu_optimizations)
        self.assertTrue(config.use_tensor_cores)
        self.assertTrue(config.use_mixed_precision)
        self.assertEqual(config.attention_implementation, "flash_attention_2")
        self.assertEqual(config.kv_cache_strategy, "hybrid")
    
    def test_gpu_config_custom_values(self):
        """Test GPU configuration with custom values."""
        config = GPUConfig(
            gpu_compute_capability=(7, 5),
            max_threads_per_block=2048,
            shared_memory_per_block=96 * 1024,
            memory_bandwidth_gbps=900.0,
            use_tensor_cores=False,
            attention_implementation="optimized",
            kv_cache_strategy="low_rank"
        )
        
        self.assertEqual(config.gpu_compute_capability, (7, 5))
        self.assertEqual(config.max_threads_per_block, 2048)
        self.assertEqual(config.shared_memory_per_block, 96 * 1024)
        self.assertEqual(config.memory_bandwidth_gbps, 900.0)
        self.assertFalse(config.use_tensor_cores)
        self.assertEqual(config.attention_implementation, "optimized")
        self.assertEqual(config.kv_cache_strategy, "low_rank")


class TestPowerManagementConfig(unittest.TestCase):
    """Test power management configuration."""
    
    def test_power_config_initialization(self):
        """Test power management configuration initialization."""
        config = PowerManagementConfig()
        
        self.assertTrue(config.enable_power_optimization)
        self.assertEqual(config.power_constraint, 0.8)
        self.assertEqual(config.thermal_constraint, 75.0)
        self.assertEqual(config.performance_target, 0.9)
        self.assertEqual(config.adaptation_frequency, 1.0)
        self.assertTrue(config.enable_dynamic_power_scaling)
    
    def test_power_config_custom_values(self):
        """Test power management configuration with custom values."""
        config = PowerManagementConfig(
            enable_power_optimization=False,
            power_constraint=0.9,
            thermal_constraint=80.0,
            performance_target=0.95,
            adaptation_frequency=2.0,
            enable_dynamic_power_scaling=False
        )
        
        self.assertFalse(config.enable_power_optimization)
        self.assertEqual(config.power_constraint, 0.9)
        self.assertEqual(config.thermal_constraint, 80.0)
        self.assertEqual(config.performance_target, 0.95)
        self.assertEqual(config.adaptation_frequency, 2.0)
        self.assertFalse(config.enable_dynamic_power_scaling)


class TestOptimizationConfig(unittest.TestCase):
    """Test optimization configuration."""
    
    def test_optimization_config_initialization(self):
        """Test optimization configuration initialization."""
        config = OptimizationConfig()
        
        self.assertTrue(config.use_memory_pooling)
        self.assertTrue(config.use_hierarchical_memory_compression)
        self.assertTrue(config.use_memory_efficient_attention)
        self.assertTrue(config.use_sparsity)
        self.assertEqual(config.sparsity_ratio, 0.5)
        self.assertTrue(config.use_dynamic_sparse_attention)
        self.assertTrue(config.use_adaptive_precision)
        self.assertTrue(config.use_moe)
        self.assertEqual(config.moe_num_experts, 4)
        self.assertEqual(config.moe_top_k, 2)
        self.assertTrue(config.use_flash_attention_2)
        self.assertTrue(config.use_adaptive_depth)
        self.assertTrue(config.use_gradient_checkpointing)
        self.assertTrue(config.use_context_adaptive_positional_encoding)
        self.assertTrue(config.use_conditional_feature_extraction)
        self.assertTrue(config.use_cross_modal_compression)
        self.assertTrue(config.use_cross_layer_memory_sharing)
        self.assertTrue(config.use_hierarchical_vision)
        self.assertTrue(config.use_learned_activation_routing)
        self.assertTrue(config.use_adaptive_batch_processing)
        self.assertTrue(config.use_adaptive_sequence_packing)
        self.assertTrue(config.use_memory_efficient_grad_accumulation)
        self.assertTrue(config.use_faster_rotary_embeddings)
        self.assertFalse(config.use_distributed_pipeline_parallelism)
        self.assertTrue(config.use_hardware_specific_kernels)
        self.assertEqual(config.performance_improvement_threshold, 0.05)
        self.assertEqual(config.accuracy_preservation_threshold, 0.95)
    
    def test_optimization_config_custom_values(self):
        """Test optimization configuration with custom values."""
        config = OptimizationConfig(
            use_memory_pooling=False,
            sparsity_ratio=0.3,
            moe_num_experts=8,
            moe_top_k=3,
            performance_improvement_threshold=0.1,
            accuracy_preservation_threshold=0.98
        )
        
        self.assertFalse(config.use_memory_pooling)
        self.assertEqual(config.sparsity_ratio, 0.3)
        self.assertEqual(config.moe_num_experts, 8)
        self.assertEqual(config.moe_top_k, 3)
        self.assertEqual(config.performance_improvement_threshold, 0.1)
        self.assertEqual(config.accuracy_preservation_threshold, 0.98)


class TestQwen3VLConfig(unittest.TestCase):
    """Test Qwen3-VL configuration."""
    
    def test_qwen3_vl_config_initialization(self):
        """Test Qwen3-VL configuration initialization."""
        config = Qwen3VLConfig()
        
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.intermediate_size, 11008)
        self.assertEqual(config.vision_num_hidden_layers, 24)
        self.assertEqual(config.vision_num_attention_heads, 16)
        self.assertEqual(config.vision_hidden_size, 1152)
        self.assertEqual(config.hardware_target, "intel_i5_10210u_nvidia_sm61_nvme")
        self.assertEqual(config.target_hardware, "nvidia_sm61")
        self.assertEqual(config.compute_units, 4)
        self.assertEqual(config.memory_gb, 8.0)
        self.assertEqual(config.torch_dtype, "float16")
        self.assertEqual(config.optimization_level, "balanced")
        
        # Check that sub-configs were properly initialized
        self.assertIsNotNone(config.memory_config)
        self.assertIsNotNone(config.cpu_config)
        self.assertIsNotNone(config.gpu_config)
        self.assertIsNotNone(config.power_config)
        self.assertIsNotNone(config.optimization_config)
    
    def test_qwen3_vl_config_post_init_validation(self):
        """Test post-initialization validation."""
        # Test valid config
        config = Qwen3VLConfig()
        # No exception means validation passed
        
        # Test invalid config - hidden_size not divisible by num_attention_heads
        with self.assertRaises(ValueError):
            Qwen3VLConfig(
                hidden_size=512,
                num_attention_heads=7  # Not a divisor of 512
            )
        
        # Test invalid config - vision_hidden_size not divisible by vision_num_attention_heads
        with self.assertRaises(ValueError):
            Qwen3VLConfig(
                vision_hidden_size=512,
                vision_num_attention_heads=7  # Not a divisor of 512
            )
    
    def test_qwen3_vl_config_with_custom_subconfigs(self):
        """Test Qwen3-VL config with custom sub-configurations."""
        memory_config = MemoryConfig(
            memory_pool_size=1024 * 1024 * 1024,  # 1GB
            enable_memory_tiering=False
        )
        
        cpu_config = CPUConfig(
            num_threads=8,
            enable_cpu_optimizations=False
        )
        
        gpu_config = GPUConfig(
            gpu_compute_capability=(7, 0),
            use_tensor_cores=False
        )
        
        power_config = PowerManagementConfig(
            enable_power_optimization=False,
            thermal_constraint=80.0
        )
        
        optimization_config = OptimizationConfig(
            use_sparsity=False,
            use_moe=False
        )
        
        config = Qwen3VLConfig(
            memory_config=memory_config,
            cpu_config=cpu_config,
            gpu_config=gpu_config,
            power_config=power_config,
            optimization_config=optimization_config
        )
        
        self.assertEqual(config.memory_config.memory_pool_size, 1024 * 1024 * 1024)
        self.assertFalse(config.memory_config.enable_memory_tiering)
        self.assertEqual(config.cpu_config.num_threads, 8)
        self.assertFalse(config.cpu_config.enable_cpu_optimizations)
        self.assertEqual(config.gpu_config.gpu_compute_capability, (7, 0))
        self.assertFalse(config.gpu_config.use_tensor_cores)
        self.assertFalse(config.power_config.enable_power_optimization)
        self.assertEqual(config.power_config.thermal_constraint, 80.0)
        self.assertFalse(config.optimization_config.use_sparsity)
        self.assertFalse(config.optimization_config.use_moe)


class TestFromPretrained(unittest.TestCase):
    """Test from_pretrained functionality."""
    
    def test_from_pretrained_local_file(self):
        """Test loading config from a local file."""
        # Create a temporary config file
        temp_config = Qwen3VLConfig(
            num_hidden_layers=16,
            num_attention_heads=16,
            hidden_size=2048
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_config.save_to_file(f.name)
            temp_file = f.name
        
        try:
            # Load from file using from_pretrained
            loaded_config = Qwen3VLConfig.from_pretrained(temp_file)
            
            self.assertEqual(loaded_config.num_hidden_layers, 16)
            self.assertEqual(loaded_config.num_attention_heads, 16)
            self.assertEqual(loaded_config.hidden_size, 2048)
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)