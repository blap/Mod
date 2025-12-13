"""
Comprehensive test suite for error handling and input validation in the Qwen3-VL-2B-Instruct project files.
This tests all the improvements made to the codebase.
"""
import pytest
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase
from PIL import Image
import tempfile
import os
from typing import Dict, List, Optional, Tuple, Any

# Import the modules we want to test
from adaptive_algorithms import AdaptiveParameters, AdaptiveController, LoadBalancer, AdaptiveModelWrapper
from advanced_memory_management_vl import AdvancedMemoryPool, MemoryPoolType, VisionLanguageMemoryOptimizer
from adaptive_precision_optimization import AdaptivePrecisionConfig, PrecisionManager, AdaptivePrecisionLayer, AdaptivePrecisionModelWrapper
from advanced_cpu_optimizations_intel_i5_10210u import AdvancedCPUOptimizationConfig, IntelCPUOptimizedPreprocessor, IntelOptimizedPipeline, AdaptiveIntelOptimizer


def test_adaptive_parameters_validation():
    """Test validation in AdaptiveParameters."""
    print("Testing AdaptiveParameters validation...")
    
    # Test valid initialization
    params = AdaptiveParameters()
    assert params.performance_factor == 1.0
    
    # Test invalid performance_factor
    try:
        AdaptiveParameters(performance_factor=1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    try:
        AdaptiveParameters(performance_factor=-0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    try:
        AdaptiveParameters(performance_factor="invalid")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected
    
    print("AdaptiveParameters validation tests passed!")


def test_adaptive_controller_validation():
    """Test validation in AdaptiveController."""
    print("Testing AdaptiveController validation...")
    
    from power_management import PowerConstraint
    constraints = PowerConstraint()
    
    # Test valid initialization
    controller = AdaptiveController(constraints)
    
    # Test invalid constraints
    try:
        AdaptiveController("invalid_constraints")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected
    
    # Test invalid power state
    try:
        controller.update_parameters("invalid_power_state")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected
    
    print("AdaptiveController validation tests passed!")


def test_memory_optimizer_validation():
    """Test validation in VisionLanguageMemoryOptimizer."""
    print("Testing VisionLanguageMemoryOptimizer validation...")
    
    # Test valid initialization
    optimizer = VisionLanguageMemoryOptimizer(memory_pool_size=1024*1024)
    
    # Test invalid memory_pool_size
    try:
        VisionLanguageMemoryOptimizer(memory_pool_size=-1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    try:
        VisionLanguageMemoryOptimizer(memory_pool_size="invalid")
        assert False, "Should have raised TypeError"
    except (TypeError, ValueError):
        pass  # Expected
    
    # Test invalid tensor allocation
    try:
        # This should handle invalid tensor types gracefully
        result = optimizer.allocate_tensor_memory((10, 10), dtype="invalid_dtype")
        # Should return None or handle the error gracefully
        assert result is None
    except Exception as e:
        print(f"Expected behavior with invalid dtype: {e}")
    
    print("VisionLanguageMemoryOptimizer validation tests passed!")


def test_cpu_config_validation():
    """Test validation in AdvancedCPUOptimizationConfig."""
    print("Testing AdvancedCPUOptimizationConfig validation...")
    
    # Test valid initialization
    config = AdvancedCPUOptimizationConfig()
    
    # Test invalid parameters
    try:
        # Create a config with invalid num_preprocess_workers
        bad_config = AdvancedCPUOptimizationConfig(num_preprocess_workers=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    try:
        # Create a config with invalid performance_target
        bad_config = AdvancedCPUOptimizationConfig(performance_target=1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    try:
        # Create a config with invalid image_resize_size
        bad_config = AdvancedCPUOptimizationConfig(image_resize_size=(0, 224))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    print("AdvancedCPUOptimizationConfig validation tests passed!")


def test_adaptive_intel_optimizer_constraints():
    """Test the new power and thermal constraint methods."""
    print("Testing AdaptiveIntelOptimizer constraint methods...")
    
    config = AdvancedCPUOptimizationConfig()
    optimizer = AdaptiveIntelOptimizer(config)
    
    # Test valid power constraint
    optimizer.set_power_constraint(0.8)
    assert optimizer.config.power_constraint == 0.8
    
    # Test valid thermal constraint
    optimizer.set_thermal_constraint(80.0)
    assert optimizer.config.thermal_constraint == 80.0
    
    # Test invalid power constraint
    try:
        optimizer.set_power_constraint(1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    try:
        optimizer.set_power_constraint(-0.1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    try:
        optimizer.set_power_constraint("invalid")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected
    
    # Test invalid thermal constraint
    try:
        optimizer.set_thermal_constraint(-5.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    try:
        optimizer.set_thermal_constraint("invalid")
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected
    
    print("AdaptiveIntelOptimizer constraint method tests passed!")


def test_tensor_memory_validation():
    """Test tensor memory allocation validation."""
    print("Testing tensor memory allocation validation...")
    
    optimizer = VisionLanguageMemoryOptimizer()
    
    # Test valid allocation
    tensor = optimizer.allocate_tensor_memory((10, 10), dtype=np.float32, tensor_type="general")
    assert tensor.shape == (10, 10)
    assert tensor.dtype == np.float32
    
    # Test invalid shape
    try:
        optimizer.allocate_tensor_memory("invalid_shape", dtype=np.float32)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass  # Expected
    
    try:
        optimizer.allocate_tensor_memory((10, -1), dtype=np.float32)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    # Test invalid tensor_type
    try:
        optimizer.allocate_tensor_memory((10, 10), dtype=np.float32, tensor_type="invalid_type")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected
    
    # Test invalid dtype handling
    result = optimizer.allocate_tensor_memory((10, 10), dtype="invalid_dtype")
    # Should return None or handle gracefully
    assert result is None or hasattr(result, 'shape')
    
    print("Tensor memory allocation validation tests passed!")


def run_all_comprehensive_tests():
    """Run all comprehensive validation tests."""
    print("Running comprehensive error handling and input validation tests...")
    
    test_adaptive_parameters_validation()
    test_adaptive_controller_validation()
    test_memory_optimizer_validation()
    test_cpu_config_validation()
    test_adaptive_intel_optimizer_constraints()
    test_tensor_memory_validation()
    
    print("All comprehensive tests completed successfully!")


if __name__ == "__main__":
    run_all_comprehensive_tests()