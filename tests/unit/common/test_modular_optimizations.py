"""
Tests for the Modular Optimization System in Inference-PIO

This module contains comprehensive tests for the modular optimization system
that manages activation/deactivation of optimizations across all models.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from src.inference_pio.common.optimization_manager import (
    ModularOptimizationManager,
    OptimizationConfig,
    OptimizationType,
    get_optimization_manager,
    register_default_optimizations
)
from src.inference_pio.common.optimization_config import (
    ModelFamily,
    ModelOptimizationConfig,
    GlobalOptimizationProfile,
    get_config_manager,
    create_balanced_profile,
    create_performance_profile,
    create_memory_efficient_profile,
    create_glm_optimization_config,
    create_qwen_optimization_config
)
from src.inference_pio.common.optimization_integration import (
    apply_model_family_optimizations,
    apply_optimizations_by_config,
    get_model_optimization_status,
    update_model_optimization,
    create_optimization_pipeline,
    get_supported_optimizations,
    reset_model_optimizations,
    apply_glm_optimizations,
    apply_qwen_optimizations
)

class MockModel(nn.Module):
    """Mock model for testing optimization systems."""
    
    def __init__(self):
        super().__init__()
        linear = nn.Linear(10, 5)
        conv = nn.Conv2d(3, 16, 3)
        
    def forward(self, x):
        return linear(x)

# TestOptimizationManager

    """Test cases for the ModularOptimizationManager."""
    
    def setup_helper():
        """Set up test fixtures."""
        manager = get_optimization_manager()
        register_default_optimizations()
        
    def register_and_create_optimization(self)():
        """Test registering and creating optimizations."""
        # Test that default optimizations are registered
        available_opts = manager.get_available_optimizations()
        assert_in("flash_attention", available_opts)
        assert_in("sparse_attention", available_opts)
        assert_in("disk_offloading", available_opts)
        
    def configure_optimization(self)():
        """Test configuring optimizations."""
        config = OptimizationConfig(
            name="test_opt",
            enabled=True,
            optimization_type=OptimizationType.COMPUTE,
            parameters={"test_param": "value"}
        )
        manager.configure_optimization("test_opt", config)
        
        assert_in("test_opt", manager.optimization_configs)
        assert_true(manager.optimization_configs["test_opt"].enabled)
        
    def apply_optimizations(self)():
        """Test applying optimizations to a model."""
        model = MockModel()
        
        # Configure a mock optimization
        config = OptimizationConfig(
            name="flash_attention",
            enabled=True,
            optimization_type=OptimizationType.ATTENTION
        )
        manager.configure_optimization("flash_attention", config)
        
        # Apply optimization
        optimized_model = manager.apply_optimizations(model, ["flash_attention"])
        
        # Check that optimization was applied
        applied_opts = manager.get_model_optimizations(optimized_model)
        assert_in("flash_attention", applied_opts)
        
    def remove_optimizations(self)():
        """Test removing optimizations from a model."""
        model = MockModel()
        
        # Configure and apply an optimization
        config = OptimizationConfig(
            name="flash_attention",
            enabled=True,
            optimization_type=OptimizationType.ATTENTION
        )
        manager.configure_optimization("flash_attention", config)
        optimized_model = manager.apply_optimizations(model, ["flash_attention"])
        
        # Verify optimization was applied
        applied_opts = manager.get_model_optimizations(optimized_model)
        assert_in("flash_attention", applied_opts)
        
        # Remove the optimization
        restored_model = manager.remove_optimizations(optimized_model, ["flash_attention"])
        
        # Verify optimization was removed
        applied_opts_after = manager.get_model_optimizations(restored_model)
        assert_not_in("flash_attention", applied_opts_after)
        
    def get_optimization_status(self)():
        """Test getting optimization status."""
        model = MockModel()
        
        # Configure an optimization
        config = OptimizationConfig(
            name="flash_attention",
            enabled=True,
            optimization_type=OptimizationType.ATTENTION
        )
        manager.configure_optimization("flash_attention", config)
        
        # Get status before applying
        status = manager.get_optimization_status("flash_attention")
        assert_equal(status["name"], "flash_attention")
        assert_true(status["configured"])
        assertTrue(status["enabled"])
        assert_false(status["active"])
        
        # Apply optimization
        manager.apply_optimizations(model)
        
        # Get status after applying
        status = manager.get_optimization_status("flash_attention")
        assert_true(status["active"])

# TestOptimizationConfig

    """Test cases for optimization configuration."""
    
    def setup_helper():
        """Set up test fixtures."""
        config_manager = get_config_manager()
        
    def create_model_optimization_config(self)():
        """Test creating model-specific optimization configurations."""
        # Test GLM config
        glm_config = create_glm_optimization_config()
        assert_equal(glm_config.model_family)
        assert_is_instance(glm_config.optimizations)
        assert_greater(len(glm_config.optimizations), 0)
        
        # Test Qwen config
        qwen_config = create_qwen_optimization_config()
        assert_equal(qwen_config.model_family, ModelFamily.QWEN)
        assert_is_instance(qwen_config.optimizations, list)
        assert_greater(len(qwen_config.optimizations), 0)
        
    def create_global_optimization_profiles(self)():
        """Test creating global optimization profiles."""
        # Test balanced profile
        balanced_profile = create_balanced_profile()
        assert_equal(balanced_profile.name, "balanced")
        assert_in("flash_attention", balanced_profile.default_settings)
        
        # Test performance profile
        perf_profile = create_performance_profile()
        assert_equal(perf_profile.name, "performance")
        assertGreaterEqual(perf_profile.performance_targets["latency_reduction"], 0.2)
        
        # Test memory efficient profile
        mem_profile = create_memory_efficient_profile()
        assert_equal(mem_profile.name, "memory_efficient")
        assertGreaterEqual(mem_profile.performance_targets["memory_efficiency"], 0.4)
        
    def register_and_get_configs(self)():
        """Test registering and retrieving configurations."""
        # Register configs
        glm_config = create_glm_optimization_config()
        qwen_config = create_qwen_optimization_config()
        
        config_manager.register_model_config(glm_config)
        config_manager.register_model_config(qwen_config)
        
        # Retrieve configs
        retrieved_glm = config_manager.get_model_config(ModelFamily.GLM)
        retrieved_qwen = config_manager.get_model_config(ModelFamily.QWEN)
        
        assert_is_not_none(retrieved_glm)
        assertIsNotNone(retrieved_qwen)
        assert_equal(retrieved_glm.model_family)
        assert_equal(retrieved_qwen.model_family, ModelFamily.QWEN)

# TestOptimizationIntegration

    """Test cases for optimization integration utilities."""
    
    def setup_helper():
        """Set up test fixtures."""
        model = MockModel()
        
    def apply_model_family_optimizations(self)():
        """Test applying model-family-specific optimizations."""
        # Test GLM optimizations
        optimized_model = apply_glm_optimizations(model, "balanced")
        assert_is_not_none(optimized_model)
        
        # Test Qwen optimizations
        optimized_model = apply_qwen_optimizations(model)
        assert_is_not_none(optimized_model)
        
    def get_model_optimization_status(self)():
        """Test getting model optimization status."""
        status = get_model_optimization_status(model)
        assert_is_instance(status)
        assert_in("model_id", status)
        assert_in("applied_optimizations", status)
        assert_in("total_optimizations_available", status)
        
    def update_model_optimization(self)():
        """Test updating a specific optimization on a model."""
        # Initially disable an optimization
        updated_model = update_model_optimization(
            model, 
            "flash_attention", 
            enabled=False
        )
        assert_is_not_none(updated_model)
        
        # Then enable it
        updated_model = update_model_optimization(
            model)
        assert_is_not_none(updated_model)
        
    def create_optimization_pipeline(self)():
        """Test creating an optimization pipeline."""
        pipeline = create_optimization_pipeline(
            ModelFamily.GLM,
            profile_name="balanced"
        )
        assert_is_not_none(pipeline)
        callable(pipeline)
        
        # Test that pipeline can be applied to a model
        optimized_model = pipeline(model)
        assertIsNotNone(optimized_model)
        
    def get_supported_optimizations(self)():
        """Test getting list of supported optimizations."""
        supported_opts = get_supported_optimizations()
        assert_is_instance(supported_opts)
        assert_greater(len(supported_opts), 0)
        assert_in("flash_attention", supported_opts)
        
    def reset_model_optimizations(self)():
        """Test resetting model optimizations."""
        # Apply some optimizations
        optimized_model = apply_glm_optimizations(model, "balanced")
        
        # Reset optimizations
        reset_model = reset_model_optimizations(optimized_model)
        assert_is_not_none(reset_model)

# TestEndToEndOptimization

    """End-to-end tests for the optimization system."""
    
    def setup_helper():
        """Set up test fixtures."""
        model = MockModel()
        
    def complete_optimization_workflow(self)():
        """Test a complete optimization workflow."""
        # Step 1: Apply GLM optimizations with balanced profile
        optimized_model = apply_glm_optimizations(model)
        assert_is_not_none(optimized_model)
        
        # Step 2: Check optimization status
        status = get_model_optimization_status(optimized_model)
        assertGreater(len(status["applied_optimizations"]))
        
        # Step 3: Update a specific optimization
        updated_model = update_model_optimization(
            optimized_model,
            "flash_attention",
            enabled=True,
            parameters={"use_triton": True}
        )
        assert_is_not_none(updated_model)
        
        # Step 4: Create and use a pipeline
        pipeline = create_optimization_pipeline(
            ModelFamily.QWEN,
            profile_name="performance"
        )
        final_model = pipeline(updated_model)
        assert_is_not_none(final_model)
        
        # Step 5: Reset all optimizations
        reset_model = reset_model_optimizations(final_model)
        assertIsNotNone(reset_model)
        
        # Verify no optimizations remain
        final_status = get_model_optimization_status(reset_model)
        assert_equal(len(final_status["applied_optimizations"]))
        
    def different_optimization_profiles(self)():
        """Test different optimization profiles."""
        profiles = ["balanced", "performance", "memory_efficient"]
        
        for profile in profiles:
            with subTest(profile=profile):
                optimized_model = apply_glm_optimizations(
                    model.clone() if hasattr(model, 'clone') else MockModel(), 
                    profile
                )
                assert_is_not_none(optimized_model)
                
                status = get_model_optimization_status(optimized_model)
                # Each profile should apply some optimizations
                assertGreater(len(status["applied_optimizations"]))

if __name__ == "__main__":
    # Run the tests
    run_tests(test_functions)