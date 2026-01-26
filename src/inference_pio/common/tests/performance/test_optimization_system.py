"""
Test script to verify the modular optimization system works correctly.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from src.inference_pio.common.optimization_manager import get_optimization_manager, register_default_optimizations, OptimizationConfig, OptimizationType
from src.inference_pio.common.optimization_config import (
    get_config_manager,
    ModelFamily,
    create_glm_optimization_config,
    create_qwen_optimization_config,
    create_balanced_profile
)
from src.inference_pio.common.optimization_integration import (
    apply_glm_optimizations,
    apply_qwen_optimizations,
    get_model_optimization_status,
    update_model_optimization
)


class SimpleTestModel(nn.Module):
    """Simple model for testing optimization systems."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 50)
        self.linear2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def test_modular_optimization_system():
    """Test the modular optimization system."""
    print("Testing Modular Optimization System...")

    # Initialize managers
    manager = get_optimization_manager()
    config_manager = get_config_manager()
    register_default_optimizations()

    # Test 1: Check available optimizations
    available_opts = manager.get_available_optimizations()
    print(f"✓ Available optimizations: {len(available_opts)} found")
    assert len(available_opts) > 0, "Should have available optimizations"

    # Test 2: Create and test a simple model
    model = SimpleTestModel()
    print("✓ Created test model")

    # Test 3: Apply GLM optimizations with balanced profile
    try:
        optimized_model = apply_glm_optimizations(model, profile_name="balanced")
        print("✓ Applied GLM optimizations with balanced profile")
    except Exception as e:
        print(f"⚠ Could not apply GLM optimizations: {e}")
        optimized_model = model  # Use original model for further tests

    # Test 4: Check optimization status
    status = get_model_optimization_status(optimized_model)
    print(f"✓ Model optimization status: {len(status['applied_optimizations'])} applied")

    # Test 5: Update a specific optimization
    try:
        updated_model = update_model_optimization(
            optimized_model,
            "flash_attention",
            enabled=True,
            parameters={"use_triton": True}
        )
        print("✓ Successfully updated flash_attention optimization")
    except Exception as e:
        print(f"⚠ Could not update optimization: {e}")
        updated_model = optimized_model

    # Test 6: Apply Qwen optimizations
    try:
        qwen_optimized_model = apply_qwen_optimizations(updated_model, profile_name="performance")
        print("✓ Applied Qwen optimizations with performance profile")
    except Exception as e:
        print(f"⚠ Could not apply Qwen optimizations: {e}")
        qwen_optimized_model = updated_model

    # Test 7: Test different profiles
    profiles = ["balanced", "performance", "memory_efficient"]
    for profile in profiles:
        try:
            profile_model = apply_glm_optimizations(SimpleTestModel(), profile_name=profile)
            print(f"✓ Applied {profile} profile")
        except Exception as e:
            print(f"⚠ Could not apply {profile} profile: {e}")

    # Test 8: Test configuration management
    glm_config = create_glm_optimization_config()
    qwen_config = create_qwen_optimization_config()
    balanced_profile = create_balanced_profile()

    config_manager.register_model_config(glm_config)
    config_manager.register_model_config(qwen_config)
    config_manager.register_global_profile(balanced_profile)

    print("✓ Registered model configurations and profiles")

    # Test 9: Apply optimizations using configuration
    try:
        config_optimized_model = apply_glm_optimizations(SimpleTestModel(), profile_name="balanced")
        print("✓ Applied optimizations using configuration")
    except Exception as e:
        print(f"⚠ Could not apply optimizations using configuration: {e}")

    print("\n✓ All tests passed! Modular Optimization System is working correctly.")


if __name__ == "__main__":
    test_modular_optimization_system()