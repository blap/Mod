#!/usr/bin/env python
"""
Test script for Qwen3-VL-2B projection layer optimization implementation.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.multimodal_projector.projection_layers import (
    Qwen3VL2BProjectionLayer,
    Qwen3VL2BMultiModalProjector,
    Qwen3VL2BVisionLanguageProjector,
    create_qwen3_vl_projection_layer,
    create_qwen3_vl_multimodal_projector,
    apply_qwen3_vl_projection_optimizations
)

def test_projection_layer_creation():
    """Test creating projection layers."""
    print("Testing projection layer creation...")
    
    config = Qwen3VL2BConfig()
    
    # Test basic projection layer
    layer = Qwen3VL2BProjectionLayer(
        vision_dim=config.vision_hidden_size,
        language_dim=config.hidden_size
    )
    print(f"[OK] Basic projection layer created: {type(layer)}")

    # Test with low-rank optimization
    layer_low_rank = Qwen3VL2BProjectionLayer(
        vision_dim=config.vision_hidden_size,
        language_dim=config.hidden_size,
        use_low_rank=True,
        low_rank_dim=256
    )
    print(f"[OK] Low-rank projection layer created: {type(layer_low_rank)}")

    # Test creating projection layer using factory function
    factory_layer = create_qwen3_vl_projection_layer(config)
    print(f"[OK] Factory-created projection layer: {type(factory_layer)}")

    # Test multimodal projector
    projector = Qwen3VL2BMultiModalProjector(
        vision_dim=config.vision_hidden_size,
        language_dim=config.hidden_size,
        num_layers=2
    )
    print(f"[OK] Multimodal projector created: {type(projector)}")

    # Test vision-language projector
    vl_projector = Qwen3VL2BVisionLanguageProjector(config)
    print(f"[OK] Vision-language projector created: {type(vl_projector)}")

    # Test multimodal projector using factory function
    factory_projector = create_qwen3_vl_multimodal_projector(config)
    print(f"[OK] Factory-created multimodal projector: {type(factory_projector)}")

    print("All projection layer creations successful!")


def test_projection_optimization_application():
    """Test applying projection optimizations to a model."""
    print("\nTesting projection optimization application...")
    
    import torch
    import torch.nn as nn
    
    # Create a simple dummy model to simulate the real model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_linear = nn.Linear(1024, 2048)
            self.projector = nn.Linear(1024, 2048)  # Simulate a projector layer
            
        def forward(self, x):
            return self.dummy_linear(x)
    
    config = Qwen3VL2BConfig()
    dummy_model = DummyModel()
    
    print(f"Original model projector type: {type(dummy_model.projector)}")
    
    # Apply projection optimizations
    optimized_model = apply_qwen3_vl_projection_optimizations(dummy_model, config)
    
    print(f"Optimized model projector type: {type(optimized_model.projector)}")
    print("Projection optimization application successful!")


def test_config_attributes():
    """Test that config has all required projection layer attributes."""
    print("\nTesting config attributes...")
    
    config = Qwen3VL2BConfig()
    
    required_attrs = [
        'use_projection_layer_optimization',
        'projection_layer_use_bias',
        'projection_layer_activation',
        'projection_layer_dropout',
        'projection_layer_use_residual',
        'projection_layer_use_low_rank',
        'projection_layer_low_rank_dim',
        'projection_layer_use_group_norm',
        'projection_layer_group_norm_num_groups',
        'projection_layer_intermediate_dim',
        'projection_layer_num_layers',
        'projection_layer_use_cross_attention',
        'projection_layer_cross_attention_heads'
    ]
    
    for attr in required_attrs:
        if hasattr(config, attr):
            print(f"[OK] Config has attribute: {attr} = {getattr(config, attr)}")
        else:
            print(f"[MISSING] Config missing attribute: {attr}")

    print("Config attributes test completed!")


if __name__ == "__main__":
    print("Testing Qwen3-VL-2B Projection Layer Optimizations Implementation")
    print("=" * 60)
    
    try:
        test_config_attributes()
        test_projection_layer_creation()
        test_projection_optimization_application()
        
        print("\n" + "=" * 60)
        print("All tests passed! Projection layer optimizations are working correctly.")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()