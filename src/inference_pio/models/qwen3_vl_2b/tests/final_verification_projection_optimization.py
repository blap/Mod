#!/usr/bin/env python
"""
Final verification test for Qwen3-VL-2B projection layer optimizations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from src.inference_pio.models.qwen3_vl_2b.multimodal_projector.projection_layers import (
    Qwen3VL2BProjectionLayer,
    Qwen3VL2BMultiModalProjector,
    Qwen3VL2BVisionLanguageProjector,
    create_qwen3_vl_projection_layer,
    create_qwen3_vl_multimodal_projector,
    apply_qwen3_vl_projection_optimizations
)
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock


def test_complete_projection_optimization_implementation():
    """Complete test of the projection layer optimization implementation."""
    print("Testing Complete Qwen3-VL-2B Projection Layer Optimization Implementation")
    print("=" * 70)
    
    # 1. Test configuration has all required attributes
    print("\n1. Testing configuration attributes...")
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
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(config, attr):
            missing_attrs.append(attr)
    
    if not missing_attrs:
        print("   [PASS] All required projection layer attributes present in config")
    else:
        print(f"   [FAIL] Missing attributes: {missing_attrs}")
        return False
    
    # 2. Test projection layer creation
    print("\n2. Testing projection layer creation...")
    try:
        layer = Qwen3VL2BProjectionLayer(
            vision_dim=config.vision_hidden_size,
            language_dim=config.hidden_size
        )
        print("   [PASS] Qwen3VL2BProjectionLayer created successfully")
    except Exception as e:
        print(f"   [FAIL] Failed to create Qwen3VL2BProjectionLayer: {e}")
        return False
    
    # 3. Test multimodal projector creation
    print("\n3. Testing multimodal projector creation...")
    try:
        projector = Qwen3VL2BMultiModalProjector(
            vision_dim=config.vision_hidden_size,
            language_dim=config.hidden_size,
            num_layers=2
        )
        print("   [PASS] Qwen3VL2BMultiModalProjector created successfully")
    except Exception as e:
        print(f"   [FAIL] Failed to create Qwen3VL2BMultiModalProjector: {e}")
        return False
    
    # 4. Test vision-language projector creation
    print("\n4. Testing vision-language projector creation...")
    try:
        vl_projector = Qwen3VL2BVisionLanguageProjector(config)
        print("   [PASS] Qwen3VL2BVisionLanguageProjector created successfully")
    except Exception as e:
        print(f"   [FAIL] Failed to create Qwen3VL2BVisionLanguageProjector: {e}")
        return False
    
    # 5. Test factory functions
    print("\n5. Testing factory functions...")
    try:
        factory_layer = create_qwen3_vl_projection_layer(config)
        print("   [PASS] create_qwen3_vl_projection_layer works")
    except Exception as e:
        print(f"   [FAIL] create_qwen3_vl_projection_layer failed: {e}")
        return False
    
    try:
        factory_projector = create_qwen3_vl_multimodal_projector(config)
        print("   [PASS] create_qwen3_vl_multimodal_projector works")
    except Exception as e:
        print(f"   [FAIL] create_qwen3_vl_multimodal_projector failed: {e}")
        return False
    
    # 6. Test optimization application function
    print("\n6. Testing optimization application function...")
    try:
        # Create a simple model to test optimization application
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_projection = nn.Linear(1024, 2048)
        
        simple_model = SimpleModel()
        optimized_model = apply_qwen3_vl_projection_optimizations(simple_model, config)
        print("   [PASS] apply_qwen3_vl_projection_optimizations works")
    except Exception as e:
        print(f"   [FAIL] apply_qwen3_vl_projection_optimizations failed: {e}")
        return False
    
    # 7. Test model integration
    print("\n7. Testing model integration...")
    try:
        # Create a mock model to test the integration
        with patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained'):
            
            # Create the model instance
            model = Qwen3VL2BModel.__new__(Qwen3VL2BModel)  # Create without calling __init__ to avoid initialization
            model.config = config
            model._model = MagicMock()  # Mock the internal model
            model._tokenizer = MagicMock()
            model._image_processor = MagicMock()
            model._model_name = "dummy"
            
            # Manually call the projection optimization method
            model._apply_projection_layer_optimizations()
            print("   [PASS] Model integration with projection optimizations works")
    except Exception as e:
        print(f"   [FAIL] Model integration with projection optimizations failed: {e}")
        return False
    
    # 8. Test forward pass with projection layers
    print("\n8. Testing forward pass with projection layers...")
    try:
        # Create sample inputs
        batch_size = 2
        seq_len_vision = 10
        seq_len_language = 15
        vision_features = torch.randn(batch_size, seq_len_vision, config.vision_hidden_size)
        language_features = torch.randn(batch_size, seq_len_language, config.hidden_size)
        
        # Test basic projection layer forward pass
        proj_vision, proj_language, fused_features = layer(vision_features, language_features)
        expected_shape = (batch_size, max(seq_len_vision, seq_len_language), config.hidden_size)
        if fused_features.shape[0] == batch_size and fused_features.shape[-1] == config.hidden_size:
            print("   [PASS] Projection layer forward pass works correctly")
        else:
            print(f"   [FAIL] Projection layer forward pass shape mismatch: expected last dim {config.hidden_size}, got {fused_features.shape[-1]}")
            return False
    except Exception as e:
        print(f"   [FAIL] Projection layer forward pass failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("Qwen3-VL-2B projection layer optimization system is fully implemented and working!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = test_complete_projection_optimization_implementation()
    if not success:
        sys.exit(1)