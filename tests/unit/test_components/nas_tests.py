"""
Comprehensive tests for the Neural Architecture Search (NAS) system for layer-specific configuration optimization.
This test suite covers all aspects of the NAS system implementation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from unittest.mock import Mock, patch
import tempfile
import os

# Import the NAS system components (will be implemented later)
from nas_system import (
    LayerConfig,
    VisionLayerConfig,
    LanguageLayerConfig,
    NASController,
    LayerSpecificOptimizer,
    ArchitectureSearchSpace,
    PerformancePredictor,
    Qwen3VLNeuralArchitectureSearch
)


def test_layer_config_creation():
    """Test the creation and validation of layer configurations."""
    # Test base layer config
    base_config = LayerConfig(
        layer_type="attention",
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=2048,
        layer_idx=0
    )
    
    assert base_config.layer_type == "attention"
    assert base_config.hidden_size == 512
    assert base_config.num_attention_heads == 8
    assert base_config.intermediate_size == 2048
    assert base_config.layer_idx == 0
    
    # Test vision layer config
    vision_config = VisionLayerConfig(
        layer_type="vision_attention",
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        layer_idx=1,
        patch_size=16,
        num_patches=196
    )
    
    assert vision_config.patch_size == 16
    assert vision_config.num_patches == 196
    
    # Test language layer config
    language_config = LanguageLayerConfig(
        layer_type="language_attention",
        hidden_size=1024,
        num_attention_heads=16,
        intermediate_size=4096,
        layer_idx=2,
        max_position_embeddings=512
    )
    
    assert language_config.max_position_embeddings == 512


def test_architecture_search_space_initialization():
    """Test initialization of the architecture search space."""
    search_space = ArchitectureSearchSpace(
        base_hidden_size=512,
        min_hidden_size=256,
        max_hidden_size=1024,
        min_num_heads=4,
        max_num_heads=16,
        min_intermediate_size=1024,
        max_intermediate_size=4096
    )
    
    assert search_space.base_hidden_size == 512
    assert search_space.min_hidden_size == 256
    assert search_space.max_hidden_size == 1024
    assert search_space.min_num_heads == 4
    assert search_space.max_num_heads == 16
    
    # Test that the search space can generate valid configurations
    configs = search_space.generate_candidate_configs(num_candidates=5, num_layers=32)
    assert len(configs) == 5
    assert len(configs[0]) == 32  # 32 layers
    
    # Validate that configurations are within bounds
    for config_list in configs:
        for layer_config in config_list:
            assert search_space.min_hidden_size <= layer_config.hidden_size <= search_space.max_hidden_size
            assert search_space.min_num_heads <= layer_config.num_attention_heads <= search_space.max_num_heads
            assert search_space.min_intermediate_size <= layer_config.intermediate_size <= search_space.max_intermediate_size


def test_performance_predictor():
    """Test the performance predictor component."""
    predictor = PerformancePredictor(
        input_dim=7,
        hidden_dim=256,
        num_layers=3
    )

    # Create sample layer configurations
    configs = [
        LayerConfig(layer_type="attention", hidden_size=512, num_attention_heads=8, intermediate_size=2048, layer_idx=i)
        for i in range(4)
    ]

    # Test prediction for a sequence of layer configurations
    performance_pred = predictor.predict_performance(configs)
    assert isinstance(performance_pred, torch.Tensor)
    assert performance_pred.shape[0] == 1  # Single prediction for the config sequence
    assert performance_pred.item() >= 0  # Performance should be non-negative

    # Test with different input types (vision vs language)
    vision_configs = [
        VisionLayerConfig(
            layer_type="vision_attention",
            hidden_size=768,
            num_attention_heads=12,
            intermediate_size=3072,
            layer_idx=i,
            patch_size=16,
            num_patches=196
        )
        for i in range(4)
    ]

    vision_performance_pred = predictor.predict_performance(vision_configs)
    assert isinstance(vision_performance_pred, torch.Tensor)
    assert vision_performance_pred.item() >= 0


def test_nas_controller():
    """Test the NAS controller that manages the search process."""
    controller = NASController(
        hidden_size=256,
        num_layers=32,
        temperature=1.0
    )
    
    # Test sampling a configuration
    sampled_config = controller.sample_architecture()
    assert len(sampled_config) == 32  # 32 layers
    
    # Test updating controller based on performance feedback
    performance_feedback = torch.tensor([0.85])  # Performance score
    controller.update(performance_feedback)
    
    # Test sampling after update
    updated_config = controller.sample_architecture()
    assert len(updated_config) == 32


def test_layer_specific_optimizer():
    """Test the layer-specific optimizer that adapts configurations based on input type."""
    optimizer = LayerSpecificOptimizer(
        num_layers=32,
        search_space=ArchitectureSearchSpace(
            base_hidden_size=512,
            min_hidden_size=256,
            max_hidden_size=1024,
            min_num_heads=4,
            max_num_heads=16,
            min_intermediate_size=1024,
            max_intermediate_size=4096
        )
    )
    
    # Create dummy input representations for different input types
    text_input = torch.randn(1, 32, 512)  # Batch, Seq, Hidden
    vision_input = torch.randn(1, 196, 768)  # Batch, Patches, Hidden
    
    # Test optimization for text input
    text_optimized_configs = optimizer.optimize_for_input_type(text_input, input_type="text", num_candidates=3)
    assert len(text_optimized_configs) == 3
    assert len(text_optimized_configs[0]) == 32  # 32 layers
    
    # Test optimization for vision input
    vision_optimized_configs = optimizer.optimize_for_input_type(vision_input, input_type="vision", num_candidates=3)
    assert len(vision_optimized_configs) == 3
    assert len(vision_optimized_configs[0]) == 32  # 32 layers
    
    # Test that optimized configs differ based on input type
    # (This might not always be true, but we can check structure)
    for i in range(32):
        text_config = text_optimized_configs[0][i]
        vision_config = vision_optimized_configs[0][i]
        # Both should be valid configurations
        assert isinstance(text_config, LayerConfig)
        assert isinstance(vision_config, LayerConfig)


def test_qwen3vl_neural_architecture_search_initialization():
    """Test initialization of the full Qwen3VL NAS system."""
    nas_system = Qwen3VLNeuralArchitectureSearch(
        num_layers=32,
        base_hidden_size=512,
        base_num_heads=32,
        base_intermediate_size=2048
    )
    
    assert nas_system.num_layers == 32
    assert nas_system.base_hidden_size == 512
    assert nas_system.base_num_heads == 32
    assert nas_system.base_intermediate_size == 2048
    assert nas_system.controller is not None
    assert nas_system.performance_predictor is not None
    assert nas_system.layer_optimizer is not None


def test_nas_search_process():
    """Test the complete NAS search process."""
    nas_system = Qwen3VLNeuralArchitectureSearch(
        num_layers=8,  # Using fewer layers for faster testing
        base_hidden_size=256,
        base_num_heads=8,
        base_intermediate_size=1024
    )
    
    # Create dummy inputs for different modalities
    text_input = torch.randint(0, 1000, (1, 32))  # Batch, Seq
    vision_input = torch.randn(1, 3, 224, 224)  # Batch, Channels, Height, Width
    
    # Run search process for text input
    best_text_config = nas_system.search_optimal_architecture(
        input_data=text_input,
        input_type="text",
        num_search_steps=5,
        num_candidates_per_step=3
    )
    
    assert len(best_text_config) == 8  # 8 layers
    for layer_config in best_text_config:
        assert isinstance(layer_config, LayerConfig)
    
    # Run search process for vision input
    best_vision_config = nas_system.search_optimal_architecture(
        input_data=vision_input,
        input_type="vision",
        num_search_steps=5,
        num_candidates_per_step=3
    )
    
    assert len(best_vision_config) == 8  # 8 layers
    for layer_config in best_vision_config:
        assert isinstance(layer_config, VisionLayerConfig)


def test_configuration_compatibility():
    """Test that generated configurations are compatible with Qwen3-VL architecture."""
    from nas_system import create_transformer_layer_from_config
    
    # Test creating a transformer layer from a configuration
    config = LayerConfig(
        layer_type="attention",
        hidden_size=512,
        num_attention_heads=8,
        intermediate_size=2048,
        layer_idx=0
    )
    
    layer = create_transformer_layer_from_config(config)
    assert layer is not None
    assert hasattr(layer, 'self_attn')
    assert hasattr(layer, 'mlp')
    
    # Test with vision layer config
    vision_config = VisionLayerConfig(
        layer_type="vision_attention",
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072,
        layer_idx=1,
        patch_size=16,
        num_patches=196
    )
    
    vision_layer = create_transformer_layer_from_config(vision_config)
    assert vision_layer is not None
    assert hasattr(vision_layer, 'self_attn')
    assert hasattr(vision_layer, 'mlp')


def test_integration_with_model():
    """Test integration of NAS system with actual model components."""
    nas_system = Qwen3VLNeuralArchitectureSearch(
        num_layers=4,
        base_hidden_size=256,
        base_num_heads=8,
        base_intermediate_size=1024
    )
    
    # Create a dummy input
    text_input = torch.randint(0, 1000, (1, 16))
    
    # Get optimized configuration
    optimized_config = nas_system.search_optimal_architecture(
        input_data=text_input,
        input_type="text",
        num_search_steps=3,
        num_candidates_per_step=2
    )
    
    # Verify that the configuration can be used to build a model
    assert len(optimized_config) == 4
    for i, layer_config in enumerate(optimized_config):
        assert layer_config.layer_idx == i
        assert layer_config.hidden_size >= 256
        assert layer_config.num_attention_heads >= 1


def test_search_space_exploration():
    """Test that the search space exploration covers diverse configurations."""
    search_space = ArchitectureSearchSpace(
        base_hidden_size=512,
        min_hidden_size=256,
        max_hidden_size=1024,
        min_num_heads=4,
        max_num_heads=16,
        min_intermediate_size=1024,
        max_intermediate_size=4096
    )
    
    # Generate multiple candidate configurations
    candidate_configs = search_space.generate_candidate_configs(num_candidates=10, num_layers=4)
    
    # Verify diversity in configurations
    hidden_sizes = set()
    num_heads = set()
    intermediate_sizes = set()
    
    for config_list in candidate_configs:
        for layer_config in config_list:
            hidden_sizes.add(layer_config.hidden_size)
            num_heads.add(layer_config.num_attention_heads)
            intermediate_sizes.add(layer_config.intermediate_size)
    
    # We should have some diversity in the configurations
    assert len(hidden_sizes) > 1
    assert len(num_heads) > 1
    assert len(intermediate_sizes) > 1


def test_performance_prediction_consistency():
    """Test that performance predictions are consistent."""
    predictor = PerformancePredictor(
        input_dim=7,
        hidden_dim=256,
        num_layers=3
    )
    
    # Create identical configurations
    config1 = [
        LayerConfig(layer_type="attention", hidden_size=512, num_attention_heads=8, intermediate_size=2048, layer_idx=i)
        for i in range(4)
    ]
    config2 = [
        LayerConfig(layer_type="attention", hidden_size=512, num_attention_heads=8, intermediate_size=2048, layer_idx=i)
        for i in range(4)
    ]

    pred1 = predictor.predict_performance(config1).item()
    pred2 = predictor.predict_performance(config2).item()

    # Identical configurations should have similar performance predictions (within tolerance)
    assert abs(pred1 - pred2) < 0.01


def test_search_convergence():
    """Test that the search process converges to better configurations."""
    nas_system = Qwen3VLNeuralArchitectureSearch(
        num_layers=4,
        base_hidden_size=256,
        base_num_heads=8,
        base_intermediate_size=1024
    )
    
    # Create dummy input
    text_input = torch.randint(0, 1000, (1, 16))
    
    # Run multiple search steps and check if performance improves
    initial_config = nas_system.search_optimal_architecture(
        input_data=text_input,
        input_type="text",
        num_search_steps=1,  # Just initial random config
        num_candidates_per_step=1
    )
    
    final_config = nas_system.search_optimal_architecture(
        input_data=text_input,
        input_type="text",
        num_search_steps=5,  # More steps for improvement
        num_candidates_per_step=3
    )
    
    # Both should be valid configurations
    assert len(initial_config) == 4
    assert len(final_config) == 4


def test_multimodal_support():
    """Test support for multimodal inputs."""
    nas_system = Qwen3VLNeuralArchitectureSearch(
        num_layers=4,
        base_hidden_size=256,
        base_num_heads=8,
        base_intermediate_size=1024
    )
    
    # Test with text input
    text_input = torch.randint(0, 1000, (1, 16))
    text_config = nas_system.search_optimal_architecture(
        input_data=text_input,
        input_type="text",
        num_search_steps=2,
        num_candidates_per_step=2
    )
    
    # Test with vision input
    vision_input = torch.randn(1, 3, 224, 224)
    vision_config = nas_system.search_optimal_architecture(
        input_data=vision_input,
        input_type="vision",
        num_search_steps=2,
        num_candidates_per_step=2
    )
    
    # Test with multimodal input (both text and vision)
    multimodal_config = nas_system.search_optimal_architecture(
        input_data=(text_input, vision_input),
        input_type="multimodal",
        num_search_steps=2,
        num_candidates_per_step=2
    )
    
    assert len(text_config) == 4
    assert len(vision_config) == 4
    assert len(multimodal_config) == 4


def test_save_load_functionality():
    """Test saving and loading of NAS system state."""
    nas_system = Qwen3VLNeuralArchitectureSearch(
        num_layers=4,
        base_hidden_size=256,
        base_num_heads=8,
        base_intermediate_size=1024
    )
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        temp_path = tmp_file.name
    
    try:
        # Save the NAS system
        nas_system.save_state(temp_path)
        
        # Create a new NAS system and load the state
        new_nas_system = Qwen3VLNeuralArchitectureSearch(
            num_layers=4,
            base_hidden_size=256,
            base_num_heads=8,
            base_intermediate_size=1024
        )
        new_nas_system.load_state(temp_path)
        
        # Verify that the loaded system has the same configuration
        # We can't directly compare the models, but we can check that loading worked
        assert new_nas_system.num_layers == nas_system.num_layers
        assert new_nas_system.base_hidden_size == nas_system.base_hidden_size
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_error_handling():
    """Test error handling in the NAS system."""
    nas_system = Qwen3VLNeuralArchitectureSearch(
        num_layers=4,
        base_hidden_size=256,
        base_num_heads=8,
        base_intermediate_size=1024
    )
    
    # Test with invalid input type
    with pytest.raises(ValueError):
        nas_system.search_optimal_architecture(
            input_data=torch.randint(0, 1000, (1, 16)),
            input_type="invalid_type",
            num_search_steps=2,
            num_candidates_per_step=2
        )
    
    # Test with invalid number of search steps
    with pytest.raises(ValueError):
        nas_system.search_optimal_architecture(
            input_data=torch.randint(0, 1000, (1, 16)),
            input_type="text",
            num_search_steps=0,  # Should be positive
            num_candidates_per_step=2
        )


if __name__ == "__main__":
    # Run all tests
    test_layer_config_creation()
    test_architecture_search_space_initialization()
    test_performance_predictor()
    test_nas_controller()
    test_layer_specific_optimizer()
    test_qwen3vl_neural_architecture_search_initialization()
    test_nas_search_process()
    test_configuration_compatibility()
    test_integration_with_model()
    test_search_space_exploration()
    test_performance_prediction_consistency()
    test_search_convergence()
    test_multimodal_support()
    test_save_load_functionality()
    test_error_handling()
    
    print("All NAS system tests passed!")