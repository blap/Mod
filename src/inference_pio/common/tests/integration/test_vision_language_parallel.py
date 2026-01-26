"""
Test suite for Vision-Language Parallelism System.

This module contains comprehensive tests for the vision-language parallelism system,
verifying that visual and textual components are processed correctly in parallel.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.inference_pio.common.vision_language_parallel import (
    VisionLanguageParallel,
    VisionLanguageConfig,
    VisualStage,
    TextualStage,
    MultimodalFusionModule,
    create_vision_language_config,
    split_model_for_vision_language
)

class DummyVisualModule(nn.Module):
    """Dummy visual module for testing."""
    def __init__(self, output_dim=512):
        super().__init__()
        conv = nn.Conv2d(1, 32, 3, padding=1)
        relu = nn.ReLU()
        pool = nn.AdaptiveAvgPool2d((4, 4))
        flatten = nn.Flatten()
        fc = nn.Linear(32 * 4 * 4, output_dim)
    
    def forward(self, x):
        x = relu(conv(x))
        x = pool(x)
        x = flatten(x)
        x = fc(x)
        return x

class DummyTextualModule(nn.Module):
    """Dummy textual module for testing."""
    def __init__(self, vocab_size=1000, embed_dim=512):
        super().__init__()
        embedding = nn.Embedding(vocab_size, embed_dim)
        linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        x = embedding(x)
        x = linear(x)
        return x

class DummyMultimodalModel(nn.Module):
    """Dummy multimodal model combining visual and textual components."""
    def __init__(self):
        super().__init__()
        visual_encoder = nn.Sequential(
            DummyVisualModule(output_dim=256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        textual_encoder = nn.Sequential(
            DummyTextualModule(vocab_size=1000, embed_dim=256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        classifier = nn.Linear(256 * 2, 10)  # Combined visual and textual features
    
    def forward(self, pixel_values=None, input_ids=None):
        features = []
        
        if pixel_values is not None:
            visual_features = visual_encoder(pixel_values)
            features.append(visual_features)
        
        if input_ids is not None:
            textual_features = textual_encoder(input_ids)
            features.append(textual_features)
        
        if len(features) > 1:
            combined_features = torch.cat(features, dim=-1)
        elif len(features) == 1:
            combined_features = features[0]
        else:
            raise ValueError("At least one input (pixel_values or input_ids) must be provided")
        
        return classifier(combined_features)

# TestVisionLanguageParallel

    """Test cases for the vision-language parallelism system."""
    
    def setup_helper():
        """Set up test fixtures before each test method."""
        model = DummyMultimodalModel()
        config = create_vision_language_config(
            num_visual_stages=2,
            num_textual_stages=2,
            enable_cross_modal_communication=True,
            pipeline_schedule='interleaved'
        )
    
    def vision_language_parallel_initialization(self)():
        """Test that the vision-language parallel model initializes correctly."""
        vl_parallel = VisionLanguageParallel(model, config)
        
        assert_is_instance(vl_parallel, VisionLanguageParallel)
        assert_equal(len(vl_parallel.visual_stages), 2)
        assert_equal(len(vl_parallel.textual_stages), 2)
        assert_true(vl_parallel.config.enable_cross_modal_communication)
    
    def visual_stage_creation(self)():
        """Test that visual stages are created correctly."""
        vl_parallel = VisionLanguageParallel(model)
        
        for stage in vl_parallel.visual_stages:
            assert_is_instance(stage, VisualStage)
            assertGreaterEqual(len(list(stage.visual_components.parameters())), 0)
    
    def textual_stage_creation(self)():
        """Test that textual stages are created correctly."""
        vl_parallel = VisionLanguageParallel(model, config)
        
        for stage in vl_parallel.textual_stages:
            assert_is_instance(stage, TextualStage)
            assertGreaterEqual(len(list(stage.textual_components.parameters())), 0)
    
    def forward_pass_single_modality_visual(self)():
        """Test forward pass with only visual input."""
        vl_parallel = VisionLanguageParallel(model, config)
        
        # Create dummy visual input (batch_size=2, channels=1, height=28, width=28)
        visual_input = torch.randn(2, 1, 28, 28)
        
        output = vl_parallel(visual_input)
        
        # Output should be a tensor with the same batch size
        assert_equal(output.shape[0], 2)
        # The output dimension depends on the last layer of the visual encoder
        assert_equal(output.shape[1], 256)  # Last layer output dim
    
    def forward_pass_single_modality_textual(self)():
        """Test forward pass with only textual input."""
        vl_parallel = VisionLanguageParallel(model, config)
        
        # Create dummy textual input (batch_size=2, seq_len=10)
        textual_input = torch.randint(0, 1000, (2, 10))
        
        output = vl_parallel(textual_input)
        
        # Output should be a tensor with the same batch size
        assert_equal(output.shape[0], 2)
        # The output dimension depends on the last layer of the textual encoder
        assert_equal(output.shape[1], 256)  # Last layer output dim
    
    def forward_pass_multimodal(self)():
        """Test forward pass with both visual and textual inputs."""
        # Create a config that enables multimodal fusion
        config = create_vision_language_config(
            num_visual_stages=1,
            num_textual_stages=1,
            enable_cross_modal_communication=True,
            enable_multimodal_fusion=True,
            pipeline_schedule='sequential'
        )
        
        vl_parallel = VisionLanguageParallel(model, config)
        
        # Create multimodal input
        inputs = {
            'pixel_values': torch.randn(2, 1, 28, 28),
            'input_ids': torch.randint(0, 1000, (2, 10))
        }
        
        output = vl_parallel(inputs)
        
        # With multimodal fusion enabled, output should be a dict with visual and textual components
        assert_is_instance(output, dict)
        assert_in('visual', output)
        assert_in('textual', output)
        assert_equal(output['visual'].shape[0], 2)
        assert_equal(output['textual'].shape[0], 2)
    
    def forward_pass_without_multimodal_fusion(self)():
        """Test forward pass without multimodal fusion."""
        # Create a config that disables multimodal fusion
        config = create_vision_language_config(
            num_visual_stages=1,
            num_textual_stages=1,
            enable_cross_modal_communication=True,
            enable_multimodal_fusion=False,
            pipeline_schedule='sequential'
        )
        
        vl_parallel = VisionLanguageParallel(model, config)
        
        # Create multimodal input
        inputs = {
            'pixel_values': torch.randn(2, 1, 28, 28),
            'input_ids': torch.randint(0, 1000, (2, 10))
        }
        
        output = vl_parallel(inputs)
        
        # Without multimodal fusion, output should still be a dict with visual and textual components
        assert_is_instance(output, dict)
        assert_in('visual', output)
        assert_in('textual', output)
        assert_equal(output['visual'].shape[0], 2)
        assert_equal(output['textual'].shape[0], 2)
    
    def multimodal_fusion_module(self)():
        """Test the multimodal fusion module directly."""
        fusion_module = MultimodalFusionModule(d_model=256, nhead=8, dropout=0.1)
        
        # Create dummy visual and textual features
        visual_features = torch.randn(2, 10, 256)  # batch_size=2, seq_len=10, d_model=256
        textual_features = torch.randn(2, 10, 256)  # batch_size=2, seq_len=10, d_model=256
        
        fused_visual, fused_textual = fusion_module(visual_features, textual_features)
        
        # Output shapes should match input shapes
        assert_equal(fused_visual.shape, visual_features.shape)
        assert_equal(fused_textual.shape, textual_features.shape)
    
    def different_pipeline_schedules(self)():
        """Test different pipeline schedules."""
        schedules = ['sequential', 'interleaved', 'async']
        
        for schedule in schedules:
            config = create_vision_language_config(
                num_visual_stages=2,
                num_textual_stages=2,
                pipeline_schedule=schedule
            )
            
            vl_parallel = VisionLanguageParallel(model, config)
            
            # Test with visual input
            visual_input = torch.randn(2, 1, 28, 28)
            output = vl_parallel(visual_input)
            
            assert_equal(output.shape[0], 2)
    
    def config_creation(self)():
        """Test creation of vision-language config with various parameters."""
        config = create_vision_language_config(
            num_visual_stages=3,
            num_textual_stages=4,
            visual_device_mapping=['cpu', 'cpu', 'cpu'],
            textual_device_mapping=['cpu', 'cpu', 'cpu', 'cpu'],
            enable_cross_modal_communication=True,
            pipeline_schedule='interleaved'
        )
        
        assert_equal(config.num_visual_stages, 3)
        assert_equal(config.num_textual_stages, 4)
        assert_equal(config.visual_device_mapping, ['cpu')
        assert_equal(config.textual_device_mapping, ['cpu')
        assert_true(config.enable_cross_modal_communication)
        assert_equal(config.pipeline_schedule)

# TestUtilityFunctions

    """Test utility functions."""
    
    def split_model_for_vision_language(self)():
        """Test the model splitting utility function."""
        model = DummyMultimodalModel()
        
        visual_stages, textual_stages = split_model_for_vision_language(
            model, num_visual_stages=2, num_textual_stages=2
        )
        
        assert_equal(len(visual_stages), 2)
        assert_equal(len(textual_stages), 2)
        
        # Each stage should be a Sequential module
        for stage in visual_stages:
            assert_is_instance(stage, nn.Sequential)
        
        for stage in textual_stages:
            assert_is_instance(stage, nn.Sequential)

if __name__ == '__main__':
    run_tests(test_functions)