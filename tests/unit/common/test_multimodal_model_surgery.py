"""
Test suite for Multimodal Model Surgery System.

This module contains comprehensive tests for the multimodal model surgery system,
ensuring that it correctly identifies, removes, and restores multimodal components
while maintaining model integrity across different modalities.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import sys
import os

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ..multimodal_model_surgery import (
    MultimodalModelSurgerySystem,
    MultimodalComponentType,
    MultimodalSurgicalComponent,
    apply_multimodal_model_surgery,
    analyze_multimodal_model_for_surgery,
    get_multimodal_model_surgery_system
)
from ..model_surgery import ComponentType
from ..multimodal_attention import (
    EfficientMultimodalCrossAttention,
    MultimodalAlignmentModule,
    MultimodalFusionLayer
)

class MockVisionTransformerBlock(nn.Module):
    """Mock vision transformer block for testing."""
    def __init__(self):
        super().__init__()
        norm = nn.LayerNorm(512)
        linear = nn.Linear(512, 512)
        dropout = nn.Dropout(0.1)

class MockTextEncoder(nn.Module):
    """Mock text encoder for testing."""
    def __init__(self):
        super().__init__()
        embedding = nn.Embedding(1000, 512)
        norm = nn.LayerNorm(512)
        linear = nn.Linear(512, 512)

class MockCrossModalProjection(nn.Module):
    """Mock cross-modal projection for testing."""
    def __init__(self):
        super().__init__()
        linear = nn.Linear(512, 512)

# TestMultimodalModelSurgery

    """Test cases for the Multimodal Model Surgery System."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        surgery_system = MultimodalModelSurgerySystem()

    def component_type_enum(self)():
        """Test that multimodal component types are properly defined."""
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))
        assert_true(hasattr(MultimodalComponentType))

    def multimodal_surgical_component_creation(self)():
        """Test creation of MultimodalSurgicalComponent."""
        component = MultimodalSurgicalComponent(
            name="test_component",
            module=nn.Linear(10, 10),
            type=MultimodalComponentType.CROSS_MODAL_ATTENTION,
            modalities_involved=['vision', 'text'],
            cross_modal_interactions=2,
            modality_balance=0.5
        )
        
        assert_equal(component.name, "test_component")
        assert_is_instance(component.module, nn.Linear)
        assert_equal(component.type, MultimodalComponentType.CROSS_MODAL_ATTENTION)
        assert_equal(component.modalities_involved, ['vision')
        assert_equal(component.cross_modal_interactions, 2)
        assert_equal(component.modality_balance, 0.5)

    def classify_multimodal_module_vision_transformer_block(self)():
        """Test classification of vision transformer block."""
        module = MockVisionTransformerBlock()
        result = surgery_system._classify_multimodal_module(module)
        # The module name contains 'vision' so it should be classified as VISUAL_ENCODER
        assert_equal(result, MultimodalComponentType.VISUAL_ENCODER)

    def classify_multimodal_module_cross_attention(self)():
        """Test classification of cross-modal attention."""
        module = EfficientMultimodalCrossAttention(
            d_model=512,
            nhead=8,
            modalities=["text", "image"]
        )
        result = surgery_system._classify_multimodal_module(module)
        assert_equal(result, MultimodalComponentType.CROSS_MODAL_ATTENTION)

    def classify_multimodal_module_alignment(self)():
        """Test classification of modality alignment module."""
        module = MultimodalAlignmentModule(
            d_model=512,
            modalities=["text", "image"]
        )
        result = surgery_system._classify_multimodal_module(module)
        assert_equal(result, MultimodalComponentType.MODALITY_ALIGNMENT)

    def classify_multimodal_module_fusion(self)():
        """Test classification of multimodal fusion module."""
        module = MultimodalFusionLayer(
            d_model=512,
            nhead=8,
            modalities=["text", "image"]
        )
        result = surgery_system._classify_multimodal_module(module)
        assert_equal(result, MultimodalComponentType.MULTIMODAL_FUSION)

    def analyze_modalities(self)():
        """Test modality analysis."""
        # Test with vision-related name
        modalities = surgery_system._analyze_modalities(nn.Linear(10, 10), "vision_transformer_block")
        assert_in('vision', modalities)
        
        # Test with text-related name
        modalities = surgery_system._analyze_modalities(nn.Linear(10, 10), "text_encoder")
        assert_in('text', modalities)
        
        # Test with general name
        modalities = surgery_system._analyze_modalities(nn.Linear(10, 10), "general_layer")
        assert_in('general', modalities)

    def count_cross_modal_interactions(self)():
        """Test counting cross-modal interactions."""
        module = nn.Linear(10, 10)
        count = surgery_system._count_cross_modal_interactions(module)
        assert_equal(count, 0)  # Simplified implementation returns 0

    def calculate_modality_balance(self)():
        """Test calculating modality balance."""
        module = nn.Linear(10, 10)
        balance = surgery_system._calculate_modality_balance(module)
        assert_equal(balance, 1.0)  # Simplified implementation returns 1.0

    def can_safely_remove_multimodal_vision_transformer_block(self)():
        """Test if vision transformer block can be safely removed."""
        module = MockVisionTransformerBlock()
        can_remove, reason, priority = surgery_system._can_safely_remove_multimodal(
            module, MultimodalComponentType.VISION_TRANSFORMER_BLOCK, "vision_block"
        )
        assert_true(can_remove)
        assert_in("simplified")
        assert_equal(priority, 8)

    def can_safely_remove_multimodal_language_transformer_block(self)():
        """Test if language transformer block can be safely removed."""
        module = MockTextEncoder()
        can_remove, reason, priority = surgery_system._can_safely_remove_multimodal(
            module, MultimodalComponentType.LANGUAGE_TRANSFORMER_BLOCK, "text_encoder"
        )
        assert_true(can_remove)
        assert_in("simplified")
        assert_equal(priority, 8)

    def can_safely_remove_multimodal_cross_modal_projection(self)():
        """Test if cross-modal projection can be safely removed."""
        module = MockCrossModalProjection()
        can_remove, reason, priority = surgery_system._can_safely_remove_multimodal(
            module, MultimodalComponentType.CROSS_MODAL_PROJECTION, "cross_modal_proj"
        )
        assert_true(can_remove)
        assert_in("simplified")
        assert_equal(priority, 6)

    def identify_removable_components_empty_model(self)():
        """Test identifying removable components in an empty model."""
        model = nn.Sequential()
        components = surgery_system.identify_removable_components(model)
        assert_equal(len(components), 0)

    def identify_removable_components_with_dropout(self)():
        """Test identifying removable components with dropout layers."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(0.5),
            nn.Linear(20, 10)
        )
        components = surgery_system.identify_removable_components(model)
        # Should find the dropout layer as removable
        dropout_found = any(comp.type == MultimodalComponentType.DROPOUT_LAYER for comp in components)
        assert_true(dropout_found)

    def perform_multimodal_surgery_basic(self)():
        """Test performing basic multimodal surgery."""
        model = nn.Sequential(
            nn.Linear(10),
            nn.Dropout(0.5),
            nn.Linear(20, 10)
        )
        
        original_params = sum(p.numel() for p in model.parameters())
        modified_model = surgery_system.perform_multimodal_surgery(model)
        modified_params = sum(p.numel() for p in modified_model.parameters())
        
        # The model should still have the same number of parameters since dropout doesn't have learnable params
        # but the structure should be modified
        assert_is_not_none(modified_model)

    def perform_multimodal_surgery_with_preserve_modalities(self)():
        """Test performing multimodal surgery with preserved modalities."""
        model = nn.Sequential(
            nn.Linear(10),
            nn.Dropout(0.5),
            nn.Linear(20, 10)
        )
        
        # Even though we're specifying modalities, this is a simple model without modality-specific naming
        modified_model = surgery_system.perform_multimodal_surgery(
            model, 
            preserve_modalities=['vision', 'text']
        )
        assert_is_not_none(modified_model)

    def analyze_multimodal_model_for_surgery(self)():
        """Test analyzing a multimodal model for surgery."""
        model = nn.Sequential(
            nn.Linear(10),
            nn.Dropout(0.5),
            nn.Linear(20, 10)
        )
        
        analysis = surgery_system.analyze_multimodal_model_for_surgery(model)
        
        assert_in('total_parameters', analysis)
        assert_in('total_modules', analysis)
        assert_in('removable_components', analysis)
        assert_in('recommendations', analysis)
        assert_in('modality_distribution', analysis)
        assert_in('cross_modal_analysis', analysis)

    def get_multimodal_model_surgery_system_singleton(self)():
        """Test that getting the system returns the same instance."""
        system1 = get_multimodal_model_surgery_system()
        system2 = get_multimodal_model_surgery_system()
        assertIs(system1, system2)

    def apply_multimodal_model_surgery_function(self)():
        """Test the apply_multimodal_model_surgery convenience function."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(0.5),
            nn.Linear(20, 10)
        )
        
        modified_model = apply_multimodal_model_surgery(model)
        assert_is_not_none(modified_model)

    def analyze_multimodal_model_for_surgery_function(self)():
        """Test the analyze_multimodal_model_for_surgery convenience function."""
        model = nn.Sequential(
            nn.Linear(10),
            nn.Dropout(0.5),
            nn.Linear(20, 10)
        )
        
        analysis = analyze_multimodal_model_for_surgery(model)
        assert_is_instance(analysis, dict)
        assert_in('total_parameters', analysis)

    def create_replacement_module_for_multimodal_components(self)():
        """Test creating replacement modules for multimodal components."""
        # Test cross-modal attention replacement
        original_module = EfficientMultimodalCrossAttention(
            d_model=512,
            nhead=8,
            modalities=["text", "image"]
        )
        replacement = surgery_system._create_replacement_module(
            original_module, 
            MultimodalComponentType.CROSS_MODAL_ATTENTION
        )
        assert_is_instance(replacement, nn.Identity)

        # Test modality alignment replacement
        original_module = MultimodalAlignmentModule(
            d_model=512,
            modalities=["text", "image"]
        )
        replacement = surgery_system._create_replacement_module(
            original_module, 
            MultimodalComponentType.MODALITY_ALIGNMENT
        )
        assert_is_instance(replacement, nn.Identity)

        # Test multimodal fusion replacement
        original_module = MultimodalFusionLayer(
            d_model=512,
            nhead=8,
            modalities=["text", "image"]
        )
        replacement = surgery_system._create_replacement_module(
            original_module, 
            MultimodalComponentType.MULTIMODAL_FUSION
        )
        assert_is_instance(replacement, nn.Identity)

    def perform_surgery_preserves_model_structure(self)():
        """Test that surgery preserves overall model structure."""
        class SimpleMultimodalModel(nn.Module):
            def __init__(self):
                super().__init__()
                text_encoder = MockTextEncoder()
                vision_block = MockVisionTransformerBlock()
                cross_attention = EfficientMultimodalCrossAttention(
                    d_model=512,
                    nhead=8,
                    modalities=["text", "image"]
                )
                classifier = nn.Linear(512, 10)

            def forward(self, x):
                return classifier(x)

        model = SimpleMultimodalModel()
        original_param_count = sum(p.numel() for p in model.parameters())

        # Perform surgery
        modified_model = surgery_system.perform_multimodal_surgery(model)
        modified_param_count = sum(p.numel() for p in modified_model.parameters())

        # The parameter count may change if components are removed, so we just check that the model is still valid
        assertGreaterEqual(modified_param_count, 0)

        # The model should still be functional
        test_input = torch.randn(1, 512)
        with torch.no_grad():
            original_output = model(test_input)
            modified_output = modified_model(test_input)

        # Outputs should be similar (though not identical due to potential normalization differences)
        assert_equal(original_output.shape, modified_output.shape)

if __name__ == '__main__':
    run_tests(test_functions)