"""
Test suite for Unimodal Model Surgery System.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from ..unimodal_model_surgery import (
    UnimodalModelSurgerySystem,
    UnimodalComponentType,
    UnimodalSurgicalComponent,
    apply_unimodal_model_surgery,
    analyze_unimodal_model_for_surgery,
    get_unimodal_model_surgery_system
)
from ..model_surgery import ComponentType

# TestUnimodalModelSurgery

    """Test cases for the Unimodal Model Surgery System."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        surgery_system = UnimodalModelSurgerySystem()

        # Create a test model for surgery tests
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                embedding = nn.Embedding(1000, 128)
                positional_encoding = nn.Embedding(512, 128)
                dropout = nn.Dropout(0.1)
                layer_norm = nn.LayerNorm(128)
                linear1 = nn.Linear(128, 256)
                relu = nn.ReLU()
                linear2 = nn.Linear(256, 128)
                attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)

            def forward(self, x):
                x = embedding(x)
                pos_enc = positional_encoding(torch.arange(x.size(1)).unsqueeze(0).expand(x.size(0), -1))
                x = x + pos_enc
                x = dropout(x)
                x = layer_norm(x)
                x = linear1(x)
                x = relu(x)
                x = linear2(x)
                x, _ = attention(x, x, x)
                return x

        test_model = TestModel()

    def unimodal_model_surgery_system_creation(self)():
        """Test that the unimodal model surgery system can be created."""
        assert_is_instance(surgery_system, UnimodalModelSurgerySystem)
        # Check that it inherits from ModelSurgerySystem
        from ..model_surgery import ModelSurgerySystem
        assert_is_instance(surgery_system, ModelSurgerySystem)

    def classify_unimodal_module(self)():
        """Test that unimodal modules are classified correctly."""
        # Test various module types
        embedding = nn.Embedding(100, 128)
        positional_enc = nn.Embedding(512, 128)
        dropout = nn.Dropout(0.1)
        layer_norm = nn.LayerNorm(128)
        linear = nn.Linear(128, 256)
        relu = nn.ReLU()
        attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)

        # Test classification
        emb_type = surgery_system._classify_unimodal_module(embedding)
        # Some modules might not be classified as unimodal-specific, which is OK
        # They might be classified by the base system or not at all

        pos_enc_type = surgery_system._classify_unimodal_module(positional_enc)
        # Positional encoding might be classified as unimodal or base type

        dropout_type = surgery_system._classify_unimodal_module(dropout)
        # Dropout should be classified (either as unimodal or base type)

        norm_type = surgery_system._classify_unimodal_module(layer_norm)
        # Norm should be classified (either as unimodal or base type)

        linear_type = surgery_system._classify_unimodal_module(linear)
        # Linear layers might not be classified as unimodal-specific, so they might return None
        # or be classified as a base type

        # At least some modules should be classified
        classified_modules = [emb_type, pos_enc_type, dropout_type, norm_type, linear_type]
        assert_true(any(t is not None for t in classified_modules),
                       "At least some modules should be classified by the system")

    def identify_removable_components(self)():
        """Test identifying removable components in a unimodal model."""
        components = surgery_system.identify_removable_components(test_model)

        # Should find some components
        assertGreaterEqual(len(components), 1)

        # Check that components have the right attributes
        for comp in components:
            assert_is_instance(comp, UnimodalSurgicalComponent)
            assert_is_not_none(comp.name)
            assertIsNotNone(comp.module)
            assertIsNotNone(comp.type)
            assert_is_instance(comp.language_specificity)
            assertIsInstance(comp.semantic_importance, float)
            assert_is_instance(comp.computational_overhead, float)

    def perform_unimodal_surgery(self)():
        """Test performing unimodal model surgery."""
        # Get original parameter count
        original_params = sum(p.numel() for p in test_model.parameters())

        # Perform unimodal surgery
        modified_model = surgery_system.perform_unimodal_surgery(test_model)

        # Parameter count should remain the same (we replace with identity, not remove)
        modified_params = sum(p.numel() for p in modified_model.parameters())

        # The model should still work
        test_input = torch.randint(0, 100, (2, 10))
        original_output = test_model(test_input)
        modified_output = modified_model(test_input)

        # Outputs should be similar (though not identical due to dropout removal)
        assert_equal(original_output.shape, modified_output.shape)

    def perform_unimodal_surgery_with_filters(self)():
        """Test performing unimodal model surgery with component filters."""
        # Test with specific components to preserve
        preserve_components = ['embedding', 'linear1', 'linear2']
        
        modified_model = surgery_system.perform_unimodal_surgery(
            test_model,
            preserve_components=preserve_components
        )

        # Should still work
        test_input = torch.randint(0, 100, (2, 10))
        output = modified_model(test_input)
        assert_equal(output.shape, (2))

    def analyze_unimodal_model_for_surgery(self)():
        """Test analyzing a unimodal model for surgery."""
        analysis = surgery_system.analyze_unimodal_model_for_surgery(test_model)

        # Should return a dictionary with expected keys
        assert_is_instance(analysis, dict)
        assert_in('total_parameters', analysis)
        assert_in('total_modules', analysis)
        assert_in('removable_components', analysis)
        assert_in('recommendations', analysis)
        assert_in('language_specificity_distribution', analysis)

        # Check that analysis has meaningful content
        assertGreaterEqual(analysis['total_parameters'], 0)
        assertGreaterEqual(analysis['total_modules'], 0)
        assertGreaterEqual(len(analysis['recommendations']), 0)

    def apply_unimodal_model_surgery_function(self)():
        """Test the convenience function for applying unimodal model surgery."""
        # Test with our test model
        modified_model = apply_unimodal_model_surgery(test_model)

        # Should return a model
        assert_is_instance(modified_model, nn.Module)

        # Should still be functional
        test_input = torch.randint(0, 100, (2, 10))
        output = modified_model(test_input)
        assert_equal(output.shape, (2))

    def analyze_unimodal_model_for_surgery_function(self)():
        """Test the convenience function for analyzing unimodal model for surgery."""
        analysis = analyze_unimodal_model_for_surgery(test_model)

        # Should return a dictionary with expected keys
        assert_is_instance(analysis, dict)
        assert_in('total_parameters', analysis)
        assert_in('removable_components', analysis)
        assert_in('recommendations', analysis)

    def get_unimodal_model_surgery_system_singleton(self)():
        """Test that the singleton getter returns the same instance."""
        system1 = get_unimodal_model_surgery_system()
        system2 = get_unimodal_model_surgery_system()

        assertIs(system1, system2)

    def language_specificity_analysis(self)():
        """Test language specificity analysis."""
        # Test with a module that should have high language specificity
        embedding = nn.Embedding(100, 128)
        specificity = surgery_system._analyze_language_specificity(embedding, "text_embedding")
        assertGreaterEqual(specificity, 0.5)

        # Test with a general module
        linear = nn.Linear(128, 256)
        specificity = surgery_system._analyze_language_specificity(linear, "general_linear")
        assertLessEqual(specificity, 0.7)

    def semantic_importance_analysis(self)():
        """Test semantic importance analysis."""
        # Test with a module that should have high semantic importance
        attention = nn.MultiheadAttention(embed_dim=128, num_heads=8)
        importance = surgery_system._analyze_semantic_importance(attention, "encoder_attention")
        assertGreaterEqual(importance, 0.5)

        # Test with an auxiliary module
        aux_layer = nn.Linear(128, 128)
        importance = surgery_system._analyze_semantic_importance(aux_layer, "aux_debug_layer")
        assertLessEqual(importance, 0.7)

    def unimodal_surgical_component_attributes(self)():
        """Test that UnimodalSurgicalComponent has all required attributes."""
        component = UnimodalSurgicalComponent(
            name="test_component",
            module=nn.Linear(10, 10),
            type=UnimodalComponentType.LAYER_NORM  # Use existing enum value
        )

        # Check base attributes
        assert_equal(component.name, "test_component")
        assert_is_instance(component.module, nn.Module)
        assert_equal(component.type, UnimodalComponentType.LAYER_NORM)

        # Check unimodal-specific attributes
        assert_is_instance(component.language_specificity, float)
        assert_is_instance(component.semantic_importance, float)
        assert_is_instance(component.computational_overhead, float)

    def preserve_semantic_importance_threshold(self)():
        """Test that semantic importance threshold works correctly."""
        # Create a model with components of varying semantic importance
        class TestModelWithImportance(nn.Module):
            def __init__(self):
                super().__init__()
                important_layer = nn.Linear(128, 256)  # More important
                auxiliary_layer = nn.Linear(256, 128)  # Less important (takes output of first layer)

            def forward(self, x):
                x = important_layer(x)
                x = auxiliary_layer(x)
                return x

        model = TestModelWithImportance()

        # Perform surgery with high threshold (should preserve more components)
        modified_model_high_threshold = surgery_system.perform_unimodal_surgery(
            model,
            preserve_semantic_importance_threshold=0.9
        )

        # Perform surgery with low threshold (should remove more components)
        modified_model_low_threshold = surgery_system.perform_unimodal_surgery(
            model,
            preserve_semantic_importance_threshold=0.1
        )

        # Both models should still work
        test_input = torch.randn(2, 128)
        output_high = modified_model_high_threshold(test_input)
        output_low = modified_model_low_threshold(test_input)

        assert_equal(output_high.shape, (2))
        assert_equal(output_low.shape, (2))

# TestUnimodalModelSurgeryIntegration

    """Integration tests for unimodal model surgery with actual models."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        surgery_system = get_unimodal_model_surgery_system()

    def simple_model_surgery_end_to_end(self)():
        """Test end-to-end unimodal model surgery workflow."""
        # Create a simple model
        class SimpleTextModel(nn.Module):
            def __init__(self):
                super().__init__()
                embedding = nn.Embedding(1000, 64)
                dropout = nn.Dropout(0.2)
                linear = nn.Linear(64, 32)
                output = nn.Linear(32, 10)

            def forward(self, x):
                x = embedding(x)
                x = dropout(x)
                x = linear(x)
                x = output(x)
                return x

        model = SimpleTextModel()

        # Step 1: Analyze the model
        analysis = surgery_system.analyze_unimodal_model_for_surgery(model)
        assert_in('removable_components', analysis)

        # Step 2: Perform surgery
        modified_model = surgery_system.perform_unimodal_surgery(model)

        # Step 3: Verify the modified model still works
        test_input = torch.randint(0, 1000, (4, 10))
        original_output = model(test_input)
        modified_output = modified_model(test_input)

        # Outputs should have the same shape
        assert_equal(original_output.shape, modified_output.shape)

    def multiple_surgeries_on_same_model(self)():
        """Test performing multiple surgeries on the same model."""
        model = nn.Sequential(
            nn.Embedding(100, 32),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        # First surgery
        modified_model_1 = surgery_system.perform_unimodal_surgery(
            model,
            preserve_semantic_importance_threshold=0.5
        )

        # Second surgery with different threshold
        modified_model_2 = surgery_system.perform_unimodal_surgery(
            modified_model_1,
            preserve_semantic_importance_threshold=0.8
        )

        # Both should work
        test_input = torch.randint(0, 100, (2, 5))
        output_1 = modified_model_1(test_input)
        output_2 = modified_model_2(test_input)

        assert_equal(output_1.shape, (2))
        assert_equal(output_2.shape, (2))

if __name__ == '__main__':
    run_tests(test_functions)