"""
Integration Test for Cross-Modal Alignment Optimization in Qwen3-VL-2B Model

This module tests the integration of cross-modal alignment optimization with the Qwen3-VL-2B model.
It verifies that the alignment system is properly implemented and works correctly with the model architecture.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import numpy as np
from PIL import Image

from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin
from src.inference_pio.common.cross_modal_alignment_optimization import (
    Qwen3VL2BCrossModalAlignmentOptimizer,
    CrossModalAlignmentManager,
    create_qwen3_vl_cross_modal_alignment,
    apply_cross_modal_alignment_to_model,
    get_cross_modal_alignment_report
)

# TestQwen3VL2BCrossModalAlignmentIntegration

    """Integration tests for cross-modal alignment optimization with Qwen3-VL-2B model."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = Qwen3VL2BConfig()
        # Disable actual model loading to avoid dependency on local model files
        config.model_path = "dummy_path"
        
    def model_creation_with_cross_modal_alignment(self)():
        """Test that Qwen3-VL-2B model can be created with cross-modal alignment enabled."""
        # Mock the model loading to avoid actual model initialization
        with patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained'):
            
            model = Qwen3VL2BModel(config)
            
            # Check that the model has cross-modal alignment attributes
            assert_true(hasattr(model))
            assert_true(hasattr(model))
            assert_true(hasattr(model))
            assert_true(hasattr(model))
            assert_true(hasattr(model))
            
            # Check that the model has the alignment method
            assert_true(hasattr(model._model))
            assert_true(hasattr(model._model))
            
            print("✓ Qwen3-VL-2B model created with cross-modal alignment integration")

    def cross_modal_alignment_manager_creation(self)():
        """Test creation of cross-modal alignment manager for Qwen3-VL-2B."""
        alignment_manager = create_qwen3_vl_cross_modal_alignment(config)
        
        assert_is_instance(alignment_manager, CrossModalAlignmentManager)
        assert_true(hasattr(alignment_manager))
        assert_in('qwen3_vl_specific', alignment_manager.alignment_methods)
        assert_in('attention', alignment_manager.alignment_methods)
        assert_in('contrastive', alignment_manager.alignment_methods)
        assert_in('learned_projection', alignment_manager.alignment_methods)
        assert_in('similarity_based', alignment_manager.alignment_methods)
        
        print("✓ Cross-modal alignment manager created successfully")

    def qwen3_vl_specific_alignment_kernel(self)():
        """Test Qwen3-VL-2B specific alignment kernel."""
        from src.inference_pio.common.cross_modal_alignment_optimization import CrossModalAlignmentConfig
        
        alignment_config = CrossModalAlignmentConfig(
            alignment_temperature=0.5,
            alignment_lambda=0.1,
            use_contrastive_alignment=True,
            contrastive_margin=0.2,
            enable_dynamic_alignment=True,
            alignment_frequency=10,
            alignment_threshold=0.8,
            use_attention_alignment=True,
            use_learned_alignment=True,
            alignment_projection_dim=config.hidden_size,
            enable_similarity_alignment=True,
            similarity_method='cosine'
        )
        
        alignment_kernel = Qwen3VL2BCrossModalAlignmentOptimizer(config, alignment_config)
        
        assert_is_instance(alignment_kernel, Qwen3VL2BCrossModalAlignmentOptimizer)
        assert_true(hasattr(alignment_kernel))
        assert_true(hasattr(alignment_kernel))
        assert_true(hasattr(alignment_kernel))
        assert_true(hasattr(alignment_kernel))
        assert_true(hasattr(alignment_kernel))
        assert_true(hasattr(alignment_kernel))
        assert_true(hasattr(alignment_kernel))
        
        print("✓ Qwen3-VL-2B specific alignment kernel created successfully")

    def cross_modal_alignment_forward_pass(self)():
        """Test forward pass of cross-modal alignment."""
        from src.inference_pio.common.cross_modal_alignment_optimization import CrossModalAlignmentConfig
        
        alignment_config = CrossModalAlignmentConfig(
            alignment_temperature=0.5,
            alignment_lambda=0.1,
            use_contrastive_alignment=True,
            contrastive_margin=0.2,
            enable_dynamic_alignment=True,
            alignment_frequency=10,
            alignment_threshold=0.8,
            use_attention_alignment=True,
            use_learned_alignment=True,
            alignment_projection_dim=config.hidden_size,
            enable_similarity_alignment=True,
            similarity_method='cosine'
        )
        
        alignment_kernel = Qwen3VL2BCrossModalAlignmentOptimizer(config, alignment_config)
        
        # Create sample vision and language features
        batch_size = 2
        vision_seq_len = 10
        lang_seq_len = 15
        hidden_size = config.hidden_size  # Use the Qwen3-VL-2B hidden size

        vision_features = torch.randn(batch_size, vision_seq_len, hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, hidden_size)
        
        # Perform alignment
        (aligned_vision, aligned_language), alignment_loss = alignment_kernel.align_modalities(
            vision_features, language_features
        )
        
        # Check output shapes
        assert_equal(aligned_vision.shape, vision_features.shape)
        assert_equal(aligned_language.shape, language_features.shape)
        assert_is_instance(alignment_loss, torch.Tensor)
        assertGreaterEqual(alignment_loss.item(), 0.0)
        
        print("✓ Cross-modal alignment forward pass completed successfully")

    def apply_cross_modal_alignment_to_model(self)():
        """Test applying cross-modal alignment to a model."""
        # Create a mock model to test the application
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                config = self
                
        mock_model = MockModel()
        
        # Apply cross-modal alignment to the mock model
        aligned_model = apply_cross_modal_alignment_to_model(mock_model, config)
        
        # Check that the model has alignment attributes
        assert_true(hasattr(aligned_model))
        assert_true(hasattr(aligned_model))
        assert_true(hasattr(aligned_model))
        
        print("✓ Cross-modal alignment applied to model successfully")

    def plugin_integration_with_cross_modal_alignment(self)():
        """Test that the plugin integrates properly with cross-modal alignment."""
        plugin = Qwen3_VL_2B_Instruct_Plugin()
        
        # Check that the plugin has cross-modal alignment methods
        assert_true(hasattr(plugin))
        assert_true(hasattr(plugin))
        
        print("✓ Plugin has cross-modal alignment integration methods")

    def cross_modal_alignment_report_generation(self)():
        """Test generating cross-modal alignment report."""
        # Create a mock model to test the report generation
        
            def __init__(self):
                config = Qwen3VL2BConfig()
                
        mock_model = MockModel()
        
        # Generate report
        report = get_cross_modal_alignment_report(mock_model, mock_model.config)
        
        # Check report structure
        assert_in('model_type', report)
        assert_in('optimization_type', report)
        assert_in('alignment_methods_registered', report)
        assert_in('alignment_enabled', report)
        assert_in('alignment_config', report)
        
        assert_equal(report['model_type'], 'Qwen3-VL-2B')
        assert_equal(report['optimization_type'], 'Cross-Modal Alignment')
        assert_true(report['alignment_enabled'])
        
        print("✓ Cross-modal alignment report generated successfully")

    def alignment_method_selection(self)():
        """Test alignment method selection based on input complexity."""
        # Mock the model loading to avoid actual model initialization
        with patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained')), \
             patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained'):
            
            model = Qwen3VL2BModel(config)
            
            # Test method selection
            simple_method = model._select_alignment_method("simple")
            medium_method = model._select_alignment_method("medium")
            complex_method = model._select_alignment_method("complex")
            
            assert_equal(simple_method, "learned_projection")
            assert_equal(medium_method, "qwen3_vl_specific")
            assert_equal(complex_method, "multi_scale")
            
            print("✓ Alignment method selection working correctly")

    def cross_modal_alignment_with_different_modalities(self)():
        """Test cross-modal alignment with different sequence lengths."""
        from src.inference_pio.common.cross_modal_alignment_optimization import CrossModalAlignmentConfig

        alignment_config = CrossModalAlignmentConfig(
            alignment_temperature=0.5,
            alignment_lambda=0.1,
            use_contrastive_alignment=True,
            contrastive_margin=0.2,
            enable_dynamic_alignment=True,
            alignment_frequency=10,
            alignment_threshold=0.8,
            use_attention_alignment=True,
            use_learned_alignment=True,
            alignment_projection_dim=config.hidden_size,
            enable_similarity_alignment=True,
            similarity_method='cosine'
        )

        alignment_kernel = Qwen3VL2BCrossModalAlignmentOptimizer(config, alignment_config)

        # Create sample features with different sequence lengths
        batch_size = 1
        vision_seq_len = 5
        lang_seq_len = 20
        hidden_size = config.hidden_size  # Use the Qwen3-VL-2B hidden size

        vision_features = torch.randn(batch_size, vision_seq_len, hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, hidden_size)

        # Perform alignment (should handle different sequence lengths)
        (aligned_vision, aligned_language), alignment_loss = alignment_kernel.align_modalities(
            vision_features, language_features
        )

        # Check output shapes match input shapes
        assert_equal(aligned_vision.shape, vision_features.shape)
        assert_equal(aligned_language.shape, language_features.shape)
        assert_is_instance(alignment_loss, torch.Tensor)

        print("✓ Cross-modal alignment handles different sequence lengths correctly")

    def alignment_quality_evaluation(self)():
        """Test alignment quality evaluation."""
        from src.inference_pio.common.cross_modal_alignment_optimization import CrossModalAlignmentManager, CrossModalAlignmentConfig

        alignment_manager = CrossModalAlignmentManager(config, CrossModalAlignmentConfig())
        
        # Create sample features
        batch_size = 2
        seq_len = 10
        hidden_size = config.hidden_size  # Use the Qwen3-VL-2B hidden size

        original_vision = torch.randn(batch_size, seq_len, hidden_size)
        original_language = torch.randn(batch_size, seq_len, hidden_size)

        # Create slightly modified aligned features
        aligned_vision = original_vision + 0.1 * torch.randn_like(original_vision)
        aligned_language = original_language + 0.1 * torch.randn_like(original_language)
        
        # Evaluate alignment quality
        quality_metrics = alignment_manager.evaluate_alignment_quality(
            original_vision, original_language, aligned_vision, aligned_language
        )
        
        # Check metrics structure
        assert_in('vision_similarity', quality_metrics)
        assert_in('language_similarity', quality_metrics)
        assert_in('cross_modal_similarity', quality_metrics)
        assert_in('alignment_improvement', quality_metrics)
        assert_in('quality_score', quality_metrics)
        
        print("✓ Alignment quality evaluation working correctly")

# TestCrossModalAlignmentPerformance

    """Performance tests for cross-modal alignment optimization."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = Qwen3VL2BConfig()
        # Disable actual model loading to avoid dependency on local model files
        config.model_path = "dummy_path"

    def alignment_performance_with_batching(self)():
        """Test alignment performance with different batch sizes."""
        from src.inference_pio.common.cross_modal_alignment_optimization import CrossModalAlignmentConfig
        
        alignment_config = CrossModalAlignmentConfig(
            alignment_temperature=0.5,
            alignment_lambda=0.1,
            use_contrastive_alignment=True,
            contrastive_margin=0.2,
            enable_dynamic_alignment=True,
            alignment_frequency=10,
            alignment_threshold=0.8,
            use_attention_alignment=True,
            use_learned_alignment=True,
            alignment_projection_dim=config.hidden_size,
            enable_similarity_alignment=True,
            similarity_method='cosine'
        )
        
        alignment_kernel = Qwen3VL2BCrossModalAlignmentOptimizer(config, alignment_config)
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4, 8]:
            vision_features = torch.randn(batch_size, 10, config.hidden_size)
            language_features = torch.randn(batch_size, 15, config.hidden_size)

            (aligned_vision, aligned_language), alignment_loss = alignment_kernel.align_modalities(
                vision_features, language_features
            )

            assert_equal(aligned_vision.shape[0], batch_size)
            assert_equal(aligned_language.shape[0], batch_size)
            assertGreaterEqual(alignment_loss.item(), 0.0)
        
        print("✓ Cross-modal alignment performs correctly with different batch sizes")

    def alignment_memory_efficiency(self)():
        """Test memory efficiency of alignment operations."""
        from src.inference_pio.common.cross_modal_alignment_optimization import CrossModalAlignmentConfig
        
        alignment_config = CrossModalAlignmentConfig(
            alignment_temperature=0.5,
            alignment_lambda=0.1,
            use_contrastive_alignment=True,
            contrastive_margin=0.2,
            enable_dynamic_alignment=True,
            alignment_frequency=10,
            alignment_threshold=0.8,
            use_attention_alignment=True,
            use_learned_alignment=True,
            alignment_projection_dim=config.hidden_size,
            enable_similarity_alignment=True,
            similarity_method='cosine'
        )
        
        alignment_kernel = Qwen3VL2BCrossModalAlignmentOptimizer(config, alignment_config)
        
        # Create moderately sized tensors
        batch_size = 2
        vision_seq_len = 32
        lang_seq_len = 64
        hidden_size = config.hidden_size  # Use the Qwen3-VL-2B hidden size

        vision_features = torch.randn(batch_size, vision_seq_len, hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, hidden_size)
        
        # Check initial memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Perform alignment
        (aligned_vision, aligned_language), alignment_loss = alignment_kernel.align_modalities(
            vision_features, language_features
        )
        
        # Check memory usage after alignment
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # The memory increase should be reasonable (less than 10x the input size)
        input_memory = vision_features.numel() * vision_features.element_size() + \
                      language_features.numel() * language_features.element_size()
        output_memory = aligned_vision.numel() * aligned_vision.element_size() + \
                       aligned_language.numel() * aligned_language.element_size()
        expected_memory_increase = (input_memory + output_memory) * 2  # Account for intermediate computations
        
        if torch.cuda.is_available():
            memory_increase = final_memory - initial_memory
            assertLessEqual(memory_increase, expected_memory_increase * 10,
                               f"Memory usage increased too much: {memory_increase} > {expected_memory_increase * 10}")
        
        print("✓ Cross-modal alignment memory efficiency verified")

def run_integration_tests():
    """Run all integration tests for cross-modal alignment optimization."""
    print("Running Cross-Modal Alignment Integration Tests for Qwen3-VL-2B...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests to the suite
    suite.addTests(loader.loadTestsFromTestCase(TestQwen3VL2BCrossModalAlignmentIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossModalAlignmentPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nIntegration Tests Summary:")
    print(f"- Tests run: {result.testsRun}")
    print(f"- Failures: {len(result.failures)}")
    print(f"- Errors: {len(result.errors)}")
    print(f"- Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_integration_tests()
    if success:
        print("\n✓ All cross-modal alignment integration tests passed!")
    else:
        print("\n✗ Some cross-modal alignment integration tests failed.")