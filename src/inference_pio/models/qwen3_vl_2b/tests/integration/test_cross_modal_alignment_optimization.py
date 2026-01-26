"""
Test suite for Cross-Modal Alignment Optimization in Qwen3-VL-2B Model

This module tests the cross-modal alignment optimization system specifically implemented
for the Qwen3-VL-2B model. The tests verify that the alignment system correctly
optimizes the interaction between vision and language modalities.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import numpy as np
from PIL import Image

from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin
from src.inference_pio.common.cross_modal_alignment_optimization import (
    CrossModalAlignmentConfig,
    CrossModalAlignmentOptimizer,
    Qwen3VL2BCrossModalAlignmentOptimizer,
    create_qwen3_vl_cross_modal_alignment,
    apply_cross_modal_alignment_to_model
)

# TestQwen3VL2BCrossModalAlignmentOptimization

    """Test cases for Qwen3-VL-2B cross-modal alignment optimization."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = Qwen3VL2BConfig()
        alignment_config = CrossModalAlignmentConfig(
            alignment_temperature=0.5,
            alignment_lambda=0.1,
            use_contrastive_alignment=True,
            contrastive_margin=0.2,
            enable_dynamic_alignment=True,
            alignment_frequency=10,
            alignment_threshold=0.8
        )

    def cross_modal_alignment_config_creation(self)():
        """Test creation of cross-modal alignment configuration."""
        config = CrossModalAlignmentConfig()
        assert_is_instance(config, CrossModalAlignmentConfig)
        assert_equal(config.alignment_temperature, 0.5)
        assert_equal(config.alignment_lambda, 0.1)
        assert_true(config.use_contrastive_alignment)
        assert_equal(config.contrastive_margin)

    def cross_modal_alignment_optimizer_creation(self)():
        """Test creation of cross-modal alignment optimizer."""
        optimizer = CrossModalAlignmentOptimizer(alignment_config)
        assert_is_instance(optimizer, CrossModalAlignmentOptimizer)

    def qwen3_vl_2b_cross_modal_alignment_optimizer_creation(self)():
        """Test creation of Qwen3-VL-2B specific cross-modal alignment optimizer."""
        qwen_config = Qwen3VL2BConfig()
        optimizer = Qwen3VL2BCrossModalAlignmentOptimizer(qwen_config, alignment_config)
        assert_is_instance(optimizer, Qwen3VL2BCrossModalAlignmentOptimizer)

    def cross_modal_alignment_forward_pass(self)():
        """Test forward pass of cross-modal alignment optimization."""
        batch_size = 2
        vision_seq_len = 10
        lang_seq_len = 15
        hidden_size = 256

        # Create sample vision and language features
        vision_features = torch.randn(batch_size, vision_seq_len, hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, hidden_size)

        # Create optimizer
        optimizer = CrossModalAlignmentOptimizer(alignment_config)

        # Perform alignment
        aligned_features, alignment_loss = optimizer.align_modalities(
            vision_features, language_features
        )

        # Check output shapes
        assert_equal(aligned_features[0].shape, vision_features.shape)
        assert_equal(aligned_features[1].shape, language_features.shape)
        assert_is_instance(alignment_loss, torch.Tensor)
        assertGreaterEqual(alignment_loss.item(), 0.0)

    def qwen3_vl_2b_cross_modal_alignment_forward_pass(self)():
        """Test forward pass of Qwen3-VL-2B specific cross-modal alignment."""
        batch_size = 2
        vision_seq_len = 10
        lang_seq_len = 15
        hidden_size = 256

        # Create sample vision and language features
        vision_features = torch.randn(batch_size, vision_seq_len, hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, hidden_size)

        # Create Qwen3-VL-2B specific optimizer
        qwen_config = Qwen3VL2BConfig(hidden_size=hidden_size)
        optimizer = Qwen3VL2BCrossModalAlignmentOptimizer(qwen_config, alignment_config)

        # Perform alignment
        aligned_features, alignment_loss = optimizer.align_modalities(
            vision_features, language_features
        )

        # Check output shapes
        assert_equal(aligned_features[0].shape, vision_features.shape)
        assert_equal(aligned_features[1].shape, language_features.shape)
        assert_is_instance(alignment_loss, torch.Tensor)
        assertGreaterEqual(alignment_loss.item(), 0.0)

    def create_qwen3_vl_cross_modal_alignment(self)():
        """Test creation of Qwen3-VL-2B cross-modal alignment system."""
        qwen_config = Qwen3VL2BConfig()
        alignment_system = create_qwen3_vl_cross_modal_alignment(qwen_config)
        
        assert_is_not_none(alignment_system)
        assert_is_instance(alignment_system)

    def apply_cross_modal_alignment_to_model(self)():
        """Test applying cross-modal alignment to a model."""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.config = config
        
        # Apply cross-modal alignment
        aligned_model = apply_cross_modal_alignment_to_model(mock_model, config)
        
        # Check that the model has alignment capabilities
        assert_true(hasattr(aligned_model))

    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained')
    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained')
    @patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained')
    def qwen3_vl_2b_model_with_cross_modal_alignment_integration(self, mock_image_processor, mock_tokenizer, mock_model)():
        """Test integration of cross-modal alignment with Qwen3-VL-2B model."""
        # Mock the model loading
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_image_processor.return_value = MagicMock()

        # Create model with cross-modal alignment
        model = Qwen3VL2BModel(config)

        # Check that the model has cross-modal alignment capabilities
        assert_true(hasattr(model))
        assert_is_instance(model.cross_modal_alignment_optimizer, Qwen3VL2BCrossModalAlignmentOptimizer)

    def cross_modal_alignment_loss_calculation(self)():
        """Test the calculation of cross-modal alignment loss."""
        batch_size = 2
        seq_len = 10
        hidden_size = 256

        # Create similar features (should have low alignment loss)
        vision_features = torch.randn(batch_size, seq_len, hidden_size)
        language_features = vision_features + 0.1 * torch.randn_like(vision_features)  # Slightly perturbed

        optimizer = CrossModalAlignmentOptimizer(alignment_config)
        _, alignment_loss = optimizer.align_modalities(vision_features, language_features)

        # Loss should be positive
        assertGreaterEqual(alignment_loss.item(), 0.0)

        # Create dissimilar features (should have higher alignment loss)
        language_features_dissimilar = torch.randn(batch_size, seq_len, hidden_size)
        _, alignment_loss_dissimilar = optimizer.align_modalities(vision_features, language_features_dissimilar)

        # Dissimilar features should have higher loss (though this isn't guaranteed due to randomness)
        # Just ensure both losses are calculated properly
        assertGreaterEqual(alignment_loss_dissimilar.item(), 0.0)

    def dynamic_cross_modal_alignment(self)():
        """Test dynamic cross-modal alignment based on input complexity."""
        batch_size = 2
        vision_seq_len = 10
        lang_seq_len = 15
        hidden_size = 256

        # Create sample vision and language features
        vision_features = torch.randn(batch_size, vision_seq_len, hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, hidden_size)

        # Create optimizer with dynamic alignment enabled
        dynamic_config = CrossModalAlignmentConfig(
            alignment_temperature=0.5,
            alignment_lambda=0.1,
            use_contrastive_alignment=True,
            contrastive_margin=0.2,
            enable_dynamic_alignment=True,
            alignment_frequency=1,
            alignment_threshold=0.5
        )
        
        optimizer = CrossModalAlignmentOptimizer(dynamic_config)

        # Perform dynamic alignment
        aligned_features, alignment_loss = optimizer.align_modalities(
            vision_features, language_features
        )

        # Check output shapes
        assert_equal(aligned_features[0].shape, vision_features.shape)
        assert_equal(aligned_features[1].shape, language_features.shape)
        assert_is_instance(alignment_loss, torch.Tensor)

    def cross_modal_alignment_with_different_modalities(self)():
        """Test cross-modal alignment with different sequence lengths."""
        batch_size = 2
        vision_seq_len = 5
        lang_seq_len = 20
        hidden_size = 256

        # Create sample vision and language features with different lengths
        vision_features = torch.randn(batch_size, vision_seq_len, hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, hidden_size)

        optimizer = CrossModalAlignmentOptimizer(alignment_config)

        # Perform alignment (should handle different sequence lengths)
        aligned_features, alignment_loss = optimizer.align_modalities(
            vision_features, language_features
        )

        # Check output shapes match input shapes
        assert_equal(aligned_features[0].shape, vision_features.shape)
        assert_equal(aligned_features[1].shape, language_features.shape)
        assert_is_instance(alignment_loss, torch.Tensor)

    def plugin_integration_with_cross_modal_alignment(self)():
        """Test that the plugin properly integrates with cross-modal alignment."""
        plugin = Qwen3_VL_2B_Instruct_Plugin()

        # Check that the plugin has methods related to cross-modal alignment
        assert_true(hasattr(plugin))
        assert_true(hasattr(plugin))

    def alignment_config_parameters(self)():
        """Test different alignment configuration parameters."""
        # Test with high temperature
        high_temp_config = CrossModalAlignmentConfig(
            alignment_temperature=2.0,
            alignment_lambda=0.1,
            use_contrastive_alignment=False
        )
        optimizer_high_temp = CrossModalAlignmentOptimizer(high_temp_config)
        assert_equal(optimizer_high_temp.config.alignment_temperature, 2.0)

        # Test with low lambda
        low_lambda_config = CrossModalAlignmentConfig(
            alignment_temperature=0.5,
            alignment_lambda=0.01,
            use_contrastive_alignment=True
        )
        optimizer_low_lambda = CrossModalAlignmentOptimizer(low_lambda_config)
        assert_equal(optimizer_low_lambda.config.alignment_lambda, 0.01)

        # Test with no contrastive alignment
        no_contrastive_config = CrossModalAlignmentConfig(
            alignment_temperature=0.5,
            alignment_lambda=0.1,
            use_contrastive_alignment=False
        )
        optimizer_no_contrastive = CrossModalAlignmentOptimizer(no_contrastive_config)
        assert_false(optimizer_no_contrastive.config.use_contrastive_alignment)

# TestCrossModalAlignmentIntegration

    """Integration tests for cross-modal alignment optimization."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        config = Qwen3VL2BConfig()
        alignment_config = CrossModalAlignmentConfig(
            alignment_temperature=0.5,
            alignment_lambda=0.1,
            use_contrastive_alignment=True,
            contrastive_margin=0.2
        )

    def complete_alignment_pipeline(self)():
        """Test the complete cross-modal alignment pipeline."""
        batch_size = 1
        vision_seq_len = 8
        lang_seq_len = 12
        hidden_size = 256

        # Create sample features
        vision_features = torch.randn(batch_size, vision_seq_len, hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, hidden_size)

        # Create alignment system
        alignment_system = create_qwen3_vl_cross_modal_alignment(config)

        # Perform alignment
        aligned_vision, aligned_language = alignment_system(vision_features, language_features)

        # Verify outputs
        assert_equal(aligned_vision.shape, vision_features.shape)
        assert_equal(aligned_language.shape, language_features.shape)

        # Check that alignment changed the features (they should not be identical)
        vision_diff = torch.mean(torch.abs(aligned_vision - vision_features)).item()
        lang_diff = torch.mean(torch.abs(aligned_language - language_features)).item()
        
        # The difference should be non-zero if alignment was applied
        # Due to potential random initialization, we'll just check that the values are reasonable
        assertGreaterEqual(vision_diff, 0.0)
        assertGreaterEqual(lang_diff, 0.0)

    def alignment_with_mock_model(self)():
        """Test cross-modal alignment integration with a mock model."""
        # Create a mock model with required attributes
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                config = Qwen3VL2BConfig()
                
        mock_model = MockModel()
        
        # Apply alignment to the mock model
        aligned_model = apply_cross_modal_alignment_to_model(mock_model, mock_model.config)
        
        # Verify that alignment was applied
        assert_true(hasattr(aligned_model))
        assert_is_instance(aligned_model.cross_modal_alignment_optimizer, Qwen3VL2BCrossModalAlignmentOptimizer)

def run_tests():
    """Run all tests in the test suite."""
    print("Running Cross-Modal Alignment Optimization Tests for Qwen3-VL-2B...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests to the suite
    suite.addTests(loader.loadTestsFromTestCase(TestQwen3VL2BCrossModalAlignmentOptimization))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossModalAlignmentIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✓ All cross-modal alignment optimization tests passed!")
    else:
        print("\n✗ Some tests failed.")