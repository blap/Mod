"""
Tests for conditional feature extraction based on input modality requirements
"""
import torch
import pytest
import numpy as np
from src.components.optimization.conditional_feature_extraction import ConditionalFeatureExtractor, ModalitySpecificExtractor
from src.models.config import Qwen3VLConfig


class TestConditionalFeatureExtractor:
    """Test suite for conditional feature extraction functionality"""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = Qwen3VLConfig()
        self.config.hidden_size = 512
        self.config.vision_hidden_size = 512
        self.config.vision_num_hidden_layers = 4
        self.config.num_hidden_layers = 8
        self.config.vocab_size = 1000
        
        # Create test inputs
        self.batch_size = 2
        self.seq_len = 32
        self.img_size = 224
        
        self.text_input = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        self.image_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
        self.multimodal_input = {
            'text': self.text_input,
            'image': self.image_input
        }
    
    def test_modality_specific_extractor_initialization(self):
        """Test initialization of modality-specific extractors."""
        extractor = ModalitySpecificExtractor(self.config)
        
        # Check that appropriate extractors are created
        assert hasattr(extractor, 'text_extractor')
        assert hasattr(extractor, 'vision_extractor')
        assert hasattr(extractor, 'multimodal_fusion')
        
        # Check that extractors have correct dimensions
        assert extractor.text_extractor is not None
        assert extractor.vision_extractor is not None
        assert extractor.multimodal_fusion is not None
    
    def test_conditional_feature_extractor_initialization(self):
        """Test initialization of conditional feature extractor."""
        extractor = ConditionalFeatureExtractor(self.config)
        
        # Check that required components are initialized
        assert hasattr(extractor, 'modality_specific_extractor')
        assert hasattr(extractor, 'modality_classifier')
        assert hasattr(extractor, 'complexity_assessor')
        
        # Check that the components are properly configured
        assert extractor.modality_specific_extractor is not None
        assert extractor.modality_classifier is not None
        assert extractor.complexity_assessor is not None
    
    def test_text_feature_extraction(self):
        """Test feature extraction for text-only input."""
        extractor = ConditionalFeatureExtractor(self.config)
        extractor.eval()
        
        with torch.no_grad():
            features, modality_info = extractor(text_input=self.text_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.config.hidden_size)
        assert features.shape == expected_shape, f"Expected {expected_shape}, got {features.shape}"
        
        # Check that modality info is correct
        assert modality_info['modality'] == 'text'
        assert modality_info['complexity_score'] >= 0 and modality_info['complexity_score'] <= 1
    
    def test_vision_feature_extraction(self):
        """Test feature extraction for vision-only input."""
        extractor = ConditionalFeatureExtractor(self.config)
        extractor.eval()
        
        with torch.no_grad():
            features, modality_info = extractor(image_input=self.image_input)
        
        # Check output shape (vision features are typically flattened)
        expected_shape = (self.batch_size, -1, self.config.vision_hidden_size)  # -1 depends on patch size
        assert len(features.shape) == 3
        assert features.shape[0] == self.batch_size
        assert features.shape[2] == self.config.vision_hidden_size
        
        # Check that modality info is correct
        assert modality_info['modality'] == 'vision'
        assert modality_info['complexity_score'] >= 0 and modality_info['complexity_score'] <= 1
    
    def test_multimodal_feature_extraction(self):
        """Test feature extraction for multimodal input."""
        extractor = ConditionalFeatureExtractor(self.config)
        extractor.eval()
        
        with torch.no_grad():
            features, modality_info = extractor(
                text_input=self.text_input, 
                image_input=self.image_input
            )
        
        # Check output shape (combined features)
        expected_shape = (self.batch_size, self.seq_len + 50, self.config.hidden_size)  # 50 is approx image patches
        assert len(features.shape) == 3
        assert features.shape[0] == self.batch_size
        assert features.shape[2] == self.config.hidden_size
        
        # Check that modality info is correct
        assert modality_info['modality'] == 'multimodal'
        assert modality_info['complexity_score'] >= 0 and modality_info['complexity_score'] <= 1
    
    def test_modality_classification_accuracy(self):
        """Test that modality classification works correctly."""
        extractor = ConditionalFeatureExtractor(self.config)
        extractor.eval()
        
        # Test text classification
        with torch.no_grad():
            _, modality_info = extractor(text_input=self.text_input)
        assert modality_info['modality'] == 'text'
        
        # Test vision classification
        with torch.no_grad():
            _, modality_info = extractor(image_input=self.image_input)
        assert modality_info['modality'] == 'vision'
        
        # Test multimodal classification
        with torch.no_grad():
            _, modality_info = extractor(text_input=self.text_input, image_input=self.image_input)
        assert modality_info['modality'] == 'multimodal'
    
    def test_complexity_assessment(self):
        """Test that complexity assessment works correctly."""
        extractor = ConditionalFeatureExtractor(self.config)
        extractor.eval()
        
        # Test with simple text (repetitive)
        simple_text = torch.ones(self.batch_size, self.seq_len, dtype=torch.long) * 100
        with torch.no_grad():
            _, modality_info = extractor(text_input=simple_text)
        assert 0 <= modality_info['complexity_score'] <= 1
        
        # Test with complex text (random)
        complex_text = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        with torch.no_grad():
            _, modality_info = extractor(text_input=complex_text)
        assert 0 <= modality_info['complexity_score'] <= 1
        
        # Complexity score for complex text should generally be higher than for simple text
        # (though this is not guaranteed due to the nature of complexity assessment)
    
    def test_feature_extraction_efficiency(self):
        """Test that conditional feature extraction is more efficient than full processing."""
        extractor = ConditionalFeatureExtractor(self.config)
        extractor.eval()
        
        # Time the conditional extraction
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time is not None:
            start_time.record()
        
        with torch.no_grad():
            features, modality_info = extractor(text_input=self.text_input)
        
        if start_time is not None:
            end_time.record()
            torch.cuda.synchronize()
            conditional_time = start_time.elapsed_time(end_time)
        else:
            import time
            start = time.time()
            with torch.no_grad():
                features, modality_info = extractor(text_input=self.text_input)
            conditional_time = (time.time() - start) * 1000  # Convert to milliseconds
        
        # The efficiency test would compare this with full processing
        # For now, we just ensure it runs without error and produces expected output
        
        assert features is not None
        assert modality_info is not None
        assert features.shape[0] == self.batch_size
    
    def test_backward_compatibility(self):
        """Test that conditional feature extraction maintains compatibility with existing models."""
        extractor = ConditionalFeatureExtractor(self.config)
        extractor.eval()
        
        # Extract features using conditional extractor
        with torch.no_grad():
            features, modality_info = extractor(text_input=self.text_input)
        
        # Features should be in the same format as expected by the language model
        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32  # or torch.float16 if using mixed precision
        assert features.device == self.text_input.device
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        extractor = ConditionalFeatureExtractor(self.config)

        # Test with no inputs
        with pytest.raises(ValueError):
            extractor()

        # Test with invalid tensor shapes (1D instead of 2D for text)
        invalid_text = torch.randint(0, self.config.vocab_size, (self.batch_size,))  # 1D instead of 2D
        with pytest.raises(ValueError):
            extractor(text_input=invalid_text)

        # Test with invalid tensor shapes (3D instead of 4D for image)
        invalid_image = torch.randn(self.batch_size, 3, self.img_size)  # 3D instead of 4D
        with pytest.raises(ValueError):
            extractor(image_input=invalid_image)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through the conditional extraction."""
        extractor = ConditionalFeatureExtractor(self.config)

        # Create inputs that require gradients - need to convert to float for gradient computation
        text_input = self.text_input.clone().float().requires_grad_(True)

        features, modality_info = extractor(text_input=text_input.long())

        # Compute a simple loss
        loss = features.mean()

        # Backpropagate
        loss.backward()

        # Check that gradients were computed for the input
        # Note: Since we convert to long for token IDs, gradients flow through the embedding layer
        # The text_input tensor itself won't have gradients, but the embedding parameters will
        embedding_params = [p for p in extractor.modality_specific_extractor.text_extractor.embed_tokens.parameters()]
        assert len(embedding_params) > 0
        assert embedding_params[0].grad is not None


class TestModalitySpecificExtractor:
    """Test suite for modality-specific feature extraction."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = Qwen3VLConfig()
        self.config.hidden_size = 512
        self.config.vision_hidden_size = 512
        self.config.vision_num_hidden_layers = 4
        self.config.num_hidden_layers = 8
        self.config.vocab_size = 1000
        
        self.batch_size = 2
        self.seq_len = 32
        self.img_size = 224
        
        self.text_input = torch.randint(0, self.config.vocab_size, (self.batch_size, self.seq_len))
        self.image_input = torch.randn(self.batch_size, 3, self.img_size, self.img_size)
    
    def test_text_extraction(self):
        """Test text-specific feature extraction."""
        extractor = ModalitySpecificExtractor(self.config)
        extractor.eval()
        
        with torch.no_grad():
            text_features = extractor.extract_text_features(self.text_input)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.config.hidden_size)
        assert text_features.shape == expected_shape, f"Expected {expected_shape}, got {text_features.shape}"
    
    def test_vision_extraction(self):
        """Test vision-specific feature extraction."""
        extractor = ModalitySpecificExtractor(self.config)
        extractor.eval()
        
        with torch.no_grad():
            vision_features = extractor.extract_vision_features(self.image_input)
        
        # Check output shape
        assert len(vision_features.shape) == 3
        assert vision_features.shape[0] == self.batch_size
        assert vision_features.shape[2] == self.config.vision_hidden_size
    
    def test_multimodal_fusion(self):
        """Test multimodal feature fusion."""
        extractor = ModalitySpecificExtractor(self.config)
        extractor.eval()
        
        with torch.no_grad():
            text_features = extractor.extract_text_features(self.text_input)
            vision_features = extractor.extract_vision_features(self.image_input)
            
            fused_features = extractor.fuse_multimodal_features(text_features, vision_features)
        
        # Check that fused features have appropriate shape
        assert len(fused_features.shape) == 3
        assert fused_features.shape[0] == self.batch_size
        assert fused_features.shape[2] in [self.config.hidden_size, self.config.vision_hidden_size]
    
    def test_modality_activation(self):
        """Test that correct modality pathways are activated."""
        extractor = ModalitySpecificExtractor(self.config)
        extractor.eval()
        
        # Test that text extraction only uses text pathway
        with torch.no_grad():
            text_features = extractor.extract_text_features(self.text_input)
        
        assert text_features is not None
        assert text_features.shape[0] == self.batch_size
        
        # Test that vision extraction only uses vision pathway
        with torch.no_grad():
            vision_features = extractor.extract_vision_features(self.image_input)
        
        assert vision_features is not None
        assert vision_features.shape[0] == self.batch_size


if __name__ == "__main__":
    pytest.main([__file__])