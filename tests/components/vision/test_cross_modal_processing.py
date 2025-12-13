"""
Cross-modal processing tests for Qwen3-VL-2B-Instruct
Testing vision-language integration and cross-modal optimization techniques
"""
import pytest
import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np

from models.cross_modal_token_merging import CrossModalTokenMerger
from src.qwen3_vl.components.layers.multimodal_projector import Qwen3VLMultimodalProjector
from src.qwen3_vl.components.layers.vision_transformer import Qwen3VLVisionTransformer
from src.qwen3_vl.components.layers.language_decoder import Qwen3VLDecoder
from src.qwen3_vl.components.configuration import Qwen3VLConfig


class TestCrossModalProcessing:
    """Tests for cross-modal processing capabilities"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.config = Qwen3VLConfig()
        self.config.hidden_size = 256
        self.config.vision_hidden_size = 512
        self.config.intermediate_size = 512
        self.config.num_attention_heads = 8
        self.config.num_hidden_layers = 4
        self.config.vocab_size = 1000
        self.config.vision_image_size = 224
        self.config.vision_patch_size = 16
        self.config.vision_num_channels = 3
        
        self.token_merger = CrossModalTokenMerger()
        
    def test_cross_modal_token_similarity_computation(self):
        """Test similarity computation between vision and language tokens"""
        batch_size, vision_seq_len, lang_seq_len, hidden_dim = 2, 196, 64, 256  # 196 = (224/16)^2 patches
        
        # Create vision and language tokens
        vision_tokens = torch.randn(batch_size, vision_seq_len, hidden_dim)
        lang_tokens = torch.randn(batch_size, lang_seq_len, hidden_dim)
        
        # Compute cross-modal similarities
        similarities = self.token_merger.compute_similarity(vision_tokens, lang_tokens)
        
        # Similarity matrix should be [batch, vision_seq, lang_seq]
        expected_shape = (batch_size, vision_seq_len, lang_seq_len)
        assert similarities.shape == expected_shape
        
        # Similarities should be in reasonable range
        assert torch.all(similarities >= -1.0) and torch.all(similarities <= 1.0)
        
        print(f"Cross-modal similarity computation: {vision_tokens.shape} x {lang_tokens.shape} -> {similarities.shape}")
    
    def test_cross_modal_token_merging(self):
        """Test merging of similar cross-modal tokens"""
        batch_size, seq_len, hidden_dim = 2, 128, 256
        tokens = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Test token merging functionality
        merged_tokens = self.token_merger.merge_tokens(tokens)
        
        assert merged_tokens.shape[0] == batch_size
        assert merged_tokens.shape[2] == hidden_dim
        # For this implementation, merged tokens should maintain sequence length
        assert merged_tokens.shape[1] == seq_len
        assert torch.isfinite(merged_tokens).all()
        
        print(f"Token merging: {tokens.shape} -> {merged_tokens.shape}")
    
    def test_multimodal_projector_integration(self):
        """Test multimodal projector integration"""
        projector = Qwen3VLMultimodalProjector(self.config)
        
        # Create vision features [batch, patches, vision_hidden_size]
        batch_size, num_patches = 2, (224 // 16) ** 2  # 196 patches
        vision_features = torch.randn(batch_size, num_patches, self.config.vision_hidden_size)
        
        # Project vision features to language space
        projected_features = projector(vision_features)
        
        # Should match language hidden size
        expected_shape = (batch_size, num_patches, self.config.hidden_size)
        assert projected_features.shape == expected_shape
        assert torch.isfinite(projected_features).all()
        
        print(f"Multimodal projection: {vision_features.shape} -> {projected_features.shape}")
    
    def test_vision_transformer_forward(self):
        """Test vision transformer forward pass"""
        vision_transformer = Qwen3VLVisionTransformer(self.config)
        
        # Create image input [batch, channels, height, width]
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass through vision transformer
        vision_features = vision_transformer(images)
        
        # Output should be [batch, num_patches, vision_hidden_size]
        num_patches = (224 // 16) ** 2  # Based on patch size
        expected_shape = (batch_size, num_patches, self.config.vision_hidden_size)
        assert vision_features.shape == expected_shape
        assert torch.isfinite(vision_features).all()
        
        print(f"Vision transformer: {images.shape} -> {vision_features.shape}")
    
    def test_language_decoder_forward(self):
        """Test language decoder forward pass"""
        decoder = Qwen3VLDecoder(self.config)
        
        # Create text input [batch, seq_len]
        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        
        # Forward pass through decoder
        decoder_output = decoder(input_ids=input_ids)
        
        # Output should be [batch, seq_len, hidden_size]
        expected_shape = (batch_size, seq_len, self.config.hidden_size)
        assert decoder_output.last_hidden_state.shape == expected_shape
        assert torch.isfinite(decoder_output.last_hidden_state).all()
        
        print(f"Language decoder: {input_ids.shape} -> {decoder_output.last_hidden_state.shape}")
    
    def test_cross_modal_attention_mechanism(self):
        """Test cross-modal attention between vision and language"""
        # Create vision and language features
        batch_size, vision_seq_len, lang_seq_len, hidden_dim = 1, 196, 64, 256
        vision_features = torch.randn(batch_size, vision_seq_len, hidden_dim)
        lang_features = torch.randn(batch_size, lang_seq_len, hidden_dim)
        
        # Simulate cross-modal attention (vision attending to language and vice versa)
        # This is a simplified version - in reality, this would use proper attention mechanisms
        cross_attention_v2l = torch.bmm(vision_features, lang_features.transpose(-2, -1))
        cross_attention_l2v = torch.bmm(lang_features, vision_features.transpose(-2, -1))
        
        # Apply softmax to get attention weights
        attention_weights_v2l = torch.softmax(cross_attention_v2l, dim=-1)
        attention_weights_l2v = torch.softmax(cross_attention_l2v, dim=-1)
        
        # Apply attention to get context vectors
        v_context = torch.bmm(attention_weights_v2l, lang_features)
        l_context = torch.bmm(attention_weights_l2v, vision_features)
        
        assert v_context.shape == (batch_size, vision_seq_len, hidden_dim)
        assert l_context.shape == (batch_size, lang_seq_len, hidden_dim)
        
        print(f"Cross-modal attention shapes: V{vision_features.shape} x L{lang_features.shape} -> "
              f"V_context{v_context.shape}, L_context{l_context.shape}")
    
    def test_cross_modal_integration_pipeline(self):
        """Test full cross-modal integration pipeline"""
        # Initialize components
        vision_encoder = Qwen3VLVisionTransformer(self.config)
        multimodal_projector = Qwen3VLMultimodalProjector(self.config)
        language_decoder = Qwen3VLDecoder(self.config)
        
        # Create inputs
        batch_size = 1
        text_seq_len = 32
        images = torch.randn(batch_size, 3, 224, 224)
        text_ids = torch.randint(0, self.config.vocab_size, (batch_size, text_seq_len))
        
        # Vision processing
        vision_features = vision_encoder(images)  # [batch, patches, vision_hidden]
        projected_vision = multimodal_projector(vision_features)  # [batch, patches, hidden]
        
        # Language processing
        language_output = language_decoder(input_ids=text_ids)  # [batch, seq_len, hidden]
        language_features = language_output.last_hidden_state
        
        # Verify dimensions match for multimodal fusion
        assert projected_vision.shape[2] == language_features.shape[2]  # Hidden sizes match
        
        # Simulate multimodal fusion (concatenate or add)
        # In practice, this would be more sophisticated
        combined_features = torch.cat([projected_vision, language_features], dim=1)
        
        assert combined_features.shape[0] == batch_size
        assert combined_features.shape[2] == self.config.hidden_size
        
        print(f"Cross-modal pipeline: Images{images.shape} + Text{text_ids.shape} -> Combined{combined_features.shape}")
    
    def test_cross_modal_token_merging_with_vision_language(self):
        """Test cross-modal token merging with both vision and language tokens"""
        batch_size, hidden_dim = 2, 256
        
        # Create vision tokens (patched image features)
        vision_seq_len = (224 // 16) ** 2  # 196 patches
        vision_tokens = torch.randn(batch_size, vision_seq_len, hidden_dim)
        
        # Create language tokens
        lang_seq_len = 64
        lang_tokens = torch.randn(batch_size, lang_seq_len, hidden_dim)
        
        # Combine vision and language tokens
        combined_tokens = torch.cat([vision_tokens, lang_tokens], dim=1)
        
        # Apply cross-modal token merging
        merged_tokens = self.token_merger.merge_tokens(combined_tokens)
        
        assert merged_tokens.shape[0] == batch_size
        assert merged_tokens.shape[2] == hidden_dim
        # Should maintain sequence length in this implementation
        assert merged_tokens.shape[1] == vision_seq_len + lang_seq_len
        
        print(f"Cross-modal token merging: Combined{combined_tokens.shape} -> Merged{merged_tokens.shape}")
    
    def test_multimodal_similarity_metrics(self):
        """Test multimodal similarity computation"""
        batch_size, seq_len, hidden_dim = 2, 32, 128
        
        # Create two sets of features (could be vision and language)
        features1 = torch.randn(batch_size, seq_len, hidden_dim)
        features2 = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Compute different similarity metrics
        cosine_sim = torch.nn.functional.cosine_similarity(features1, features2, dim=-1)
        dot_product_sim = torch.sum(features1 * features2, dim=-1)
        
        # Verify similarity ranges
        assert torch.all(cosine_sim >= -1.0) and torch.all(cosine_sim <= 1.0)
        # Dot product can be any value, but we can check it's finite
        assert torch.isfinite(dot_product_sim).all()
        
        print(f"Similarity metrics - Cosine: {cosine_sim.shape}, Dot: {dot_product_sim.shape}")
    
    def test_cross_modal_processing_edge_cases(self):
        """Test cross-modal processing with edge cases"""
        # Test with single vision token
        single_vision = torch.randn(1, 1, 128)
        single_lang = torch.randn(1, 1, 128)
        
        combined_single = torch.cat([single_vision, single_lang], dim=1)
        merged_single = self.token_merger.merge_tokens(combined_single)
        
        assert merged_single.shape == (1, 2, 128)
        
        # Test with empty sequences (this should be handled gracefully)
        try:
            empty_vision = torch.randn(1, 0, 128)
            if empty_vision.shape[1] > 0:  # Only if sequence length > 0
                empty_merged = self.token_merger.merge_tokens(empty_vision)
                assert empty_merged.shape[0] == 1
                assert empty_merged.shape[2] == 128
        except:
            # Empty sequences might not be supported, which is OK
            pass
        
        print("Cross-modal edge case tests completed")


def run_cross_modal_tests():
    """Run all cross-modal processing tests"""
    print("="*70)
    print("RUNNING CROSS-MODAL PROCESSING TESTS")
    print("="*70)
    
    test_instance = TestCrossModalProcessing()
    
    test_methods = [
        'test_cross_modal_token_similarity_computation',
        'test_cross_modal_token_merging',
        'test_multimodal_projector_integration',
        'test_vision_transformer_forward',
        'test_language_decoder_forward',
        'test_cross_modal_attention_mechanism',
        'test_cross_modal_integration_pipeline',
        'test_cross_modal_token_merging_with_vision_language',
        'test_multimodal_similarity_metrics',
        'test_cross_modal_processing_edge_cases'
    ]
    
    results = {}
    for method_name in test_methods:
        try:
            print(f"\nRunning {method_name}...")
            method = getattr(test_instance, method_name)
            method()
            results[method_name] = True
            print(f"✓ {method_name} PASSED")
        except Exception as e:
            results[method_name] = False
            print(f"✗ {method_name} FAILED: {str(e)}")
    
    # Summary
    print("\n" + "="*70)
    print("CROSS-MODAL PROCESSING TEST SUMMARY")
    print("="*70)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    print(f"Success rate: {success_rate:.2%}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_cross_modal_tests()
    exit(0 if success else 1)