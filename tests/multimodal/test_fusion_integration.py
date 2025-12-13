"""
Test for integrating cross-modal compression into the Qwen3-VL architecture.
This shows how the compression system would be used during vision-language fusion.
"""
import torch
from models.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from src.qwen3_vl.multimodal.cross_modal_compression import CrossModalFusionCompressor, CrossModalCompressionConfig


def test_fusion_integration():
    """Test how cross-modal compression would be integrated into the model."""
    # Create a model configuration
    config = Qwen3VLConfig()
    config.num_hidden_layers = 4  # Reduced for testing
    config.num_attention_heads = 8
    config.hidden_size = 256
    config.vision_hidden_size = 256
    config.vision_num_hidden_layers = 2
    config.vocab_size = 1000
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Create test inputs
    batch_size = 1
    seq_len = 16
    img_size = 224
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, img_size, img_size)
    
    # Get vision and language features
    with torch.no_grad():
        # Extract vision features
        vision_features = model.vision_tower(pixel_values)
        print(f"Vision features shape: {vision_features.shape}")
        
        # Extract language features
        lang_features = model.language_model.embed_tokens(input_ids)
        print(f"Language features shape: {lang_features.shape}")
    
    # Create compression configuration for fusion
    compression_config = CrossModalCompressionConfig(
        compression_ratio=0.6,  # Moderate compression
        low_rank_dimension=128,  # Reduce feature dimension
        semantic_preservation_strength=0.8,
        use_cross_attention_selection=True,
        use_low_rank_compression=True
    )
    
    # Create fusion compressor
    fusion_compressor = CrossModalFusionCompressor(compression_config)
    
    # Apply compression during fusion stage
    compressed_vision, compressed_lang, compression_info = fusion_compressor(
        vision_features, lang_features
    )
    
    print(f"Compressed vision features shape: {compressed_vision.shape}")
    print(f"Compressed language features shape: {compressed_lang.shape}")
    print(f"Compression ratios - Vision: {compression_info['compression_ratios']['vision']:.2%}, "
          f"Language: {compression_info['compression_ratios']['language']:.2%}")
    print(f"Memory reduction: {compression_info['memory_reduction_ratio']:.2%}")
    
    # The compressed features can now be used in the model with reduced memory usage
    # For example, they can be projected and combined:
    combined_features = torch.cat([compressed_vision, compressed_lang], dim=1)
    print(f"Combined features shape: {combined_features.shape}")
    
    # This would then be processed by the language model with significantly reduced memory requirements
    # compared to using the full uncompressed features
    
    print("Fusion integration test passed!")


def test_full_model_integration():
    """Test integration with the full model forward pass."""
    # Create a model configuration
    config = Qwen3VLConfig()
    config.num_hidden_layers = 2  # Very reduced for testing
    config.num_attention_heads = 4
    config.hidden_size = 128
    config.vision_hidden_size = 128
    config.vision_num_hidden_layers = 2
    config.vocab_size = 1000
    config.vision_patch_size = 16  # Smaller patches for smaller images
    config.vision_image_size = 224
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Create smaller test inputs to make the test faster
    batch_size = 1
    seq_len = 8
    img_size = 224  # Using 224x224 which should work with patch size 16
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"Input shapes - input_ids: {input_ids.shape}, pixel_values: {pixel_values.shape}")
    
    # Run the model forward pass with both modalities
    with torch.no_grad():
        output = model(input_ids=input_ids, pixel_values=pixel_values)
        print(f"Model output shape: {output.shape}")
    
    print("Full model integration test passed!")


def test_compression_with_different_ratios():
    """Test compression with different compression ratios."""
    # Create sample features
    batch_size = 1
    vision_seq_len = 196  # 14x14 patches
    lang_seq_len = 64
    hidden_dim = 256
    
    vision_features = torch.randn(batch_size, vision_seq_len, hidden_dim)
    lang_features = torch.randn(batch_size, lang_seq_len, hidden_dim)
    
    # Test different compression ratios
    compression_ratios = [0.3, 0.5, 0.7, 0.9]
    
    for ratio in compression_ratios:
        print(f"\nTesting compression ratio: {ratio}")
        
        config = CrossModalCompressionConfig(
            compression_ratio=ratio,
            low_rank_dimension=128
        )
        compressor = CrossModalFusionCompressor(config)
        
        compressed_vision, compressed_lang, info = compressor(vision_features, lang_features)
        
        print(f"  Original vision: {vision_features.shape} -> Compressed: {compressed_vision.shape}")
        print(f"  Original language: {lang_features.shape} -> Compressed: {compressed_lang.shape}")
        print(f"  Actual compression ratios - Vision: {info['compression_ratios']['vision']:.2%}, "
              f"Language: {info['compression_ratios']['language']:.2%}")
        print(f"  Memory reduction: {info['memory_reduction_ratio']:.2%}")
        
        # Verify compression worked as expected
        expected_vision_len = max(1, int(vision_seq_len * ratio))
        expected_lang_len = max(1, int(lang_seq_len * ratio))
        assert compressed_vision.shape[1] == expected_vision_len
        assert compressed_lang.shape[1] == expected_lang_len
    
    print("\nDifferent compression ratios test passed!")


if __name__ == "__main__":
    test_fusion_integration()
    test_full_model_integration()
    test_compression_with_different_ratios()
    print("\nAll integration tests passed!")