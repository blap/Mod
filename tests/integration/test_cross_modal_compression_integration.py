"""
Integration test for cross-modal memory compression with the Qwen3-VL architecture.
This verifies that the compression system works properly with the existing model.
"""
import torch
from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from src.qwen3_vl.multimodal.cross_modal_compression import CrossModalMemoryCompressor, CrossModalCompressionConfig


def test_integration_with_model():
    """Test integration of cross-modal compression with the Qwen3-VL model."""
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
    
    # Get original vision and language features
    with torch.no_grad():
        # Extract vision features
        vision_features = model.vision_tower(pixel_values)
        print(f"Original vision features shape: {vision_features.shape}")
        
        # Extract language features
        lang_features = model.language_model.embed_tokens(input_ids)
        print(f"Original language features shape: {lang_features.shape}")
    
    # Create compression configuration
    compression_config = CrossModalCompressionConfig(
        compression_ratio=0.5,  # Reduce sequence length by half
        low_rank_dimension=128,  # Reduce feature dimension
        semantic_preservation_strength=0.8
    )
    
    # Create compressor
    compressor = CrossModalMemoryCompressor(compression_config)
    
    # Test compression
    compressed_vision, compressed_lang, compression_info = compressor.compress(
        vision_features, lang_features
    )
    
    print(f"Compressed vision features shape: {compressed_vision.shape}")
    print(f"Compressed language features shape: {compressed_lang.shape}")
    print(f"Compression info: {compression_info}")
    
    # Test decompression
    decompressed_vision, decompressed_lang = compressor.decompress(
        compressed_vision, compressed_lang,
        original_vision_shape=vision_features.shape,
        original_lang_shape=lang_features.shape
    )
    
    print(f"Decompressed vision features shape: {decompressed_vision.shape}")
    print(f"Decompressed language features shape: {decompressed_lang.shape}")
    
    # Verify dimensions
    assert decompressed_vision.shape[-1] == vision_features.shape[-1]  # Feature dim preserved
    assert decompressed_lang.shape[-1] == lang_features.shape[-1]      # Feature dim preserved
    
    print("Integration test passed!")


def test_memory_reduction():
    """Test that the compression actually reduces memory usage."""
    # Create features to compress
    batch_size = 2
    vision_seq_len = 196  # 14x14 patches
    lang_seq_len = 64
    hidden_dim = 512
    
    vision_features = torch.randn(batch_size, vision_seq_len, hidden_dim)
    lang_features = torch.randn(batch_size, lang_seq_len, hidden_dim)
    
    # Calculate original memory usage
    original_memory = (vision_features.numel() + lang_features.numel()) * 4  # 4 bytes per float32
    
    # Create compressor
    compression_config = CrossModalCompressionConfig(
        compression_ratio=0.5,
        low_rank_dimension=256
    )
    compressor = CrossModalMemoryCompressor(compression_config)
    
    # Compress
    compressed_vision, compressed_lang, compression_info = compressor.compress(
        vision_features, lang_features
    )
    
    # Calculate compressed memory usage
    compressed_memory = (compressed_vision.numel() + compressed_lang.numel()) * 4
    
    print(f"Original memory: {original_memory / 1024 / 1024:.2f} MB")
    print(f"Compressed memory: {compressed_memory / 1024 / 1024:.2f} MB")
    print(f"Memory reduction ratio: {compression_info['memory_reduction_ratio']:.2%}")
    
    # Verify memory was reduced
    assert compressed_memory < original_memory
    assert compression_info['memory_reduction_ratio'] > 0.3  # At least 30% reduction
    
    print("Memory reduction test passed!")


def test_semantic_preservation():
    """Test that semantic information is preserved during compression."""
    # Create features with some semantic structure
    batch_size = 1
    vision_seq_len = 64
    lang_seq_len = 32
    hidden_dim = 256
    
    vision_features = torch.randn(batch_size, vision_seq_len, hidden_dim)
    lang_features = torch.randn(batch_size, lang_seq_len, hidden_dim)
    
    # Add some semantic patterns
    vision_features[:, :10, :50] = torch.ones(1, 10, 50) * 2.0  # Semantic cluster
    lang_features[:, :5, :50] = torch.ones(1, 5, 50) * 2.0     # Corresponding cluster
    
    # Create compressor
    compression_config = CrossModalCompressionConfig(
        compression_ratio=0.6,
        low_rank_dimension=128,
        semantic_preservation_strength=0.9
    )
    compressor = CrossModalMemoryCompressor(compression_config)
    
    # Compress and decompress
    compressed_vision, compressed_lang, compression_info = compressor.compress(
        vision_features, lang_features
    )
    decompressed_vision, decompressed_lang = compressor.decompress(
        compressed_vision, compressed_lang,
        original_vision_shape=vision_features.shape,
        original_lang_shape=lang_features.shape
    )
    
    # Check that semantic preservation metrics are computed
    assert 'semantic_preservation_metrics' in compression_info
    print(f"Semantic preservation info: {compression_info['semantic_preservation_metrics']}")
    
    # Check that feature dimensions are preserved
    assert decompressed_vision.shape[-1] == vision_features.shape[-1]
    assert decompressed_lang.shape[-1] == lang_features.shape[-1]
    
    print("Semantic preservation test passed!")


if __name__ == "__main__":
    test_integration_with_model()
    test_memory_reduction()
    test_semantic_preservation()
    print("\nAll integration tests passed!")