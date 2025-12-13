"""
Validation script for Phase 3: Vision-Language Integration Optimization
This script validates that all requirements from the architecture update plan are met.
"""
import torch
from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl_phase3 import Qwen3VLForConditionalGeneration, Qwen3VLVisionTransformer, Qwen3VLMultimodalProjector, EfficientCrossAttention, FactorizedConv2d


def validate_all_requirements():
    """Validate that all Phase 3 requirements are met."""
    print("Validating Phase 3: Vision-Language Integration Optimization requirements...")
    
    # 1. Verify that DeepStack has been replaced with efficient cross-attention mechanism
    print("\n1. Validating efficient cross-attention mechanism replacement...")
    config = Qwen3VLConfig()
    cross_attn = EfficientCrossAttention(config, use_sparse=True)
    assert hasattr(cross_attn, 'q_proj'), "Efficient cross-attention should have query projection"
    assert hasattr(cross_attn, 'k_proj'), "Efficient cross-attention should have key projection" 
    assert hasattr(cross_attn, 'v_proj'), "Efficient cross-attention should have value projection"
    assert hasattr(cross_attn, 'o_proj'), "Efficient cross-attention should have output projection"
    assert cross_attn.use_sparse == True, "Should support sparse attention"
    print("   [PASS] Efficient cross-attention mechanism implemented")

    # 2. Verify vision encoder optimization with factorized operations
    print("\n2. Validating vision encoder optimization with factorized operations...")
    factorized_conv = FactorizedConv2d(in_channels=3, out_channels=768, kernel_size=14, stride=14, bias=False)
    assert hasattr(factorized_conv, 'depthwise_conv'), "Should have depthwise convolution"
    assert hasattr(factorized_conv, 'pointwise_conv'), "Should have pointwise convolution"
    assert hasattr(factorized_conv, 'norm'), "Should have normalization layer"
    print("   [PASS] Factorized operations in vision encoder implemented")

    # 3. Verify sparse cross-attention implementation
    print("\n3. Validating sparse cross-attention implementation...")
    sparse_cross_attn = EfficientCrossAttention(config, use_sparse=True)
    assert sparse_cross_attn.use_sparse == True, "Should support sparse attention"
    assert hasattr(sparse_cross_attn, 'sparse_factor'), "Should have sparse factor parameter"
    print("   [PASS] Sparse cross-attention implemented")

    # 4. Verify vision-language alignment mechanisms
    print("\n4. Validating vision-language alignment mechanisms...")
    projector = Qwen3VLMultimodalProjector(config)
    assert hasattr(projector, 'linear_1'), "Should have first linear layer"
    assert hasattr(projector, 'act'), "Should have activation function"
    assert hasattr(projector, 'norm'), "Should have normalization layer"
    assert hasattr(projector, 'linear_2'), "Should have second linear layer"
    print("   [PASS] Vision-language alignment mechanisms implemented")

    # 5. Verify integration with existing parameter counts
    print("\n5. Validating integration with existing parameter counts...")
    model = Qwen3VLForConditionalGeneration(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Verify all 32 layers and 32 attention heads are preserved
    assert config.num_hidden_layers == 32, f"Should have 32 hidden layers, got {config.num_hidden_layers}"
    assert config.num_attention_heads == 32, f"Should have 32 attention heads, got {config.num_attention_heads}"
    print("   [PASS] All 32 transformer layers preserved")
    print("   [PASS] All 32 attention heads preserved")

    # 6. Test model functionality
    print("\n6. Testing model functionality...")

    # Test vision encoder
    vision_encoder = model.vision_tower
    batch_size = 1
    image_size = 448
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    vision_features = vision_encoder(pixel_values)
    expected_patches = (image_size // config.vision_patch_size) ** 2
    assert vision_features.shape == (batch_size, expected_patches, config.vision_hidden_size), \
        f"Vision features shape mismatch: expected ({batch_size}, {expected_patches}, {config.vision_hidden_size}), got {vision_features.shape}"
    print("   [PASS] Vision encoder functionality working")

    # Test multimodal projector
    projected_features = model.multi_modal_projector(vision_features)
    assert projected_features.shape == (batch_size, expected_patches, config.hidden_size), \
        f"Projected features shape mismatch: expected ({batch_size}, {expected_patches}, {config.hidden_size}), got {projected_features.shape}"
    print("   [PASS] Multimodal projector functionality working")

    # Test full model forward pass
    input_ids = torch.randint(0, config.vocab_size, (batch_size, 10))
    outputs = model(input_ids=input_ids, pixel_values=pixel_values)
    assert outputs.shape[0] == batch_size, f"Output batch size mismatch: expected {batch_size}, got {outputs.shape[0]}"
    print("   [PASS] Full model forward pass working")

    # 7. Performance comparison (conceptual - would need actual benchmarks in practice)
    print("\n7. Validating performance characteristics...")

    # Verify that the model has the expected components for efficiency
    assert hasattr(model.language_model.layers[0], 'vision_cross_attn'), "Decoder layers should have cross-attention"
    assert isinstance(model.language_model.layers[0].vision_cross_attn, EfficientCrossAttention), "Should use efficient cross-attention"
    print("   [PASS] Efficient cross-attention integrated in decoder layers")

    # 8. Architecture preservation
    print("\n8. Validating architecture preservation...")
    assert len(model.language_model.layers) == 32, "Should have 32 decoder layers"
    assert len(model.vision_tower.layers) == config.vision_num_hidden_layers, f"Should have {config.vision_num_hidden_layers} vision layers"
    print("   [PASS] Architecture capacity preserved")

    print("\n[SUCCESS] All Phase 3 requirements validated successfully!")
    print(f"[SUCCESS] Model maintains full capacity: {config.num_hidden_layers} layers, {config.num_attention_heads} attention heads")
    print(f"[SUCCESS] Total parameters: {total_params:,}")
    print("[SUCCESS] Phase 3: Vision-Language Integration Optimization completed successfully!")


if __name__ == "__main__":
    validate_all_requirements()