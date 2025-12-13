"""
Simple test to verify model creation with all optimizations
"""
import torch
from src.models.config import Qwen3VLConfig
from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

def test_model_creation():
    print("Testing model creation with all optimizations enabled...")
    
    config = Qwen3VLConfig()
    
    # Enable all 10 optimization techniques
    config.use_adaptive_precision = True
    config.use_sparsity = True
    config.use_dynamic_sparse_attention = True
    config.attention_implementation = "kv_cache_optimized"
    config.use_context_adaptive_positional_encoding = True
    config.use_conditional_feature_extraction = True
    config.enable_cross_modal_compression = True
    config.enable_cross_layer_memory_sharing = True
    config.use_adaptive_depth = True
    config.use_moe = True
    config.moe_num_experts = 4
    config.moe_top_k = 2
    config.sparsity_ratio = 0.5
    config.exit_threshold = 0.8
    config.compression_ratio = 0.7
    config.min_depth_ratio = 0.3
    config.max_depth_ratio = 1.0
    config.vision_sparse_attention_sparsity_ratio = 0.4
    config.use_vision_adaptive_depth = True
    config.vision_min_depth_ratio = 0.4
    config.vision_max_depth_ratio = 1.0
    
    print("Configuration created successfully")
    print(f"Adaptive Precision: {config.use_adaptive_precision}")
    print(f"Sparsity: {config.use_sparsity}")
    print(f"Dynamic Sparse Attention: {config.use_dynamic_sparse_attention}")
    print(f"KV Cache Optimization: {config.attention_implementation}")
    print(f"Context-Adaptive Positional Encoding: {config.use_context_adaptive_positional_encoding}")
    print(f"Conditional Feature Extraction: {config.use_conditional_feature_extraction}")
    print(f"Cross-Modal Compression: {config.enable_cross_modal_compression}")
    print(f"Cross-Layer Memory Sharing: {config.enable_cross_layer_memory_sharing}")
    print(f"Adaptive Depth: {config.use_adaptive_depth}")
    print(f"Mixture of Experts: {config.use_moe}")
    
    try:
        model = Qwen3VLForConditionalGeneration(config)
        print(f"\nSUCCESS: Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test a simple forward pass
        input_ids = torch.randint(0, 1000, (1, 8))
        pixel_values = torch.randn(1, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids, pixel_values=pixel_values)
            print(f"Forward pass successful, output shape: {output.shape if isinstance(output, torch.Tensor) else 'Complex output'}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_creation()
    print(f"\nModel creation test: {'PASSED' if success else 'FAILED'}")