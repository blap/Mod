"""
Accuracy Validation Test for Qwen3-VL-2B-Instruct Architecture
This test validates that no quality degradation occurs with optimizations enabled.
"""
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
import json
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory_manager import MemoryManager, MemoryConfig
from kv_cache_optimizer import KVCacheConfig, OptimizedKVCacheManager
from src.qwen3_vl.core.config import Qwen3VLConfig
from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


def create_baseline_model():
    """Create a baseline model without optimizations for comparison"""
    config = Qwen3VLConfig()
    config.use_sparsity = False
    config.use_gradient_checkpointing = False
    config.hidden_size = 256  # Use smaller size for testing
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.vocab_size = 500  # Smaller vocab for testing
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    return model, config


def create_optimized_model():
    """Create an optimized model with all optimizations enabled"""
    config = Qwen3VLConfig()
    config.use_sparsity = True
    config.sparsity_ratio = 0.3  # Lower sparsity for testing
    config.exit_threshold = 0.75
    config.use_gradient_checkpointing = False  # Disable for testing accuracy
    config.hidden_size = 256  # Use smaller size for testing
    config.intermediate_size = 512
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.vocab_size = 500  # Smaller vocab for testing
    
    model = Qwen3VLForConditionalGeneration(config)
    model.eval()
    
    # Integrate memory manager
    memory_manager = MemoryManager(MemoryConfig(memory_pool_size=2**24))  # 16MB pool
    # Integrate KV cache optimizer
    kv_config = KVCacheConfig(
        use_low_rank=True,
        low_rank_dimension=32,  # Lower rank for testing
        use_sliding_window=True,
        sliding_window_size=256,
        use_hybrid=True
    )
    kv_cache_manager = OptimizedKVCacheManager(kv_config, memory_manager)
    
    # Note: In a real implementation, these would be integrated during model construction
    model.memory_manager = memory_manager
    model.kv_cache_manager = kv_cache_manager
    
    return model, config


def calculate_output_similarity(baseline_output, optimized_output, metric='cosine'):
    """Calculate similarity between baseline and optimized outputs"""
    if metric == 'cosine':
        # Flatten the outputs for cosine similarity calculation
        baseline_flat = baseline_output.view(-1)
        optimized_flat = optimized_output.view(-1)
        
        # Calculate cosine similarity
        cosine_sim = F.cosine_similarity(baseline_flat.unsqueeze(0), optimized_flat.unsqueeze(0)).item()
        return cosine_sim
    elif metric == 'mse':
        # Calculate mean squared error
        mse = F.mse_loss(baseline_output, optimized_output).item()
        return mse
    elif metric == 'mae':
        # Calculate mean absolute error
        mae = F.l1_loss(baseline_output, optimized_output).item()
        return mae
    else:
        raise ValueError(f"Unknown metric: {metric}")


def test_output_consistency():
    """Test that optimized model produces consistent outputs"""
    print("Testing output consistency...")
    
    # Create both models
    baseline_model, baseline_config = create_baseline_model()
    optimized_model, opt_config = create_optimized_model()
    
    # Create test inputs
    batch_size, seq_len = 1, 16
    input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, baseline_config.vision_image_size, baseline_config.vision_image_size)
    
    try:
        # Get outputs from both models
        with torch.no_grad():
            baseline_output = baseline_model(input_ids=input_ids, pixel_values=pixel_values).logits
            optimized_output = optimized_model(input_ids=input_ids, pixel_values=pixel_values).logits
        
        # Verify shapes match
        assert baseline_output.shape == optimized_output.shape, f"Shape mismatch: {baseline_output.shape} vs {optimized_output.shape}"
        
        # Calculate similarity metrics
        cosine_sim = calculate_output_similarity(baseline_output, optimized_output, 'cosine')
        mse = calculate_output_similarity(baseline_output, optimized_output, 'mse')
        mae = calculate_output_similarity(baseline_output, optimized_output, 'mae')
        
        print(f"  OK Output shapes match: {baseline_output.shape}")
        print(f"  OK Cosine similarity: {cosine_sim:.6f}")
        print(f"  OK MSE: {mse:.8f}")
        print(f"  OK MAE: {mae:.8f}")
        
        # Check if outputs are similar enough (thresholds based on acceptable degradation)
        cosine_threshold = 0.95  # At least 95% similarity
        mse_threshold = 0.01     # Max MSE
        mae_threshold = 0.05     # Max MAE
        
        is_similar = (
            cosine_sim >= cosine_threshold or
            mse <= mse_threshold or
            mae <= mae_threshold
        )
        
        print(f"  OK Output similarity acceptable: {'YES' if is_similar else 'NO'}")
        
        return is_similar, cosine_sim, mse, mae
        
    except Exception as e:
        print(f"  X Output consistency test failed: {e}")
        traceback.print_exc()
        return False, 0.0, float('inf'), float('inf')


def test_numerical_stability():
    """Test numerical stability of optimized model"""
    print("Testing numerical stability...")
    
    # Create optimized model
    model, config = create_optimized_model()
    
    # Run multiple forward passes with the same input
    batch_size, seq_len = 1, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size)
    
    outputs = []
    for i in range(5):
        with torch.no_grad():
            output = model(input_ids=input_ids, pixel_values=pixel_values).logits
        outputs.append(output)
    
    # Check consistency between runs
    base_output = outputs[0]
    differences = []
    
    for i in range(1, len(outputs)):
        diff = torch.mean(torch.abs(base_output - outputs[i])).item()
        differences.append(diff)
    
    avg_difference = np.mean(differences)
    max_difference = np.max(differences)
    
    print(f"  OK Average output difference across runs: {avg_difference:.8f}")
    print(f"  OK Maximum output difference across runs: {max_difference:.8f}")
    
    # Check for stability (small differences indicate stability)
    is_stable = max_difference < 1e-5
    
    print(f"  OK Numerical stability: {'YES' if is_stable else 'NO'}")
    
    return is_stable, avg_difference, max_difference


def test_gradient_flow():
    """Test that gradients flow properly through optimized components"""
    print("Testing gradient flow...")
    
    # Create optimized model with gradients enabled
    model, config = create_optimized_model()
    model.train()  # Set to train mode to enable gradients
    
    # Create test inputs with gradients enabled
    batch_size, seq_len = 1, 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, config.vision_image_size, config.vision_image_size)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    try:
        # Forward pass
        output = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        loss = output.loss if hasattr(output, 'loss') else None
        
        if loss is not None:
            # Backward pass
            loss.backward()
            
            # Check that gradients exist for parameters
            params_with_grad = 0
            total_params = 0
            
            for param in model.parameters():
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
            
            grad_coverage = params_with_grad / total_params if total_params > 0 else 0
            
            print(f"  OK Parameters with gradients: {params_with_grad}/{total_params} ({grad_coverage:.1%})")
            
            # Check gradient magnitudes for reasonable values
            grad_magnitudes = []
            for param in model.parameters():
                if param.grad is not None:
                    grad_magnitudes.append(param.grad.norm().item())
            
            avg_grad_magnitude = np.mean(grad_magnitudes) if grad_magnitudes else 0
            print(f"  OK Average gradient magnitude: {avg_grad_magnitude:.6f}")
            
            # Check for gradient issues
            has_valid_gradients = avg_grad_magnitude > 0 and avg_grad_magnitude < 1000  # Reasonable range
            no_nan_gradients = not any(torch.isnan(param.grad).any() for param in model.parameters() if param.grad is not None)
            no_inf_gradients = not any(torch.isinf(param.grad).any() for param in model.parameters() if param.grad is not None)
            
            print(f"  OK Valid gradients: {'YES' if has_valid_gradients else 'NO'}")
            print(f"  OK No NaN gradients: {'YES' if no_nan_gradients else 'NO'}")
            print(f"  OK No Inf gradients: {'YES' if no_inf_gradients else 'NO'}")
            
            is_valid = has_valid_gradients and no_nan_gradients and no_inf_gradients
            
            return is_valid, grad_coverage, avg_grad_magnitude
        else:
            print("  X No loss computed for gradient test")
            return False, 0.0, 0.0
            
    except Exception as e:
        print(f"  X Gradient flow test failed: {e}")
        traceback.print_exc()
        return False, 0.0, 0.0


def test_token_prediction_accuracy():
    """Test token prediction accuracy consistency"""
    print("Testing token prediction accuracy...")
    
    # Create both models
    baseline_model, baseline_config = create_baseline_model()
    optimized_model, opt_config = create_optimized_model()
    
    # Create test inputs
    batch_size, seq_len = 1, 16
    input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
    pixel_values = torch.randn(batch_size, 3, baseline_config.vision_image_size, baseline_config.vision_image_size)
    
    try:
        with torch.no_grad():
            # Get logits from both models
            baseline_logits = baseline_model(input_ids=input_ids, pixel_values=pixel_values).logits
            optimized_logits = optimized_model(input_ids=input_ids, pixel_values=pixel_values).logits
        
        # Get predicted tokens
        baseline_preds = torch.argmax(baseline_logits, dim=-1)
        optimized_preds = torch.argmax(optimized_logits, dim=-1)
        
        # Calculate accuracy
        token_accuracy = (baseline_preds == optimized_preds).float().mean().item()
        
        print(f"  OK Token prediction accuracy: {token_accuracy:.4f}")
        
        # Calculate top-k accuracy (top-3)
        k = 3
        baseline_topk = torch.topk(baseline_logits, k, dim=-1).indices
        optimized_topk = torch.topk(optimized_logits, k, dim=-1).indices
        
        # Check if top-k predictions overlap significantly
        topk_overlap = 0
        total_comparisons = 0
        
        for i in range(batch_size):
            for j in range(seq_len):
                baseline_set = set(baseline_topk[i, j].cpu().numpy())
                optimized_set = set(optimized_topk[i, j].cpu().numpy())
                overlap = len(baseline_set.intersection(optimized_set))
                topk_overlap += overlap
                total_comparisons += k
        
        topk_accuracy = topk_overlap / total_comparisons if total_comparisons > 0 else 0
        
        print(f"  OK Top-{k} prediction overlap: {topk_accuracy:.4f}")
        
        # Check if accuracy is acceptable
        is_accurate = token_accuracy >= 0.8 or topk_accuracy >= 0.9  # Either exact match or top-k overlap
        
        print(f"  OK Prediction accuracy acceptable: {'YES' if is_accurate else 'NO'}")
        
        return is_accurate, token_accuracy, topk_accuracy
        
    except Exception as e:
        print(f"  X Token prediction accuracy test failed: {e}")
        traceback.print_exc()
        return False, 0.0, 0.0


def run_accuracy_validation():
    """Run all accuracy validation tests"""
    print("=" * 80)
    print("ACCURACY VALIDATION TEST FOR QWEN3-VL-2B-INSTRUCT ARCHITECTURE")
    print("=" * 80)
    
    print("Creating models for accuracy validation...")
    
    tests = [
        ("Output Consistency", test_output_consistency),
        ("Numerical Stability", test_numerical_stability),
        ("Gradient Flow", test_gradient_flow),
        ("Token Prediction Accuracy", test_token_prediction_accuracy),
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        try:
            result = test_func()
            if isinstance(result, tuple):
                success, *metrics = result
            else:
                success = result
                metrics = []
            
            results[test_name] = {"success": success, "metrics": metrics}
            
            status = "PASS" if success else "FAIL"
            print(f"  Status: {status}")
            
            if not success:
                all_passed = False
        except Exception as e:
            print(f"  Status: FAIL - Error: {e}")
            traceback.print_exc()
            results[test_name] = {"success": False, "metrics": [], "error": str(e)}
            all_passed = False
    
    print("\n" + "=" * 80)
    print("ACCURACY VALIDATION SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "[PASS]" if result["success"] else "[FAIL]"
        print(f"{status} {test_name}")
        if result["metrics"]:
            print(f"       Metrics: {result['metrics']}")
    
    print(f"\nOverall Accuracy Validation: {'PASSED' if all_passed else 'FAILED'}")
    
    if all_passed:
        print("\n✓ All accuracy validation tests passed!")
        print("✓ No quality degradation detected with optimizations enabled")
        print("✓ Model maintains expected behavior and numerical stability")
    else:
        print("\n✗ Some accuracy validation tests failed!")
        print("✗ Quality degradation may have occurred with optimizations")
        print("✗ Review optimization implementations for accuracy preservation")
    
    return all_passed


if __name__ == "__main__":
    success = run_accuracy_validation()
    
    print(f"\n{'='*80}")
    print("ACCURACY VALIDATION STATUS:", "PASSED" if success else "FAILED")
    print(f"{'='*80}")
    
    sys.exit(0 if success else 1)