"""
Comprehensive Test and Benchmark Suite for Qwen3-VL Model Components

This script implements a standardized test and benchmark suite for all Qwen3-VL model components,
ensuring consistency, reliability, and performance validation across the codebase.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import time
import gc
import psutil
import GPUtil
import math
from typing import Optional, Dict, Any, Tuple
import json


def test_config_capacity():
    """Test that configuration maintains full capacity (32 layers, 32 attention heads)."""
    print("Testing configuration capacity preservation...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        config = Qwen3VLConfig()
        
        print(f"  V Config created with {config.num_hidden_layers} layers, {config.num_attention_heads} attention heads")
        print(f"  V Vision config: {config.vision_num_hidden_layers} vision layers, {config.vision_num_attention_heads} vision heads")
        
        # Verify capacity preservation
        assert config.num_hidden_layers == 32, f"Expected 32 hidden layers, got {config.num_hidden_layers}"
        assert config.num_attention_heads == 32, f"Expected 32 attention heads, got {config.num_attention_heads}"
        assert config.vision_num_hidden_layers == 24, f"Expected 24 vision layers, got {config.vision_num_hidden_layers}"
        assert config.vision_num_attention_heads == 16, f"Expected 16 vision attention heads, got {config.vision_num_attention_heads}"
        
        print("  V Full model capacity preserved (32 transformer layers, 32 attention heads)")
        return True
    except Exception as e:
        print(f"  X Config capacity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test model creation with optimized configurations."""
    print("\nTesting model creation...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        # Create a smaller config for testing
        config = Qwen3VLConfig()
        config.hidden_size = 128
        config.num_attention_heads = 4
        config.num_hidden_layers = 2
        config.vision_hidden_size = 128
        config.vision_num_attention_heads = 4
        config.vision_num_hidden_layers = 2
        config.vocab_size = 1000
        
        print("  Creating model with optimized configuration...")
        model = Qwen3VLForConditionalGeneration(config)
        print(f"  V Model created with {len(model.language_model.layers)} language layers")
        print(f"  V Model has {len(model.vision_tower.layers)} vision layers")
        
        # Verify the model has the expected components
        assert hasattr(model, 'vision_tower')
        assert hasattr(model, 'language_model')
        assert hasattr(model, 'multi_modal_projector')
        print("  V Model has all required components")
        
        return True
    except Exception as e:
        print(f"  X Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward_pass():
    """Test model forward pass functionality."""
    print("\nTesting model forward pass...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        # Create test config
        config = Qwen3VLConfig()
        config.hidden_size = 128
        config.num_attention_heads = 4
        config.num_hidden_layers = 2
        config.vision_hidden_size = 128
        config.vision_num_attention_heads = 4
        config.vision_num_hidden_layers = 2
        config.vocab_size = 1000
        
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()
        print("  V Model initialized in evaluation mode")
        
        # Create test inputs
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        print(f"  Created test inputs: {input_ids.shape}, {pixel_values.shape}")
        
        # Forward pass with text only
        with torch.no_grad():
            text_output = model(input_ids=input_ids)
        print(f"  V Text-only forward pass completed, output shape: {text_output.shape}")
        
        # Forward pass with image only
        with torch.no_grad():
            vision_output = model(pixel_values=pixel_values)
        print(f"  V Vision-only forward pass completed, output shape: {vision_output.shape}")
        
        # Forward pass with multimodal input
        with torch.no_grad():
            multimodal_output = model(input_ids=input_ids, pixel_values=pixel_values)
        print(f"  V Multimodal forward pass completed, output shape: {multimodal_output.shape}")
        
        # Verify outputs are valid
        assert torch.all(torch.isfinite(text_output)), "Text output should contain finite values"
        assert torch.all(torch.isfinite(vision_output)), "Vision output should contain finite values"
        assert torch.all(torch.isfinite(multimodal_output)), "Multimodal output should contain finite values"
        print("  V All outputs contain finite values")
        
        return True
    except Exception as e:
        print(f"  X Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_generation():
    """Test model generation functionality."""
    print("\nTesting model generation...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        # Create test config
        config = Qwen3VLConfig()
        config.hidden_size = 128
        config.num_attention_heads = 4
        config.num_hidden_layers = 2
        config.vision_hidden_size = 128
        config.vision_num_attention_heads = 4
        config.vision_num_hidden_layers = 2
        config.vocab_size = 1000
        
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()
        print("  V Model initialized for generation")
        
        # Create test inputs
        batch_size = 1
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 5))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        print(f"  Created test inputs for generation")
        
        # Generate text-only
        generated_text = model.generate(
            input_ids=input_ids,
            max_new_tokens=3,
            do_sample=False
        )
        print(f"  V Text-only generation completed, output shape: {generated_text.shape}")
        
        # Generate multimodal
        generated_multimodal = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=3,
            do_sample=False
        )
        print(f"  V Multimodal generation completed, output shape: {generated_multimodal.shape}")
        
        # Verify generated tokens are valid
        assert torch.all(generated_text >= 0) and torch.all(generated_text < config.vocab_size), "Generated tokens should be valid"
        assert torch.all(generated_multimodal >= 0) and torch.all(generated_multimodal < config.vocab_size), "Generated multimodal tokens should be valid"
        print("  V Generated tokens are valid")
        
        return True
    except Exception as e:
        print(f"  X Model generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark_model_performance():
    """Benchmark model performance with standardized measurements."""
    print("\nBenchmarking model performance...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        # Create test config
        config = Qwen3VLConfig()
        config.hidden_size = 256
        config.num_attention_heads = 8
        config.num_hidden_layers = 4
        config.vision_hidden_size = 256
        config.vision_num_attention_heads = 8
        config.vision_num_hidden_layers = 4
        config.vocab_size = 2000
        
        model = Qwen3VLForConditionalGeneration(config)
        model.eval()
        print("  V Model initialized for benchmarking")
        
        # Create test inputs
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        print(f"  Created benchmark inputs")
        
        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
        
        # Benchmark forward pass
        start_time = time.time()
        num_runs = 10
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(input_ids=input_ids, pixel_values=pixel_values)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        print(f"  V Forward pass benchmark completed: {avg_time:.4f}s per run")
        
        # Benchmark generation
        start_time = time.time()
        for _ in range(3):
            with torch.no_grad():
                _ = model.generate(input_ids=input_ids, max_new_tokens=5, do_sample=False)
        gen_time = (time.time() - start_time) / 3
        print(f"  V Generation benchmark completed: {gen_time:.4f}s per run")
        
        # Memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        with torch.no_grad():
            output = model(input_ids=input_ids, pixel_values=pixel_values)
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        print(f"  Memory usage - Initial: {initial_memory/(1024**2):.2f}MB, Final: {final_memory/(1024**2):.2f}MB, Peak: {peak_memory/(1024**2):.2f}MB")
        
        return {
            'forward_pass_time': avg_time,
            'generation_time': gen_time,
            'memory_initial': initial_memory,
            'memory_final': final_memory,
            'memory_peak': peak_memory
        }
    except Exception as e:
        print(f"  X Model benchmarking failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\nTesting memory efficiency optimizations...")
    
    try:
        from src.qwen3_vl.config.config import Qwen3VLConfig
        from src.qwen3_vl.core.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        
        # Create baseline config (without optimizations)
        baseline_config = Qwen3VLConfig()
        baseline_config.hidden_size = 128
        baseline_config.num_attention_heads = 4
        baseline_config.num_hidden_layers = 2
        baseline_config.vision_hidden_size = 128
        baseline_config.vision_num_attention_heads = 4
        baseline_config.vision_num_hidden_layers = 2
        baseline_config.vocab_size = 1000
        baseline_config.use_gradient_checkpointing = False
        baseline_config.use_sparsity = False
        
        # Create optimized config (with optimizations)
        optimized_config = Qwen3VLConfig()
        optimized_config.hidden_size = 128
        optimized_config.num_attention_heads = 4
        optimized_config.num_hidden_layers = 2
        optimized_config.vision_hidden_size = 128
        optimized_config.vision_num_attention_heads = 4
        optimized_config.vision_num_hidden_layers = 2
        optimized_config.vocab_size = 1000
        optimized_config.use_gradient_checkpointing = True
        optimized_config.use_sparsity = True
        optimized_config.sparsity_ratio = 0.5
        
        # Create models
        baseline_model = Qwen3VLForConditionalGeneration(baseline_config)
        optimized_model = Qwen3VLForConditionalGeneration(optimized_config)
        
        # Copy weights to ensure fair comparison
        optimized_model.load_state_dict(baseline_model.state_dict(), strict=False)
        
        baseline_model.eval()
        optimized_model.eval()
        print("  V Baseline and optimized models created")
        
        # Create test inputs
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, baseline_config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        # Measure baseline memory
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        baseline_start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        with torch.no_grad():
            baseline_output = baseline_model(input_ids=input_ids, pixel_values=pixel_values)
        baseline_peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        # Measure optimized memory
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        optimized_start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        with torch.no_grad():
            optimized_output = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
        optimized_peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        
        print(f"  Baseline memory - Peak: {baseline_peak_memory/(1024**2):.2f}MB")
        print(f"  Optimized memory - Peak: {optimized_peak_memory/(1024**2):.2f}MB")
        
        # Verify outputs are similar (allowing for some variation due to sparsity)
        similarity = torch.cosine_similarity(
            baseline_output.flatten(),
            optimized_output.flatten(),
            dim=0
        ).item()
        
        print(f"  Output similarity: {similarity:.4f}")
        print(f"  V Memory efficiency test completed")
        
        return {
            'baseline_peak_memory': baseline_peak_memory,
            'optimized_peak_memory': optimized_peak_memory,
            'similarity': similarity,
            'memory_saved_mb': (baseline_peak_memory - optimized_peak_memory) / (1024**2) if torch.cuda.is_available() else 0
        }
    except Exception as e:
        print(f"  X Memory efficiency test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_comprehensive_test_suite():
    """Run all tests in the comprehensive suite."""
    print("="*80)
    print("COMPREHENSIVE TEST AND BENCHMARK SUITE FOR QWEN3-VL MODEL COMPONENTS")
    print("="*80)
    
    tests = [
        ("Configuration Capacity", test_config_capacity),
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_model_forward_pass),
        ("Generation", test_model_generation),
    ]
    
    benchmarks = [
        ("Performance Benchmark", benchmark_model_performance),
        ("Memory Efficiency", test_memory_efficiency),
    ]
    
    results = {}
    
    # Run tests
    print("\nRunning Tests:")
    print("-" * 15)
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        success = test_func()
        results[test_name] = success
    
    # Run benchmarks
    print("\n\nRunning Benchmarks:")
    print("-" * 18)
    for bench_name, bench_func in benchmarks:
        print(f"\n{bench_name}:")
        result = bench_func()
        results[bench_name] = result
    
    # Summary
    print("\n" + "="*80)
    print("TEST AND BENCHMARK SUMMARY")
    print("="*80)
    
    passed_tests = 0
    total_tests = 0
    
    for test_name, result in results.items():
        if isinstance(result, bool):  # Test result
            status = "PASS" if result else "FAIL"
            symbol = "V" if result else "X"
            print(f"{symbol} {test_name}: {status}")
            if result:
                passed_tests += 1
            total_tests += 1
        elif result is not None:  # Benchmark result
            print(f"V {test_name}: Completed")
            passed_tests += 1
            total_tests += 1
        else:  # Benchmark failed
            print(f"X {test_name}: Failed")
            total_tests += 1
    
    print(f"\nOverall: {passed_tests}/{total_tests} components passed")
    
    if passed_tests == total_tests:
        print("\nSUCCESS: ALL TESTS AND BENCHMARKS PASSED! Qwen3-VL model components are working correctly.")
        return True
    else:
        print(f"\nFAILURE: {total_tests - passed_tests} COMPONENT(S) FAILED!")
        return False


def main():
    """Main function to execute the test and benchmark suite."""
    success = run_comprehensive_test_suite()
    
    # Save results to JSON file
    try:
        with open('test_benchmark_results.json', 'w') as f:
            json.dump({"overall_success": success, "timestamp": time.time()}, f, indent=2)
        print(f"\nResults saved to 'test_benchmark_results.json'")
    except Exception as e:
        print(f"\nCould not save results to JSON: {e}")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)