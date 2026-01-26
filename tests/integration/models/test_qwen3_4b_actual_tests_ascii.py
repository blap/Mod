"""
Actual Implementation: Update and Run Tests and Benchmarks for All Models

This script runs actual tests and benchmarks for all 4 models:
- GLM-4-7
- Qwen3-4b-instruct-2507
- Qwen3-coder-30b
- Qwen3-vl-2b
"""

import time
import torch
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the plugin creation functions
try:
    from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
    print("SUCCESS: Successfully imported GLM-4-7 plugin")
except Exception as e:
    print(f"ERROR: Failed to import GLM-4-7 plugin: {e}")
    create_glm_4_7_plugin = None

try:
    from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
    print("SUCCESS: Successfully imported Qwen3-4B-Instruct-2507 plugin")
except Exception as e:
    print(f"ERROR: Failed to import Qwen3-4B-Instruct-2507 plugin: {e}")
    create_qwen3_4b_instruct_2507_plugin = None

try:
    from src.inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
    print("SUCCESS: Successfully imported Qwen3-Coder-30B plugin")
except Exception as e:
    print(f"ERROR: Failed to import Qwen3-Coder-30B plugin: {e}")
    create_qwen3_coder_30b_plugin = None

try:
    from src.inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
    print("SUCCESS: Successfully imported Qwen3-VL-2B plugin")
except Exception as e:
    print(f"ERROR: Failed to import Qwen3-VL-2B plugin: {e}")
    create_qwen3_vl_2b_instruct_plugin = None


def run_basic_test(model_name, plugin_creator_func):
    """Run basic functionality test for a specific model."""
    if plugin_creator_func is None:
        print(f"SKIPPED: {model_name} (import failed)")
        return False
    
    print(f"\n{'='*60}")
    print(f"RUNNING BASIC TEST FOR {model_name}")
    print(f"{'='*60}")
    
    try:
        # Create plugin instance
        plugin = plugin_creator_func()
        if plugin is None:
            print(f"   RESULT: FAIL - {model_name} plugin creation returned None")
            return False
        
        print(f"   RESULT: PASS - {model_name} plugin created successfully")
        
        # Initialize plugin
        success = plugin.initialize(device="cpu")
        if not success:
            print(f"   RESULT: FAIL - {model_name} plugin initialization failed")
            return False
        
        print(f"   RESULT: PASS - {model_name} plugin initialized successfully")
        
        # Load model
        model = plugin.load_model()
        if model is None:
            print(f"   RESULT: FAIL - {model_name} model loading returned None")
            return False
        
        print(f"   RESULT: PASS - {model_name} model loaded successfully")
        
        # Basic inference test
        test_input = "Hello, how are you?"
        result = plugin.infer(test_input)
        if result is None:
            print(f"   RESULT: FAIL - {model_name} inference returned None")
            return False
        
        print(f"   RESULT: PASS - {model_name} basic inference completed")
        print(f"   Details: Result type: {type(result)}, length: {len(str(result)) if result else 0}")
        
        # Text generation test
        generated = plugin.generate_text("The weather today is", max_new_tokens=10)
        if generated is None:
            print(f"   RESULT: FAIL - {model_name} text generation returned None")
            return False
        
        print(f"   RESULT: PASS - {model_name} text generation completed")
        print(f"   Details: Generated: {generated[:50]}..." if len(generated) > 50 else f"   Details: Generated: {generated}")
        
        # Get model info
        info = plugin.get_model_info()
        if info is None:
            print(f"   RESULT: FAIL - {model_name} model info retrieval failed")
            return False
        
        print(f"   RESULT: PASS - {model_name} model info retrieved")
        print(f"   Details: Model name in info: {info.get('name', 'N/A')}")
        
        print(f"\nOVERALL RESULT: SUCCESS - ALL BASIC TESTS PASSED FOR {model_name}")
        return True
        
    except Exception as e:
        print(f"   RESULT: FAIL - {model_name} test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_basic_benchmark(model_name, plugin_creator_func):
    """Run basic benchmark for a specific model."""
    if plugin_creator_func is None:
        print(f"SKIPPED: {model_name} (import failed)")
        return False
    
    print(f"\n{'='*60}")
    print(f"RUNNING BASIC BENCHMARK FOR {model_name}")
    print(f"{'='*60}")
    
    try:
        # Create plugin instance
        plugin = plugin_creator_func()
        if plugin is None:
            print(f"   RESULT: FAIL - {model_name} plugin creation returned None")
            return False
        
        # Initialize plugin
        success = plugin.initialize(device="cpu")
        if not success:
            print(f"   RESULT: FAIL - {model_name} plugin initialization failed")
            return False
        
        print(f"   RESULT: PASS - {model_name} plugin initialized successfully")
        
        # Benchmark 1: Inference Speed
        print(f"\n1. Running {model_name} inference speed benchmark...")
        test_inputs = [
            "Short input",
            "Medium length input for testing purposes",
            "Longer input to test performance with more tokens for comprehensive evaluation"
        ]
        
        for i, test_input in enumerate(test_inputs):
            # Warmup
            for _ in range(2):
                _ = plugin.infer(test_input)
            
            # Timing run
            start_time = time.time()
            for j in range(3):  # Run 3 times for average
                _ = plugin.infer(test_input)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / 3
            tokens_per_sec = len(test_input.split()) / avg_time if avg_time > 0 else 0
            
            print(f"   Input {i+1} ('{test_input[:15]}...'): {avg_time:.4f}s avg, {tokens_per_sec:.2f} tokens/sec")
        
        print(f"   RESULT: PASS - {model_name} inference speed benchmark completed")
        
        # Benchmark 2: Generation Speed
        print(f"\n2. Running {model_name} generation speed benchmark...")
        prompts = [
            "Once upon a time",
            "The future of AI",
            "Explain quantum computing briefly"
        ]
        
        total_chars = 0
        total_time = 0
        
        for prompt in prompts:
            start_time = time.time()
            generated = plugin.generate_text(prompt, max_new_tokens=20)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_chars += len(generated) if generated else 0
        
        avg_chars_per_sec = total_chars / total_time if total_time > 0 else 0
        print(f"   RESULT: PASS - {model_name} generation: {avg_chars_per_sec:.2f} chars/sec across {len(prompts)} prompts")
        
        print(f"\nOVERALL RESULT: SUCCESS - ALL BENCHMARKS COMPLETED FOR {model_name}")
        return True
        
    except Exception as e:
        print(f"   RESULT: FAIL - {model_name} benchmark failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run tests and benchmarks for all models."""
    print("ACTUAL TESTS AND BENCHMARKS FOR ALL MODELS")
    print("="*80)
    print("Running actual tests and benchmarks for:")
    print("- GLM-4-7")
    print("- Qwen3-4b-instruct-2507")
    print("- Qwen3-coder-30b")
    print("- Qwen3-vl-2b")
    print("="*80)
    
    # Define models and their creators
    models = [
        ("GLM-4-7-Flash", create_glm_4_7_flash_plugin),
        ("Qwen3-4b-instruct-2507", create_qwen3_4b_instruct_2507_plugin),
        ("Qwen3-coder-30b", create_qwen3_coder_30b_plugin),
        ("Qwen3-vl-2b", create_qwen3_vl_2b_instruct_plugin)
    ]
    
    # Run tests for all models
    print("\n" + "="*80)
    print("EXECUTING BASIC TESTS FOR ALL MODELS")
    print("="*80)
    
    test_results = {}
    for model_name, creator_func in models:
        result = run_basic_test(model_name, creator_func)
        test_results[model_name] = result
    
    # Run benchmarks for all models
    print("\n" + "="*80)
    print("EXECUTING BASIC BENCHMARKS FOR ALL MODELS")
    print("="*80)
    
    benchmark_results = {}
    for model_name, creator_func in models:
        result = run_basic_benchmark(model_name, creator_func)
        benchmark_results[model_name] = result
    
    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY OF ACTUAL TESTS AND BENCHMARKS")
    print("="*80)
    
    print("\nTEST RESULTS:")
    for model, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {model:<25}: {status}")
    
    print("\nBENCHMARK RESULTS:")
    for model, result in benchmark_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {model:<25}: {status}")
    
    # Overall summary
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    total_benchmarks = len(benchmark_results)
    passed_benchmarks = sum(1 for result in benchmark_results.values() if result)
    
    print(f"\nOVERALL SUMMARY:")
    print(f"  Tests: {passed_tests}/{total_tests} passed")
    print(f"  Benchmarks: {passed_benchmarks}/{total_benchmarks} passed")
    print(f"  Total Success Rate: {(passed_tests + passed_benchmarks) / (total_tests + total_benchmarks) * 100:.1f}%")
    
    # Detailed results
    print(f"\nDETAILED RESULTS:")
    for model_name in [m[0] for m in models]:  # All model names
        test_result = test_results.get(model_name, "N/A")
        bench_result = benchmark_results.get(model_name, "N/A")
        print(f"  {model_name}:")
        print(f"    Test:     {'PASS' if test_result else 'FAIL' if test_result is not None else 'SKIPPED'}")
        print(f"    Benchmark: {'PASS' if bench_result else 'FAIL' if bench_result is not None else 'SKIPPED'}")
    
    print(f"\n{'='*80}")
    print("ACTUAL TESTS AND BENCHMARKS COMPLETED")
    print("All tests and benchmarks have been executed with actual results.")
    print("="*80)


if __name__ == "__main__":
    main()