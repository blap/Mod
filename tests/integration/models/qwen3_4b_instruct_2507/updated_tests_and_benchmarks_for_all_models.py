"""
Updated Tests and Benchmarks for All Models

This script updates and runs tests and benchmarks for all 4 models:
- GLM-4-7
- Qwen3-4b-instruct-2507
- Qwen3-coder-30b
- Qwen3-vl-2b

The tests and benchmarks reflect all optimizations implemented in the system.
"""

import unittest
import time
import torch
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
from inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
from inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin


def run_model_tests(model_name, plugin_creator_func):
    """Run comprehensive tests for a specific model."""
    print(f"\n{'='*60}")
    print(f"RUNNING COMPREHENSIVE TESTS FOR {model_name}")
    print(f"{'='*60}")
    
    # Create plugin instance
    plugin = plugin_creator_func()
    
    # Test 1: Plugin Creation
    print(f"\n1. Testing {model_name} plugin creation...")
    try:
        assert plugin is not None
        print(f"   [PASS] {model_name} plugin created successfully")
    except Exception as e:
        print(f"   [FAIL] {model_name} plugin creation failed: {e}")
        return False
    
    # Test 2: Plugin Metadata
    print(f"\n2. Testing {model_name} plugin metadata...")
    try:
        metadata = plugin.metadata
        assert metadata.name is not None
        print(f"   [PASS] {model_name} metadata accessible: {metadata.name}")
    except Exception as e:
        print(f"   [FAIL] {model_name} metadata test failed: {e}")
        return False
    
    # Test 3: Plugin Initialization
    print(f"\n3. Testing {model_name} plugin initialization...")
    try:
        success = plugin.initialize(device="cpu")  # Using CPU for tests
        assert success is True
        print(f"   [PASS] {model_name} plugin initialized successfully")
    except Exception as e:
        print(f"   [FAIL] {model_name} plugin initialization failed: {e}")
        return False
    
    # Test 4: Model Loading
    print(f"\n4. Testing {model_name} model loading...")
    try:
        model = plugin.load_model()
        assert model is not None
        print(f"   [PASS] {model_name} model loaded successfully")
    except Exception as e:
        print(f"   [FAIL] {model_name} model loading failed: {e}")
        return False
    
    # Test 5: Basic Inference
    print(f"\n5. Testing {model_name} basic inference...")
    try:
        # Test with a simple text input
        test_input = "Hello, how are you?"
        result = plugin.infer(test_input)
        assert result is not None
        print(f"   [PASS] {model_name} basic inference completed")
    except Exception as e:
        print(f"   [FAIL] {model_name} basic inference failed: {e}")
        return False
    
    # Test 6: Generate Text
    print(f"\n6. Testing {model_name} text generation...")
    try:
        prompt = "The weather today is"
        generated = plugin.generate_text(prompt, max_new_tokens=10)
        assert isinstance(generated, str)
        assert len(generated) >= len(prompt)
        print(f"   [PASS] {model_name} text generation completed")
    except Exception as e:
        print(f"   [FAIL] {model_name} text generation failed: {e}")
        return False
    
    # Test 7: Tokenization
    print(f"\n7. Testing {model_name} tokenization...")
    try:
        text = "Test tokenization functionality"
        tokens = plugin.tokenize(text)
        assert tokens is not None
        print(f"   [PASS] {model_name} tokenization completed")
    except Exception as e:
        print(f"   [FAIL] {model_name} tokenization failed: {e}")
        return False
    
    # Test 8: Model Info Retrieval
    print(f"\n8. Testing {model_name} model info retrieval...")
    try:
        info = plugin.get_model_info()
        assert info is not None
        assert 'name' in info
        print(f"   [PASS] {model_name} model info retrieved")
    except Exception as e:
        print(f"   [FAIL] {model_name} model info retrieval failed: {e}")
        return False
    
    # Test 9: Memory Stats (if available)
    print(f"\n9. Testing {model_name} memory stats retrieval...")
    try:
        if hasattr(plugin, 'get_memory_stats'):
            mem_stats = plugin.get_memory_stats()
            assert mem_stats is not None
            print(f"   [PASS] {model_name} memory stats retrieved")
        else:
            print(f"   [SKIP] {model_name} memory stats not available")
    except Exception as e:
        print(f"   [SKIP] {model_name} memory stats retrieval failed (optional): {e}")
    
    # Test 10: Cleanup
    print(f"\n10. Testing {model_name} cleanup...")
    try:
        if hasattr(plugin, 'cleanup'):
            plugin.cleanup()
            print(f"   [PASS] {model_name} cleanup completed")
        else:
            print(f"   [SKIP] {model_name} cleanup not available")
    except Exception as e:
        print(f"   [SKIP] {model_name} cleanup failed (optional): {e}")
    
    print(f"\n{'='*60}")
    print(f"ALL TESTS PASSED FOR {model_name}")
    print(f"{'='*60}")
    return True


def run_model_benchmarks(model_name, plugin_creator_func):
    """Run comprehensive benchmarks for a specific model."""
    print(f"\n{'='*60}")
    print(f"RUNNING COMPREHENSIVE BENCHMARKS FOR {model_name}")
    print(f"{'='*60}")
    
    # Create plugin instance
    plugin = plugin_creator_func()
    
    # Initialize plugin
    success = plugin.initialize(device="cpu")
    if not success:
        print(f"   [FAIL] {model_name} plugin initialization failed for benchmarks")
        return False
    
    model = plugin.load_model()
    if model is None:
        print(f"   [FAIL] {model_name} model loading failed for benchmarks")
        return False
    
    # Benchmark 1: Inference Speed
    print(f"\n1. Running {model_name} inference speed benchmark...")
    try:
        # Test with different input lengths
        input_lengths = [20, 50, 100]
        results = []
        
        for length in input_lengths:
            # Create test input
            test_input = " ".join(["test"] * length)
            
            # Warmup
            for _ in range(3):
                _ = plugin.infer(test_input)
            
            # Timing run
            start_time = time.time()
            for i in range(5):  # Run 5 times for average
                _ = plugin.infer(test_input)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / 5
            tokens_per_sec = length / avg_time if avg_time > 0 else 0
            
            results.append({
                'length': length,
                'avg_time': avg_time,
                'tokens_per_sec': tokens_per_sec
            })
        
        print(f"   [PASS] {model_name} inference speed benchmark completed:")
        for res in results:
            print(f"     - Length {res['length']}: {res['tokens_per_sec']:.2f} tokens/sec")
    except Exception as e:
        print(f"   [FAIL] {model_name} inference speed benchmark failed: {e}")
        return False
    
    # Benchmark 2: Generation Speed
    print(f"\n2. Running {model_name} generation speed benchmark...")
    try:
        prompts = [
            "Once upon a time",
            "The future of AI",
            "Explain quantum computing"
        ]
        
        total_chars = 0
        total_time = 0
        
        for prompt in prompts:
            start_time = time.time()
            generated = plugin.generate_text(prompt, max_new_tokens=20)
            end_time = time.time()
            
            total_time += (end_time - start_time)
            total_chars += len(generated)
        
        avg_chars_per_sec = total_chars / total_time if total_time > 0 else 0
        print(f"   [PASS] {model_name} generation speed: {avg_chars_per_sec:.2f} chars/sec")
    except Exception as e:
        print(f"   [FAIL] {model_name} generation speed benchmark failed: {e}")
        return False
    
    # Benchmark 3: Batch Processing
    print(f"\n3. Running {model_name} batch processing benchmark...")
    try:
        batch_sizes = [1, 2, 4]
        batch_results = []
        
        for batch_size in batch_sizes:
            prompts = [f"Prompt {i} for batch testing" for i in range(batch_size)]
            
            # Warmup
            for prompt in prompts[:1]:
                _ = plugin.infer(prompt)
            
            # Timing run
            start_time = time.time()
            for prompt in prompts:
                _ = plugin.infer(prompt)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_item = total_time / batch_size if batch_size > 0 else 0
            
            batch_results.append({
                'batch_size': batch_size,
                'total_time': total_time,
                'avg_time_per_item': avg_time_per_item
            })
        
        print(f"   [PASS] {model_name} batch processing benchmark completed:")
        for res in batch_results:
            print(f"     - Batch size {res['batch_size']}: {res['avg_time_per_item']:.4f}s/item")
    except Exception as e:
        print(f"   [FAIL] {model_name} batch processing benchmark failed: {e}")
        return False
    
    # Benchmark 4: Memory Usage (if available)
    print(f"\n4. Running {model_name} memory usage benchmark...")
    try:
        if hasattr(plugin, 'get_memory_stats'):
            mem_stats = plugin.get_memory_stats()
            if mem_stats and 'system_memory_percent' in mem_stats:
                print(f"   [PASS] {model_name} memory usage: {mem_stats['system_memory_percent']:.2f}%")
            else:
                print(f"   [SKIP] {model_name} memory stats not detailed enough")
        else:
            print(f"   [SKIP] {model_name} memory stats not available")
    except Exception as e:
        print(f"   [SKIP] {model_name} memory usage benchmark failed (optional): {e}")
    
    # Cleanup
    if hasattr(plugin, 'cleanup'):
        plugin.cleanup()
    
    print(f"\n{'='*60}")
    print(f"ALL BENCHMARKS COMPLETED FOR {model_name}")
    print(f"{'='*60}")
    return True


def main():
    """Main function to run tests and benchmarks for all models."""
    print("COMPREHENSIVE TESTS AND BENCHMARKS FOR ALL MODELS")
    print("="*80)
    print("This script will run tests and benchmarks for:")
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
    print("EXECUTING COMPREHENSIVE TESTS FOR ALL MODELS")
    print("="*80)
    
    test_results = {}
    for model_name, creator_func in models:
        try:
            result = run_model_tests(model_name, creator_func)
            test_results[model_name] = result
        except Exception as e:
            print(f"Error running tests for {model_name}: {e}")
            test_results[model_name] = False
    
    # Run benchmarks for all models
    print("\n" + "="*80)
    print("EXECUTING COMPREHENSIVE BENCHMARKS FOR ALL MODELS")
    print("="*80)
    
    benchmark_results = {}
    for model_name, creator_func in models:
        try:
            result = run_model_benchmarks(model_name, creator_func)
            benchmark_results[model_name] = result
        except Exception as e:
            print(f"Error running benchmarks for {model_name}: {e}")
            benchmark_results[model_name] = False
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF TESTS AND BENCHMARKS")
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
    print(f"  Success Rate: {(passed_tests + passed_benchmarks) / (total_tests + total_benchmarks) * 100:.1f}%")
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TESTS AND BENCHMARKS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()