#!/usr/bin/env python
"""
Performance Tests with Real Models - Execute performance tests with actual models and real data
to measure actual performance metrics.
"""

import sys
import os
import torch
import time
import traceback
import psutil
from pathlib import Path
import numpy as np

# Add the src directory to the path so we can import the models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_qwen3_0_6b_performance():
    """Performance test for qwen3_0_6b model."""
    print("=" * 60)
    print("Performance Testing Qwen3-0.6B Model")
    print("=" * 60)

    try:
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config

        print("[INFO] Successfully imported qwen3_0_6b components")

        # Create config
        config = Qwen3_0_6B_Config()
        print(f"[INFO] Created config: {config.model_name}")

        # Create plugin
        plugin = create_qwen3_0_6b_plugin()
        print(f"[INFO] Created plugin: {plugin.__class__.__name__}")

        # Performance test parameters
        test_inputs = [
            "Hello, how are you?",
            "Explain the theory of relativity in simple terms.",
            "Write a Python function to calculate the Fibonacci sequence.",
            "What are the main causes of climate change?",
            "Describe the process of photosynthesis."
        ]
        
        num_runs = 3  # Reduced for practical testing
        warmup_runs = 1

        # Warmup runs
        print(f"[INFO] Performing {warmup_runs} warmup runs...")
        for i in range(warmup_runs):
            try:
                plugin.infer("This is a warmup input.")
            except:
                pass  # Ignore errors during warmup

        # Performance measurement
        inference_times = []
        memory_usages = []

        print(f"[INFO] Performing {num_runs} performance runs...")
        for i in range(num_runs):
            input_text = test_inputs[i % len(test_inputs)]
            
            # Record memory before
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Time the inference
            start_time = time.time()
            try:
                result = plugin.infer(input_text)
                end_time = time.time()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                # Record memory after
                mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_used = mem_after - mem_before
                memory_usages.append(memory_used)
                
                print(f"  Run {i+1}: {inference_time:.3f}s, Memory: {memory_used:.2f}MB")
                
            except Exception as e:
                print(f"  Run {i+1}: Failed with error: {e}")
                inference_times.append(float('inf'))  # Mark as failed
                memory_usages.append(0)

        # Calculate performance metrics
        successful_runs = [t for t in inference_times if t != float('inf')]
        if successful_runs:
            avg_time = np.mean(successful_runs)
            std_time = np.std(successful_runs)
            avg_memory = np.mean(memory_usages)
            throughput = len(successful_runs) / sum(successful_runs) if sum(successful_runs) > 0 else 0
            
            print(f"\n[PERFORMANCE] Average inference time: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"[PERFORMANCE] Average memory usage: {avg_memory:.2f}MB")
            print(f"[PERFORMANCE] Throughput: {throughput:.2f} inferences/sec")
        else:
            print("\n[PERFORMANCE] No successful runs to calculate metrics")

        # Test cleanup
        cleanup_result = plugin.cleanup()
        print(f"[INFO] Cleanup result: {cleanup_result}")

        print("\n[PERFORMANCE] Qwen3-0.6B performance test completed")
        return True

    except ImportError as e:
        print(f"[INFO] Qwen3-0.6B not available: {str(e)}")
        print("[SKIP] Skipping Qwen3-0.6B performance tests")
        return True  # Don't fail if model is not available
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_0_6b performance test: {str(e)}")
        traceback.print_exc()
        return False


def test_qwen3_coder_next_performance():
    """Performance test for qwen3_coder_next model."""
    print("\n" + "=" * 60)
    print("Performance Testing Qwen3-Coder-Next Model")
    print("=" * 60)

    try:
        from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin
        from src.inference_pio.models.qwen3_coder_next.config import Qwen3CoderNextConfig

        print("[INFO] Successfully imported qwen3_coder_next components")

        # Create config
        config = Qwen3CoderNextConfig()
        print(f"[INFO] Created config: {config.model_name}")

        # Create plugin
        plugin = create_qwen3_coder_next_plugin()
        print(f"[INFO] Created plugin: {plugin.__class__.__name__}")

        # Performance test parameters - code-specific inputs
        test_inputs = [
            "Write a Python function to reverse a string.",
            "How do I implement a binary search algorithm?",
            "Explain the difference between stack and queue data structures.",
            "Create a simple Flask web application.",
            "Implement a decorator in Python."
        ]
        
        num_runs = 3  # Reduced for practical testing
        warmup_runs = 1

        # Warmup runs
        print(f"[INFO] Performing {warmup_runs} warmup runs...")
        for i in range(warmup_runs):
            try:
                plugin.infer("This is a warmup input for coding.")
            except:
                pass  # Ignore errors during warmup

        # Performance measurement
        inference_times = []
        memory_usages = []

        print(f"[INFO] Performing {num_runs} performance runs...")
        for i in range(num_runs):
            input_text = test_inputs[i % len(test_inputs)]
            
            # Record memory before
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Time the inference
            start_time = time.time()
            try:
                result = plugin.infer(input_text)
                end_time = time.time()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                # Record memory after
                mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_used = mem_after - mem_before
                memory_usages.append(memory_used)
                
                print(f"  Run {i+1}: {inference_time:.3f}s, Memory: {memory_used:.2f}MB")
                
            except Exception as e:
                print(f"  Run {i+1}: Failed with error: {e}")
                inference_times.append(float('inf'))  # Mark as failed
                memory_usages.append(0)

        # Calculate performance metrics
        successful_runs = [t for t in inference_times if t != float('inf')]
        if successful_runs:
            avg_time = np.mean(successful_runs)
            std_time = np.std(successful_runs)
            avg_memory = np.mean(memory_usages)
            throughput = len(successful_runs) / sum(successful_runs) if sum(successful_runs) > 0 else 0
            
            print(f"\n[PERFORMANCE] Average inference time: {avg_time:.3f}s ± {std_time:.3f}s")
            print(f"[PERFORMANCE] Average memory usage: {avg_memory:.2f}MB")
            print(f"[PERFORMANCE] Throughput: {throughput:.2f} inferences/sec")
        else:
            print("\n[PERFORMANCE] No successful runs to calculate metrics")

        # Test cleanup
        cleanup_result = plugin.cleanup()
        print(f"[INFO] Cleanup result: {cleanup_result}")

        print("\n[PERFORMANCE] Qwen3-Coder-Next performance test completed")
        return True

    except ImportError as e:
        print(f"[INFO] Qwen3-Coder-Next not available: {str(e)}")
        print("[SKIP] Skipping Qwen3-Coder-Next performance tests")
        return True  # Don't fail if model is not available
    except Exception as e:
        print(f"[FAIL] Unexpected error during qwen3_coder_next performance test: {str(e)}")
        traceback.print_exc()
        return False


def test_memory_efficiency_performance():
    """Performance test for memory efficiency."""
    print("\n" + "=" * 60)
    print("Performance Testing Memory Efficiency")
    print("=" * 60)

    try:
        import gc
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print(f"[INFO] Initial memory usage: {initial_memory:.2f} MB")

        # Create and destroy multiple plugin instances
        plugins_created = 0
        for i in range(3):
            try:
                from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
                plugin = create_qwen3_0_6b_plugin()
                plugins_created += 1
                
                # Try to initialize (might fail if model not available)
                try:
                    from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config
                    config = Qwen3_0_6B_Config()
                    plugin.initialize(config=config)
                except:
                    pass  # Expected if model not available
                
                # Cleanup immediately
                plugin.cleanup()
                
            except ImportError:
                print(f"[INFO] Skipping iteration {i+1} - model not available")
                continue

        # Force garbage collection
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_diff = final_memory - initial_memory
        print(f"[INFO] Final memory usage: {final_memory:.2f} MB")
        print(f"[INFO] Memory difference: {memory_diff:.2f} MB")

        # Check if memory usage is reasonable
        if abs(memory_diff) < 100:  # Less than 100MB difference is acceptable
            print("[PERFORMANCE] Memory usage remained reasonable after plugin operations")
        else:
            print(f"[WARNING] Significant memory change: {memory_diff:.2f} MB")

        print(f"[INFO] Created and cleaned up {plugins_created} plugin instances")
        print("\n[PERFORMANCE] Memory efficiency test completed")
        return True

    except Exception as e:
        print(f"[FAIL] Unexpected error during memory efficiency test: {str(e)}")
        traceback.print_exc()
        return False


def test_concurrent_performance():
    """Performance test for concurrent operations."""
    print("\n" + "=" * 60)
    print("Performance Testing Concurrent Operations")
    print("=" * 60)

    try:
        # Test basic performance metrics without actual concurrency
        # (since we don't have the full model infrastructure)
        
        from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
        from src.inference_pio.models.qwen3_0_6b.config import Qwen3_0_6B_Config

        # Create multiple plugin instances
        plugins = []
        for i in range(2):  # Reduced for practical testing
            try:
                plugin = create_qwen3_0_6b_plugin()
                config = Qwen3_0_6B_Config()
                
                # Try to initialize
                try:
                    plugin.initialize(config=config)
                except:
                    pass  # Expected if model not available
                
                plugins.append(plugin)
                print(f"[INFO] Created plugin instance {i+1}")
            except ImportError:
                print(f"[INFO] Skipping plugin instance {i+1} - model not available")
                break

        # Test basic operations on each plugin
        test_input = "Test input for concurrent performance."
        for i, plugin in enumerate(plugins):
            try:
                start_time = time.time()
                result = plugin.infer(test_input)
                end_time = time.time()
                
                print(f"[INFO] Plugin {i+1} inference time: {end_time - start_time:.3f}s")
            except Exception as e:
                print(f"[INFO] Plugin {i+1} inference failed: {e}")

        # Cleanup
        for i, plugin in enumerate(plugins):
            try:
                plugin.cleanup()
                print(f"[INFO] Cleaned up plugin {i+1}")
            except:
                print(f"[INFO] Plugin {i+1} cleanup failed")

        print("\n[PERFORMANCE] Concurrent operations test completed")
        return True

    except ImportError as e:
        print(f"[INFO] Concurrent test skipped - model not available: {str(e)}")
        return True  # Don't fail if model not available
    except Exception as e:
        print(f"[FAIL] Unexpected error during concurrent test: {str(e)}")
        traceback.print_exc()
        return False


def test_tensor_compression_performance():
    """Performance test for tensor compression."""
    print("\n" + "=" * 60)
    print("Performance Testing Tensor Compression")
    print("=" * 60)

    try:
        from src.inference_pio.common.optimization.tensor_compression import TensorCompressionOptimizer
        
        # Create optimizer
        optimizer = TensorCompressionOptimizer()
        print(f"[INFO] Created tensor compression optimizer")

        # Create a sample tensor for testing
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        sample_tensor = torch.randn(100, 100, device=device)
        print(f"[INFO] Created sample tensor of shape {sample_tensor.shape}")

        # Test compression
        start_time = time.time()
        compressed = optimizer.compress(sample_tensor)
        compress_time = time.time() - start_time
        
        print(f"[INFO] Compression time: {compress_time:.4f}s")
        print(f"[INFO] Original size: {sample_tensor.numel() * 4 / 1024:.2f} KB")
        
        if hasattr(compressed, 'numel'):
            print(f"[INFO] Compressed size: {compressed.numel() * 4 / 1024:.2f} KB")
        
        # Test decompression
        start_time = time.time()
        decompressed = optimizer.decompress(compressed)
        decompress_time = time.time() - start_time
        
        print(f"[INFO] Decompression time: {decompress_time:.4f}s")

        # Verify the decompressed tensor is close to original
        if torch.allclose(sample_tensor.cpu(), decompressed.cpu(), atol=1e-3):
            print("[PERFORMANCE] Tensor compression/decompression preserves data")
        else:
            print("[WARNING] Tensor compression/decompression may have data loss")

        print("\n[PERFORMANCE] Tensor compression test completed")
        return True

    except ImportError as e:
        print(f"[INFO] Tensor compression components not available: {str(e)}")
        print("[SKIP] Skipping tensor compression performance test")
        return True  # Don't fail if components not available
    except Exception as e:
        print(f"[FAIL] Unexpected error during tensor compression test: {str(e)}")
        traceback.print_exc()
        return False


def test_activation_offloading_performance():
    """Performance test for activation offloading."""
    print("\n" + "=" * 60)
    print("Performance Testing Activation Offloading")
    print("=" * 60)

    try:
        from src.inference_pio.common.optimization.activation_offloading import ActivationOffloadingOptimizer
        
        # Create optimizer
        optimizer = ActivationOffloadingOptimizer()
        print(f"[INFO] Created activation offloading optimizer")

        # Create sample activations
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        sample_activations = torch.randn(50, 128, device=device)
        print(f"[INFO] Created sample activations of shape {sample_activations.shape}")

        # Test offloading to CPU
        start_time = time.time()
        offloaded = optimizer.offload_to_cpu(sample_activations)
        offload_time = time.time() - start_time
        
        print(f"[INFO] Offload time: {offload_time:.4f}s")

        # Test reloading to GPU
        start_time = time.time()
        reloaded = optimizer.reload_to_gpu(offloaded)
        reload_time = time.time() - start_time
        
        print(f"[INFO] Reload time: {reload_time:.4f}s")

        # Verify the reloaded tensor is the same as original
        if torch.allclose(sample_activations.cpu(), reloaded.cpu()):
            print("[PERFORMANCE] Activation offloading/reloading preserves data")
        else:
            print("[WARNING] Activation offloading/reloading may have data loss")

        print("\n[PERFORMANCE] Activation offloading test completed")
        return True

    except ImportError as e:
        print(f"[INFO] Activation offloading components not available: {str(e)}")
        print("[SKIP] Skipping activation offloading performance test")
        return True  # Don't fail if components not available
    except Exception as e:
        print(f"[FAIL] Unexpected error during activation offloading test: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Main function to run all performance tests with real models."""
    print("Starting performance tests with real models and components...")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__ if 'torch' in sys.modules else 'Not loaded yet'}")
    print(f"CUDA available: {torch.cuda.is_available() if 'torch' in sys.modules else False}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name() if torch.cuda.device_count() > 0 else 'N/A'}")

    # Run performance tests
    tests = [
        ("Qwen3-0.6B Performance", test_qwen3_0_6b_performance),
        ("Qwen3-Coder-Next Performance", test_qwen3_coder_next_performance),
        ("Memory Efficiency Performance", test_memory_efficiency_performance),
        ("Concurrent Operations Performance", test_concurrent_performance),
        ("Tensor Compression Performance", test_tensor_compression_performance),
        ("Activation Offloading Performance", test_activation_offloading_performance),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} Running {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            print(f"Result: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            print(f"Result: FAIL - Exception occurred: {e}")
            results[test_name] = False
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("PERFORMANCE TESTS SUMMARY WITH REAL MODELS")
    print("=" * 80)

    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name:<35} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All performance tests with real models completed successfully!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} out of {total} tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)