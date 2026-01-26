"""
Standardized Benchmark for Model Comparison - Qwen3-4B-Instruct-2507 vs Others

This module benchmarks the performance comparison between Qwen3-4B-Instruct-2507 and other models.
"""

import time
import unittest
import torch
import psutil
import os
from inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
from inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
from inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin


class BenchmarkQwen34BInstruct2507Comparison(unittest.TestCase):
    """Benchmark cases for comparing Qwen3-4B-Instruct-2507 with other models."""

    def setUp(self):
        """Set up benchmark fixtures before each test method."""
        # Create plugins for all models
        self.models = {}

        # Qwen3-4B-Instruct-2507
        self.models['Qwen3-4B-Instruct-2507'] = create_qwen3_4b_instruct_2507_plugin()
        success = self.models['Qwen3-4B-Instruct-2507'].initialize(device="cpu")
        self.assertTrue(success)

        # GLM-4.7
        try:
            self.models['GLM-4.7-Flash'] = create_glm_4_7_flash_plugin()
            success = self.models['GLM-4.7-Flash'].initialize(device="cpu")
            if not success:
                del self.models['GLM-4.7-Flash']
        except Exception:
            pass  # Skip if not available

        # Qwen3-Coder-30B
        try:
            self.models['Qwen3-Coder-30B'] = create_qwen3_coder_30b_plugin()
            success = self.models['Qwen3-Coder-30B'].initialize(device="cpu")
            if not success:
                del self.models['Qwen3-Coder-30B']
        except Exception:
            pass  # Skip if not available

        # Qwen3-VL-2B
        try:
            self.models['Qwen3-VL-2B'] = create_qwen3_vl_2b_instruct_plugin()
            success = self.models['Qwen3-VL-2B'].initialize(device="cpu")
            if not success:
                del self.models['Qwen3-VL-2B']
        except Exception:
            pass  # Skip if not available

    def benchmark_model_inference_speed(self, model_plugin, model_name, input_length=50, num_iterations=5):
        """Benchmark inference speed for a specific model."""
        # Load model
        model = model_plugin.load_model()
        self.assertIsNotNone(model)
        
        # Prepare input
        input_ids = torch.randint(0, 1000, (1, input_length))
        
        # Warmup
        for _ in range(3):
            _ = model_plugin.infer(input_ids)
        
        # Timing run
        start_time = time.time()
        for i in range(num_iterations):
            _ = model_plugin.infer(input_ids)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_inference = total_time / num_iterations
        tokens_per_second = input_length / avg_time_per_inference if avg_time_per_inference > 0 else float('inf')
        
        return {
            'model_name': model_name,
            'total_time': total_time,
            'avg_time_per_inference': avg_time_per_inference,
            'tokens_per_second': tokens_per_second,
            'num_iterations': num_iterations
        }

    def benchmark_model_memory_usage(self, model_plugin, model_name):
        """Benchmark memory usage for a specific model."""
        process = psutil.Process()
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load model
        model = model_plugin.load_model()
        self.assertIsNotNone(model)
        
        # Get memory after loading
        memory_after_load = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_after_load - baseline_memory
        
        # Run inference
        input_ids = torch.randint(0, 1000, (1, 50))
        result = model_plugin.infer(input_ids)
        self.assertIsNotNone(result)
        
        # Get memory after inference
        memory_after_inference = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'model_name': model_name,
            'baseline_memory': baseline_memory,
            'memory_after_load': memory_after_load,
            'memory_increase': memory_increase,
            'memory_after_inference': memory_after_inference
        }

    def test_inference_speed_comparison(self):
        """Compare inference speed between models."""
        print(f"\n{'='*60}")
        print("MODEL INFERENCE SPEED COMPARISON")
        print(f"{'='*60}")
        
        results = []
        for model_name, model_plugin in self.models.items():
            try:
                result = self.benchmark_model_inference_speed(model_plugin, model_name, 
                                                            input_length=50, num_iterations=3)
                results.append(result)
                
                print(f"{model_name:20} | Tokens/sec: {result['tokens_per_second']:8.2f} | "
                      f"Avg time: {result['avg_time_per_inference']:.4f}s")
            except Exception as e:
                print(f"{model_name:20} | Error: {str(e)[:50]}")
        
        # Basic sanity check - all models that ran should have positive throughput
        for result in results:
            if result['tokens_per_second'] != float('inf'):
                self.assertGreater(result['tokens_per_second'], 0)

    def test_memory_usage_comparison(self):
        """Compare memory usage between models."""
        print(f"\n{'='*60}")
        print("MODEL MEMORY USAGE COMPARISON")
        print(f"{'='*60}")
        
        results = []
        for model_name, model_plugin in self.models.items():
            try:
                result = self.benchmark_model_memory_usage(model_plugin, model_name)
                results.append(result)
                
                print(f"{model_name:20} | Load increase: {result['memory_increase']:6.1f}MB | "
                      f"After inference: {result['memory_after_inference']:6.1f}MB")
            except Exception as e:
                print(f"{model_name:20} | Error: {str(e)[:50]}")
        
        # Basic sanity check - all models that ran should have consumed some memory
        for result in results:
            self.assertGreaterEqual(result['memory_increase'], 0)

    def test_generation_quality_comparison(self):
        """Compare generation quality between models."""
        print(f"\n{'='*60}")
        print("MODEL GENERATION QUALITY COMPARISON")
        print(f"{'='*60}")
        
        prompt = "The future of artificial intelligence"
        
        for model_name, model_plugin in self.models.items():
            try:
                # Load model if not already loaded
                if not model_plugin.is_loaded:
                    model = model_plugin.load_model()
                    self.assertIsNotNone(model)
                
                # Generate text
                generated = model_plugin.generate_text(prompt, max_new_tokens=30)
                
                print(f"{model_name:20} | Generated: {generated[:60]}...")
            except Exception as e:
                print(f"{model_name:20} | Error: {str(e)[:50]}")

    def test_model_loading_time_comparison(self):
        """Compare model loading times between models."""
        print(f"\n{'='*60}")
        print("MODEL LOADING TIME COMPARISON")
        print(f"{'='*60}")
        
        results = []
        for model_name, model_plugin in self.models.items():
            try:
                # Measure loading time
                start_time = time.time()
                model = model_plugin.load_model()
                end_time = time.time()
                
                loading_time = end_time - start_time
                results.append((model_name, loading_time))
                
                print(f"{model_name:20} | Loading time: {loading_time:.4f}s")
                
                # Basic sanity check
                self.assertGreater(loading_time, 0)
            except Exception as e:
                print(f"{model_name:20} | Error: {str(e)[:50]}")

    def test_parameter_efficiency_comparison(self):
        """Compare parameter efficiency between models."""
        print(f"\n{'='*60}")
        print("MODEL PARAMETER EFFICIENCY COMPARISON")
        print(f"{'='*60}")
        
        for model_name, model_plugin in self.models.items():
            try:
                # Load model
                model = model_plugin.load_model()
                self.assertIsNotNone(model)
                
                # Count parameters
                param_count = sum(p.numel() for p in model.parameters())
                
                # Do a quick performance test
                input_ids = torch.randint(0, 1000, (1, 30))
                
                start_time = time.time()
                _ = model_plugin.infer(input_ids)
                inference_time = time.time() - start_time
                
                params_millions = param_count / 1_000_000
                efficiency_ratio = params_millions / inference_time if inference_time > 0 else float('inf')
                
                print(f"{model_name:20} | Params: {params_millions:6.2f}M | "
                      f"Time: {inference_time:.4f}s | Efficiency: {efficiency_ratio:8.2f}")
                
            except Exception as e:
                print(f"{model_name:20} | Error: {str(e)[:50]}")

    def test_batch_processing_comparison(self):
        """Compare batch processing capabilities between models."""
        print(f"\n{'='*60}")
        print("MODEL BATCH PROCESSING COMPARISON")
        print(f"{'='*60}")
        
        batch_sizes = [1, 2, 4]
        
        for model_name, model_plugin in self.models.items():
            print(f"\n{model_name}:")
            try:
                # Load model
                model = model_plugin.load_model()
                self.assertIsNotNone(model)
                
                for batch_size in batch_sizes:
                    input_ids = torch.randint(0, 1000, (batch_size, 25))
                    
                    start_time = time.time()
                    result = model_plugin.infer(input_ids)
                    end_time = time.time()
                    
                    processing_time = end_time - start_time
                    
                    print(f"  Batch {batch_size:2d}: {processing_time:.4f}s")
                    
                    # Basic sanity check
                    self.assertGreater(processing_time, 0)
            except Exception as e:
                print(f"  Error: {str(e)[:50]}")

    def test_model_info_comparison(self):
        """Compare model information between models."""
        print(f"\n{'='*60}")
        print("MODEL INFORMATION COMPARISON")
        print(f"{'='*60}")
        
        for model_name, model_plugin in self.models.items():
            try:
                info = model_plugin.get_model_info()
                print(f"{model_name:20} | Type: {info.get('model_type', 'N/A'):10} | "
                      f"Family: {info.get('model_family', 'N/A'):15}")
            except Exception as e:
                print(f"{model_name:20} | Error: {str(e)[:50]}")

    def tearDown(self):
        """Clean up after each test method."""
        for model_name, model_plugin in self.models.items():
            if hasattr(model_plugin, 'cleanup') and model_plugin.is_loaded:
                model_plugin.cleanup()


if __name__ == '__main__':
    unittest.main()