"""
Scalability Tests for Qwen3-VL Model Optimizations
This module implements tests to ensure optimizations work across different input sizes and scales.
"""
import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Tuple
import gc
import psutil
from dataclasses import dataclass
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.qwen3_vl.config.config import Qwen3VLConfig
from models.modeling_qwen3_vl_phase2 import Qwen3VLForConditionalGeneration


@dataclass
class ScalabilityTestConfig:
    """Configuration for scalability tests"""
    # Model parameters
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    vocab_size: int = 1000
    max_position_embeddings: int = 512  # Increased for scalability testing
    
    # Scalability test parameters
    batch_size_range: List[int] = None
    sequence_length_range: List[int] = None
    image_size_range: List[int] = None
    
    def __post_init__(self):
        if self.batch_size_range is None:
            self.batch_size_range = [1, 2, 4, 8, 16]  # Test different batch sizes
        if self.sequence_length_range is None:
            self.sequence_length_range = [64, 128, 256, 512]  # Test different sequence lengths
        if self.image_size_range is None:
            self.image_size_range = [224, 336, 448]  # Test different image sizes


class ScalabilityTester:
    """Class to test scalability of optimizations across different input sizes"""
    
    def __init__(self, config: ScalabilityTestConfig = None):
        self.config = config or ScalabilityTestConfig()
    
    def create_model(self, use_optimizations: bool = True) -> Tuple[nn.Module, Qwen3VLConfig]:
        """Create a model with or without optimizations"""
        qwen_config = Qwen3VLConfig()
        qwen_config.hidden_size = self.config.hidden_size
        qwen_config.num_hidden_layers = self.config.num_hidden_layers
        qwen_config.num_attention_heads = self.config.num_attention_heads
        qwen_config.vocab_size = self.config.vocab_size
        qwen_config.max_position_embeddings = self.config.max_position_embeddings
        
        if use_optimizations:
            # Enable all optimizations
            qwen_config.use_sparsity = True
            qwen_config.sparsity_ratio = 0.5
            qwen_config.exit_threshold = 0.75
            qwen_config.use_gradient_checkpointing = True
            qwen_config.use_moe = True
            qwen_config.moe_num_experts = 4
            qwen_config.moe_top_k = 2
            qwen_config.use_flash_attention_2 = True
            qwen_config.use_dynamic_sparse_attention = True
            qwen_config.use_adaptive_depth = True
            qwen_config.use_context_adaptive_positional_encoding = True
            qwen_config.use_conditional_feature_extraction = True
        else:
            # Disable all optimizations
            qwen_config.use_sparsity = False
            qwen_config.use_gradient_checkpointing = False
            qwen_config.use_moe = False
            qwen_config.use_flash_attention_2 = False
            qwen_config.use_dynamic_sparse_attention = False
            qwen_config.use_adaptive_depth = False
            qwen_config.use_context_adaptive_positional_encoding = False
            qwen_config.use_conditional_feature_extraction = False
        
        model = Qwen3VLForConditionalGeneration(qwen_config)
        model.eval()
        return model, qwen_config
    
    def measure_performance_and_memory(self, model: nn.Module, input_ids: torch.Tensor, 
                                      pixel_values: torch.Tensor, device: torch.device) -> Dict[str, Any]:
        """Measure performance and memory usage for a single forward pass"""
        # Clear cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Record initial memory
        initial_cpu_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        if torch.cuda.is_available():
            initial_gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            torch.cuda.reset_peak_memory_stats()
        else:
            initial_gpu_memory = 0
        
        # Measure performance
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        with torch.no_grad():
            _ = model(input_ids=input_ids, pixel_values=pixel_values)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        # Record memory after forward pass
        cpu_memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            peak_gpu_memory = torch.cuda.max_memory_reserved() / (1024**2)  # MB
        else:
            gpu_memory_after = 0
            peak_gpu_memory = 0
        
        return {
            'execution_time': end_time - start_time,
            'cpu_memory_initial_mb': initial_cpu_memory,
            'cpu_memory_after_mb': cpu_memory_after,
            'cpu_memory_increase_mb': cpu_memory_after - initial_cpu_memory,
            'gpu_memory_initial_mb': initial_gpu_memory if torch.cuda.is_available() else 0,
            'gpu_memory_after_mb': gpu_memory_after if torch.cuda.is_available() else 0,
            'gpu_memory_increase_mb': gpu_memory_after - initial_gpu_memory if torch.cuda.is_available() else 0,
            'gpu_peak_memory_mb': peak_gpu_memory if torch.cuda.is_available() else 0
        }
    
    def test_scalability_by_batch_size(self, device: torch.device) -> Dict[str, Any]:
        """Test scalability across different batch sizes"""
        print("Testing scalability by batch size...")
        
        # Create optimized model
        model, _ = self.create_model(use_optimizations=True)
        model = model.to(device)
        
        # Use fixed sequence length and image size for batch size testing
        seq_len = 128
        image_size = 224
        
        batch_scalability_results = {}
        
        for batch_size in self.config.batch_size_range:
            print(f"  Testing batch_size={batch_size}...")
            
            # Create inputs
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
            pixel_values = torch.randn(batch_size, 3, image_size, image_size).to(device)
            
            # Measure performance and memory
            metrics = self.measure_performance_and_memory(model, input_ids, pixel_values, device)
            
            batch_scalability_results[f"batch_{batch_size}"] = {
                'input_config': {
                    'batch_size': batch_size,
                    'sequence_length': seq_len,
                    'image_size': image_size
                },
                'metrics': metrics
            }
        
        # Analyze scalability
        scalability_analysis = self.analyze_scalability(batch_scalability_results, 'batch_size')
        
        return {
            'results': batch_scalability_results,
            'analysis': scalability_analysis
        }
    
    def test_scalability_by_sequence_length(self, device: torch.device) -> Dict[str, Any]:
        """Test scalability across different sequence lengths"""
        print("Testing scalability by sequence length...")
        
        # Create optimized model
        model, _ = self.create_model(use_optimizations=True)
        model = model.to(device)
        
        # Use fixed batch size and image size for sequence length testing
        batch_size = 2
        image_size = 224
        
        seq_scalability_results = {}
        
        for seq_len in self.config.sequence_length_range:
            print(f"  Testing sequence_length={seq_len}...")
            
            # Create inputs
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
            pixel_values = torch.randn(batch_size, 3, image_size, image_size).to(device)
            
            # Measure performance and memory
            metrics = self.measure_performance_and_memory(model, input_ids, pixel_values, device)
            
            seq_scalability_results[f"seq_{seq_len}"] = {
                'input_config': {
                    'batch_size': batch_size,
                    'sequence_length': seq_len,
                    'image_size': image_size
                },
                'metrics': metrics
            }
        
        # Analyze scalability
        scalability_analysis = self.analyze_scalability(seq_scalability_results, 'sequence_length')
        
        return {
            'results': seq_scalability_results,
            'analysis': scalability_analysis
        }
    
    def test_scalability_by_image_size(self, device: torch.device) -> Dict[str, Any]:
        """Test scalability across different image sizes"""
        print("Testing scalability by image size...")
        
        # Create optimized model
        model, _ = self.create_model(use_optimizations=True)
        model = model.to(device)
        
        # Use fixed batch size and sequence length for image size testing
        batch_size = 1
        seq_len = 64
        
        image_scalability_results = {}
        
        for image_size in self.config.image_size_range:
            print(f"  Testing image_size={image_size}x{image_size}...")
            
            # Create inputs
            input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
            pixel_values = torch.randn(batch_size, 3, image_size, image_size).to(device)
            
            # Measure performance and memory
            metrics = self.measure_performance_and_memory(model, input_ids, pixel_values, device)
            
            image_scalability_results[f"image_{image_size}x{image_size}"] = {
                'input_config': {
                    'batch_size': batch_size,
                    'sequence_length': seq_len,
                    'image_size': image_size
                },
                'metrics': metrics
            }
        
        # Analyze scalability
        scalability_analysis = self.analyze_scalability(image_scalability_results, 'image_size')
        
        return {
            'results': image_scalability_results,
            'analysis': scalability_analysis
        }
    
    def analyze_scalability(self, results: Dict[str, Any], dimension: str) -> Dict[str, Any]:
        """Analyze scalability based on test results"""
        # Extract metrics for analysis
        input_sizes = []
        execution_times = []
        peak_memory_mb = []
        
        for key, data in results.items():
            config = data['input_config']
            metrics = data['metrics']
            
            if dimension == 'batch_size':
                input_sizes.append(config['batch_size'])
            elif dimension == 'sequence_length':
                input_sizes.append(config['sequence_length'])
            elif dimension == 'image_size':
                input_sizes.append(config['image_size'] ** 2)  # Use area for image size
            
            execution_times.append(metrics['execution_time'])
            peak_memory_mb.append(metrics['gpu_peak_memory_mb'])
        
        # Calculate scalability metrics
        input_sizes = np.array(input_sizes)
        execution_times = np.array(execution_times)
        peak_memory_mb = np.array(peak_memory_mb)
        
        # Calculate time complexity (how execution time scales with input size)
        if len(input_sizes) > 1:
            # Calculate scaling factor (slope of log-log plot for power law)
            log_input = np.log(input_sizes[input_sizes > 0])
            log_time = np.log(execution_times[input_sizes > 0])
            
            if len(log_input) > 1:
                scaling_factor = np.polyfit(log_input, log_time, 1)[0] if len(log_input) > 1 else 1.0
            else:
                scaling_factor = 1.0
            
            # Calculate efficiency (time per unit of input)
            time_per_unit = execution_times / input_sizes
            avg_efficiency = np.mean(time_per_unit)
        else:
            scaling_factor = 1.0
            avg_efficiency = execution_times[0] / input_sizes[0] if len(input_sizes) > 0 and input_sizes[0] > 0 else 0
        
        # Determine if scaling is efficient
        scaling_efficient = scaling_factor < 2.0  # Should be better than O(n^2)
        
        analysis = {
            'dimension': dimension,
            'input_sizes_tested': input_sizes.tolist(),
            'execution_times': execution_times.tolist(),
            'peak_memory_mb': peak_memory_mb.tolist(),
            'scaling_factor': scaling_factor,
            'avg_time_per_unit': avg_efficiency,
            'scaling_efficient': scaling_efficient,
            'scalability_score': min(10.0, 10.0 / max(scaling_factor, 0.1))  # Higher score for better scaling
        }
        
        return analysis
    
    def run_all_scalability_tests(self, device: torch.device = None) -> Dict[str, Any]:
        """Run all scalability tests"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("=" * 80)
        print("SCALABILITY TESTS FOR QWEN3-VL OPTIMIZATIONS")
        print("=" * 80)
        
        print(f"Using device: {device}")
        
        results = {}
        
        # Run scalability tests by batch size
        results['by_batch_size'] = self.test_scalability_by_batch_size(device)
        
        # Run scalability tests by sequence length
        results['by_sequence_length'] = self.test_scalability_by_sequence_length(device)
        
        # Run scalability tests by image size
        results['by_image_size'] = self.test_scalability_by_image_size(device)
        
        # Generate summary
        summary = self.generate_summary(results)
        
        print("\n" + "=" * 80)
        print("SCALABILITY TEST SUMMARY")
        print("=" * 80)
        
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return {
            'results': results,
            'summary': summary
        }
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of scalability test results"""
        summary = {}
        
        # Summary for batch size scalability
        if 'by_batch_size' in results:
            batch_analysis = results['by_batch_size']['analysis']
            summary['batch_size_scaling_factor'] = batch_analysis['scaling_factor']
            summary['batch_size_scalability_efficient'] = batch_analysis['scaling_efficient']
            summary['batch_size_scalability_score'] = batch_analysis['scalability_score']
        
        # Summary for sequence length scalability
        if 'by_sequence_length' in results:
            seq_analysis = results['by_sequence_length']['analysis']
            summary['sequence_length_scaling_factor'] = seq_analysis['scaling_factor']
            summary['sequence_length_scalability_efficient'] = seq_analysis['scaling_efficient']
            summary['sequence_length_scalability_score'] = seq_analysis['scalability_score']
        
        # Summary for image size scalability
        if 'by_image_size' in results:
            image_analysis = results['by_image_size']['analysis']
            summary['image_size_scaling_factor'] = image_analysis['scaling_factor']
            summary['image_size_scalability_efficient'] = image_analysis['scaling_efficient']
            summary['image_size_scalability_score'] = image_analysis['scalability_score']
        
        # Overall scalability assessment
        batch_efficient = summary.get('batch_size_scalability_efficient', True)
        seq_efficient = summary.get('sequence_length_scalability_efficient', True)
        image_efficient = summary.get('image_size_scalability_efficient', True)
        
        overall_scalability_score = np.mean([
            summary.get('batch_size_scalability_score', 0),
            summary.get('sequence_length_scalability_score', 0),
            summary.get('image_size_scalability_score', 0)
        ])
        
        summary['overall_scalability_efficient'] = batch_efficient and seq_efficient and image_efficient
        summary['overall_scalability_score'] = overall_scalability_score
        
        return summary


def run_scalability_tests():
    """Run all scalability tests"""
    config = ScalabilityTestConfig()
    tester = ScalabilityTester(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = tester.run_all_scalability_tests(device)
    
    # Save results
    Path("benchmark_results").mkdir(parents=True, exist_ok=True)
    with open("benchmark_results/scalability_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_scalability_tests()