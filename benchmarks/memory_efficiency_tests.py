"""
Memory Efficiency Tests for Qwen3-VL Model Optimizations
This module implements tests to verify memory usage reduction from various optimizations.
"""
import sys
import os
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
class MemoryEfficiencyConfig:
    """Configuration for memory efficiency tests"""
    # Model parameters
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 8
    vocab_size: int = 1000
    max_position_embeddings: int = 128
    
    # Memory test parameters
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    test_durations: List[int] = None  # in seconds
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4]
        if self.sequence_lengths is None:
            self.sequence_lengths = [64, 128, 256]
        if self.test_durations is None:
            self.test_durations = [5, 10, 15]


class MemoryEfficiencyTester:
    """Class to test memory efficiency of optimizations"""
    
    def __init__(self, config: MemoryEfficiencyConfig = None):
        self.config = config or MemoryEfficiencyConfig()
    
    def create_models_with_specific_optimizations(self) -> Dict[str, Tuple[nn.Module, Qwen3VLConfig]]:
        """Create models with specific optimizations enabled"""
        models = {}
        
        # Baseline model (no optimizations)
        baseline_config = Qwen3VLConfig()
        baseline_config.hidden_size = self.config.hidden_size
        baseline_config.num_hidden_layers = self.config.num_hidden_layers
        baseline_config.num_attention_heads = self.config.num_attention_heads
        baseline_config.vocab_size = self.config.vocab_size
        baseline_config.max_position_embeddings = self.config.max_position_embeddings
        
        # Disable all optimizations
        baseline_config.use_sparsity = False
        baseline_config.use_gradient_checkpointing = False
        baseline_config.use_moe = False
        baseline_config.use_flash_attention_2 = False
        baseline_config.use_dynamic_sparse_attention = False
        baseline_config.use_adaptive_depth = False
        baseline_config.use_context_adaptive_positional_encoding = False
        baseline_config.use_conditional_feature_extraction = False
        
        baseline_model = Qwen3VLForConditionalGeneration(baseline_config)
        baseline_model.eval()
        models['baseline'] = (baseline_model, baseline_config)
        
        # Model with sparsity only
        sparsity_config = Qwen3VLConfig()
        sparsity_config.hidden_size = self.config.hidden_size
        sparsity_config.num_hidden_layers = self.config.num_hidden_layers
        sparsity_config.num_attention_heads = self.config.num_attention_heads
        sparsity_config.vocab_size = self.config.vocab_size
        sparsity_config.max_position_embeddings = self.config.max_position_embeddings
        
        sparsity_config.use_sparsity = True
        sparsity_config.sparsity_ratio = 0.5
        sparsity_config.exit_threshold = 0.75
        # Disable other optimizations
        sparsity_config.use_gradient_checkpointing = False
        sparsity_config.use_moe = False
        sparsity_config.use_flash_attention_2 = False
        sparsity_config.use_dynamic_sparse_attention = False
        sparsity_config.use_adaptive_depth = False
        sparsity_config.use_context_adaptive_positional_encoding = False
        sparsity_config.use_conditional_feature_extraction = False
        
        sparsity_model = Qwen3VLForConditionalGeneration(sparsity_config)
        sparsity_model.eval()
        sparsity_model.load_state_dict(baseline_model.state_dict(), strict=False)
        models['sparsity_only'] = (sparsity_model, sparsity_config)
        
        # Model with gradient checkpointing only
        grad_config = Qwen3VLConfig()
        grad_config.hidden_size = self.config.hidden_size
        grad_config.num_hidden_layers = self.config.num_hidden_layers
        grad_config.num_attention_heads = self.config.num_attention_heads
        grad_config.vocab_size = self.config.vocab_size
        grad_config.max_position_embeddings = self.config.max_position_embeddings
        
        grad_config.use_gradient_checkpointing = True
        # Disable other optimizations
        grad_config.use_sparsity = False
        grad_config.use_moe = False
        grad_config.use_flash_attention_2 = False
        grad_config.use_dynamic_sparse_attention = False
        grad_config.use_adaptive_depth = False
        grad_config.use_context_adaptive_positional_encoding = False
        grad_config.use_conditional_feature_extraction = False
        
        grad_model = Qwen3VLForConditionalGeneration(grad_config)
        grad_model.eval()
        grad_model.load_state_dict(baseline_model.state_dict(), strict=False)
        models['grad_checkpoint_only'] = (grad_model, grad_config)
        
        # Model with MoE only
        moe_config = Qwen3VLConfig()
        moe_config.hidden_size = self.config.hidden_size
        moe_config.num_hidden_layers = self.config.num_hidden_layers
        moe_config.num_attention_heads = self.config.num_attention_heads
        moe_config.vocab_size = self.config.vocab_size
        moe_config.max_position_embeddings = self.config.max_position_embeddings
        
        moe_config.use_moe = True
        moe_config.moe_num_experts = 4
        moe_config.moe_top_k = 2
        # Disable other optimizations
        moe_config.use_sparsity = False
        moe_config.use_gradient_checkpointing = False
        moe_config.use_flash_attention_2 = False
        moe_config.use_dynamic_sparse_attention = False
        moe_config.use_adaptive_depth = False
        moe_config.use_context_adaptive_positional_encoding = False
        moe_config.use_conditional_feature_extraction = False
        
        moe_model = Qwen3VLForConditionalGeneration(moe_config)
        moe_model.eval()
        moe_model.load_state_dict(baseline_model.state_dict(), strict=False)
        models['moe_only'] = (moe_model, moe_config)
        
        # Model with all optimizations
        all_config = Qwen3VLConfig()
        all_config.hidden_size = self.config.hidden_size
        all_config.num_hidden_layers = self.config.num_hidden_layers
        all_config.num_attention_heads = self.config.num_attention_heads
        all_config.vocab_size = self.config.vocab_size
        all_config.max_position_embeddings = self.config.max_position_embeddings
        
        all_config.use_sparsity = True
        all_config.sparsity_ratio = 0.5
        all_config.exit_threshold = 0.75
        all_config.use_gradient_checkpointing = True
        all_config.use_moe = True
        all_config.moe_num_experts = 4
        all_config.moe_top_k = 2
        all_config.use_flash_attention_2 = True
        all_config.use_dynamic_sparse_attention = True
        all_config.use_adaptive_depth = True
        all_config.use_context_adaptive_positional_encoding = True
        all_config.use_conditional_feature_extraction = True
        
        all_model = Qwen3VLForConditionalGeneration(all_config)
        all_model.eval()
        all_model.load_state_dict(baseline_model.state_dict(), strict=False)
        models['all_optimizations'] = (all_model, all_config)
        
        return models
    
    def measure_memory_usage(self, model: nn.Module, input_ids: torch.Tensor, pixel_values: torch.Tensor, 
                            device: torch.device) -> Dict[str, float]:
        """Measure memory usage for a single forward pass"""
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
        
        # Run forward pass
        with torch.no_grad():
            _ = model(input_ids=input_ids, pixel_values=pixel_values)
        
        # Record memory after forward pass
        cpu_memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            peak_gpu_memory = torch.cuda.max_memory_reserved() / (1024**2)  # MB
        else:
            gpu_memory_after = 0
            peak_gpu_memory = 0
        
        return {
            'cpu_memory_initial_mb': initial_cpu_memory,
            'cpu_memory_after_mb': cpu_memory_after,
            'cpu_memory_increase_mb': cpu_memory_after - initial_cpu_memory,
            'gpu_memory_initial_mb': initial_gpu_memory if torch.cuda.is_available() else 0,
            'gpu_memory_after_mb': gpu_memory_after if torch.cuda.is_available() else 0,
            'gpu_memory_increase_mb': gpu_memory_after - initial_gpu_memory if torch.cuda.is_available() else 0,
            'gpu_peak_memory_mb': peak_gpu_memory if torch.cuda.is_available() else 0
        }
    
    def test_memory_efficiency_by_optimization(self, device: torch.device) -> Dict[str, Any]:
        """Test memory efficiency of different optimizations"""
        print("Testing memory efficiency by optimization type...")
        
        # Create models with different optimizations
        models = self.create_models_with_specific_optimizations()
        
        memory_results = {}
        
        # Test with a standard input size
        batch_size, seq_len = 2, 128
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
        pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
        
        for model_name, (model, config) in models.items():
            print(f"  Testing {model_name}...")
            
            model = model.to(device)
            
            # Measure memory usage
            memory_usage = self.measure_memory_usage(model, input_ids, pixel_values, device)
            
            memory_results[model_name] = {
                'config': {
                    'use_sparsity': config.use_sparsity,
                    'use_gradient_checkpointing': config.use_gradient_checkpointing,
                    'use_moe': config.use_moe,
                    'use_flash_attention_2': config.use_flash_attention_2,
                    'use_dynamic_sparse_attention': config.use_dynamic_sparse_attention,
                    'use_adaptive_depth': config.use_adaptive_depth,
                    'use_context_adaptive_positional_encoding': config.use_context_adaptive_positional_encoding,
                    'use_conditional_feature_extraction': config.use_conditional_feature_extraction
                },
                'memory_usage': memory_usage
            }
        
        return memory_results
    
    def test_memory_efficiency_by_input_size(self, device: torch.device) -> Dict[str, Any]:
        """Test memory efficiency across different input sizes"""
        print("Testing memory efficiency by input size...")
        
        # Create model with all optimizations
        _, all_model, _, _ = self.create_models_with_specific_optimizations()['all_optimizations']
        all_model = all_model.to(device)
        all_model.eval()
        
        # Create baseline model
        _, baseline_model, _, _ = self.create_models_with_specific_optimizations()['baseline']
        baseline_model = baseline_model.to(device)
        baseline_model.eval()
        
        size_results = {}
        
        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                print(f"  Testing batch_size={batch_size}, seq_len={seq_len}...")
                
                # Create inputs
                input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
                pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
                
                # Test baseline model
                baseline_memory = self.measure_memory_usage(baseline_model, input_ids, pixel_values, device)
                
                # Test optimized model
                optimized_memory = self.measure_memory_usage(all_model, input_ids, pixel_values, device)
                
                config_key = f"batch_{batch_size}_seq_{seq_len}"
                size_results[config_key] = {
                    'input_size': {
                        'batch_size': batch_size,
                        'sequence_length': seq_len
                    },
                    'baseline_memory': baseline_memory,
                    'optimized_memory': optimized_memory,
                    'improvement': {
                        'cpu_memory_reduction_mb': baseline_memory['cpu_memory_increase_mb'] - optimized_memory['cpu_memory_increase_mb'],
                        'gpu_memory_reduction_mb': baseline_memory['gpu_peak_memory_mb'] - optimized_memory['gpu_peak_memory_mb'],
                        'cpu_memory_reduction_percent': (
                            (baseline_memory['cpu_memory_increase_mb'] - optimized_memory['cpu_memory_increase_mb']) / 
                            baseline_memory['cpu_memory_increase_mb'] * 100
                            if baseline_memory['cpu_memory_increase_mb'] != 0 else 0
                        ),
                        'gpu_memory_reduction_percent': (
                            (baseline_memory['gpu_peak_memory_mb'] - optimized_memory['gpu_peak_memory_mb']) / 
                            baseline_memory['gpu_peak_memory_mb'] * 100
                            if baseline_memory['gpu_peak_memory_mb'] != 0 and torch.cuda.is_available() else 0
                        )
                    }
                }
        
        return size_results
    
    def test_memory_efficiency_over_time(self, device: torch.device) -> Dict[str, Any]:
        """Test memory efficiency over extended periods"""
        print("Testing memory efficiency over time...")
        
        # Create models
        _, baseline_model, _, _ = self.create_models_with_specific_optimizations()['baseline']
        _, optimized_model, _, _ = self.create_models_with_specific_optimizations()['all_optimizations']
        
        baseline_model = baseline_model.to(device)
        optimized_model = optimized_model.to(device)
        baseline_model.eval()
        optimized_model.eval()
        
        time_results = {}
        
        for duration in self.config.test_durations:
            print(f"  Testing for {duration} seconds...")
            
            # Test baseline model
            baseline_start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            batch_count = 0
            seq_len = 64
            
            while time.time() - start_time < duration:
                batch_size = 1
                input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
                pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
                
                with torch.no_grad():
                    _ = baseline_model(input_ids=input_ids, pixel_values=pixel_values)
                
                batch_count += 1
            
            baseline_end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            if torch.cuda.is_available():
                baseline_peak_gpu_memory = torch.cuda.max_memory_reserved() / (1024**2)  # MB
            else:
                baseline_peak_gpu_memory = 0
            
            # Test optimized model
            optimized_start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            start_time = time.time()
            opt_batch_count = 0
            
            while time.time() - start_time < duration:
                batch_size = 1
                input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(device)
                pixel_values = torch.randn(batch_size, 3, 224, 224).to(device)
                
                with torch.no_grad():
                    _ = optimized_model(input_ids=input_ids, pixel_values=pixel_values)
                
                opt_batch_count += 1
            
            optimized_end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
            if torch.cuda.is_available():
                optimized_peak_gpu_memory = torch.cuda.max_memory_reserved() / (1024**2)  # MB
            else:
                optimized_peak_gpu_memory = 0
            
            duration_key = f"{duration}s"
            time_results[duration_key] = {
                'duration_seconds': duration,
                'baseline': {
                    'batches_processed': batch_count,
                    'cpu_memory_start_mb': baseline_start_memory,
                    'cpu_memory_end_mb': baseline_end_memory,
                    'cpu_memory_increase_mb': baseline_end_memory - baseline_start_memory,
                    'gpu_peak_memory_mb': baseline_peak_gpu_memory
                },
                'optimized': {
                    'batches_processed': opt_batch_count,
                    'cpu_memory_start_mb': optimized_start_memory,
                    'cpu_memory_end_mb': optimized_end_memory,
                    'cpu_memory_increase_mb': optimized_end_memory - optimized_start_memory,
                    'gpu_peak_memory_mb': optimized_peak_gpu_memory
                },
                'improvement': {
                    'cpu_memory_reduction_mb': (baseline_end_memory - baseline_start_memory) - (optimized_end_memory - optimized_start_memory),
                    'gpu_memory_reduction_mb': baseline_peak_gpu_memory - optimized_peak_gpu_memory,
                    'cpu_memory_reduction_percent': (
                        ((baseline_end_memory - baseline_start_memory) - (optimized_end_memory - optimized_start_memory)) / 
                        (baseline_end_memory - baseline_start_memory) * 100
                        if (baseline_end_memory - baseline_start_memory) != 0 else 0
                    ),
                    'gpu_memory_reduction_percent': (
                        (baseline_peak_gpu_memory - optimized_peak_gpu_memory) / 
                        baseline_peak_gpu_memory * 100
                        if baseline_peak_gpu_memory != 0 and torch.cuda.is_available() else 0
                    )
                }
            }
        
        return time_results
    
    def run_all_memory_efficiency_tests(self, device: torch.device = None) -> Dict[str, Any]:
        """Run all memory efficiency tests"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("=" * 80)
        print("MEMORY EFFICIENCY TESTS FOR QWEN3-VL OPTIMIZATIONS")
        print("=" * 80)
        
        print(f"Using device: {device}")
        
        results = {}
        
        # Run memory efficiency by optimization type
        results['by_optimization_type'] = self.test_memory_efficiency_by_optimization_type(device)
        
        # Run memory efficiency by input size
        results['by_input_size'] = self.test_memory_efficiency_by_input_size(device)
        
        # Run memory efficiency over time
        results['over_time'] = self.test_memory_efficiency_over_time(device)
        
        # Generate summary
        summary = self.generate_summary(results)
        
        print("\n" + "=" * 80)
        print("MEMORY EFFICIENCY TEST SUMMARY")
        print("=" * 80)
        
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        return {
            'results': results,
            'summary': summary
        }
    
    def generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of memory efficiency test results"""
        summary = {}
        
        # Summary for optimization type results
        if 'by_optimization_type' in results:
            opt_results = results['by_optimization_type']
            
            # Calculate memory savings for each optimization
            baseline_gpu_peak = opt_results['baseline']['memory_usage']['gpu_peak_memory_mb']
            
            for opt_name, data in opt_results.items():
                if opt_name != 'baseline':
                    gpu_reduction = baseline_gpu_peak - data['memory_usage']['gpu_peak_memory_mb']
                    gpu_reduction_percent = (gpu_reduction / baseline_gpu_peak) * 100 if baseline_gpu_peak > 0 else 0
                    summary[f'{opt_name}_gpu_memory_reduction_mb'] = gpu_reduction
                    summary[f'{opt_name}_gpu_memory_reduction_percent'] = gpu_reduction_percent
        
        # Summary for input size results
        if 'by_input_size' in results:
            size_results = results['by_input_size']
            
            # Calculate average memory improvements
            gpu_reductions = [data['improvement']['gpu_memory_reduction_percent'] for data in size_results.values()]
            cpu_reductions = [data['improvement']['cpu_memory_reduction_percent'] for data in size_results.values()]
            
            avg_gpu_reduction = np.mean(gpu_reductions) if gpu_reductions else 0
            avg_cpu_reduction = np.mean(cpu_reductions) if cpu_reductions else 0
            
            summary['avg_gpu_memory_reduction_percent_by_size'] = avg_gpu_reduction
            summary['avg_cpu_memory_reduction_percent_by_size'] = avg_cpu_reduction
            summary['max_gpu_memory_reduction_percent'] = max(gpu_reductions) if gpu_reductions else 0
        
        # Summary for time results
        if 'over_time' in results:
            time_results = results['over_time']
            
            # Calculate average memory improvements over time
            gpu_reductions = [data['improvement']['gpu_memory_reduction_percent'] for data in time_results.values()]
            cpu_reductions = [data['improvement']['cpu_memory_reduction_percent'] for data in time_results.values()]
            
            avg_gpu_reduction = np.mean(gpu_reductions) if gpu_reductions else 0
            avg_cpu_reduction = np.mean(cpu_reductions) if cpu_reductions else 0
            
            summary['avg_gpu_memory_reduction_percent_over_time'] = avg_gpu_reduction
            summary['avg_cpu_memory_reduction_percent_over_time'] = avg_cpu_reduction
        
        # Overall memory efficiency assessment
        overall_gpu_improvement = summary.get('all_optimizations_gpu_memory_reduction_percent', 0)
        memory_efficiency_good = overall_gpu_improvement > 10  # At least 10% improvement
        
        summary['memory_efficiency_improvement_significant'] = memory_efficiency_good
        summary['overall_memory_efficiency_score'] = overall_gpu_improvement
        
        return summary


def run_memory_efficiency_tests():
    """Run all memory efficiency tests"""
    config = MemoryEfficiencyConfig()
    tester = MemoryEfficiencyTester(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = tester.run_all_memory_efficiency_tests(device)
    
    # Save results
    Path("benchmark_results").mkdir(parents=True, exist_ok=True)
    with open("benchmark_results/memory_efficiency_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_memory_efficiency_tests()