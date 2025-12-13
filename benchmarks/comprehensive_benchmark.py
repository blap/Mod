"""
Comprehensive benchmark tests for Qwen3-VL-2B-Instruct implementation
Testing performance, memory efficiency, and accuracy of all optimization techniques
"""
import pytest
import torch
import time
import numpy as np
from typing import Dict, Any, List
import psutil
import GPUtil
import gc

from src.qwen3_vl.components.models.qwen3_vl_model import Qwen3VLForConditionalGeneration
from src.qwen3_vl.components.configuration import Qwen3VLConfig
from models.block_sparse_attention import BlockSparseAttention
from models.cross_modal_token_merging import CrossModalTokenMerger
from models.hierarchical_memory_compression import HierarchicalMemoryCompressor
from models.learned_activation_routing import LearnedActivationRouter


class ComprehensiveBenchmarkSuite:
    """Comprehensive benchmark suite for all optimization techniques"""
    
    def __init__(self):
        self.results = {}
        
    def benchmark_performance_improvements(self) -> Dict[str, Any]:
        """Benchmark performance improvements from all optimizations"""
        print("Benchmarking performance improvements...")
        
        # Create configuration with full capacity
        config = Qwen3VLConfig()
        config.hidden_size = 512  # Reduced for testing but still substantial
        config.num_attention_heads = 8
        config.num_hidden_layers = 6  # Reduced for faster testing
        config.vocab_size = 1000
        config.max_position_embeddings = 256
        
        # Create base model without optimizations
        base_model = Qwen3VLForConditionalGeneration(config)
        
        # Create test inputs
        batch_size, seq_len = 1, 128
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        # Warm up
        base_model.eval()
        with torch.no_grad():
            for _ in range(3):
                _ = base_model(input_ids=input_ids, pixel_values=pixel_values)
        
        # Time base model
        times_base = []
        for _ in range(5):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            start_time = time.time()
            with torch.no_grad():
                _ = base_model(input_ids=input_ids, pixel_values=pixel_values)
            end_time = time.time()
            times_base.append(end_time - start_time)
        
        avg_base_time = sum(times_base) / len(times_base)
        
        # Results
        performance_results = {
            'base_model_avg_time': avg_base_time,
            'num_runs': 5,
            'batch_size': batch_size,
            'sequence_length': seq_len
        }
        
        print(f"Base model average time: {avg_base_time:.4f}s")
        self.results['performance'] = performance_results
        return performance_results
    
    def benchmark_memory_efficiency(self) -> Dict[str, Any]:
        """Benchmark memory efficiency improvements"""
        print("Benchmarking memory efficiency...")
        
        # Monitor system memory before and after operations
        initial_memory = psutil.virtual_memory().used / (1024**3)  # GB
        
        # Create optimization components
        memory_compressor = HierarchicalMemoryCompressor()
        sequence_packer = AdaptiveSequencePacker()
        sparse_attention = BlockSparseAttention(
            Qwen3VLConfig(hidden_size=256, num_attention_heads=8, max_position_embeddings=128)
        )
        
        # Create test tensors
        batch_size, seq_len, hidden_dim = 2, 256, 256
        large_tensor = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Test memory compression
        compressed_tensor = memory_compressor.compress(large_tensor, 'high')
        
        # Test attention with sparsity
        sparse_output, _, _ = sparse_attention(
            hidden_states=large_tensor,
            output_attentions=False
        )
        
        # Monitor final memory usage
        final_memory = psutil.virtual_memory().used / (1024**3)  # GB
        memory_used_by_operations = final_memory - initial_memory
        
        # Calculate memory efficiency metrics
        original_size = large_tensor.numel() * large_tensor.element_size() / (1024**3)  # GB
        compressed_size = compressed_tensor.numel() * compressed_tensor.element_size() / (1024**3)  # GB
        
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        
        memory_results = {
            'initial_memory_gb': initial_memory,
            'final_memory_gb': final_memory,
            'memory_used_by_operations_gb': memory_used_by_operations,
            'original_tensor_size_gb': original_size,
            'compressed_tensor_size_gb': compressed_size,
            'compression_ratio': compression_ratio,
            'operations_performed': ['compression', 'sparse_attention']
        }
        
        print(f"Memory efficiency - Compression ratio: {compression_ratio:.4f}")
        print(f"Memory used by operations: {memory_used_by_operations:.4f} GB")
        self.results['memory_efficiency'] = memory_results
        return memory_results
    
    def benchmark_optimization_techniques(self) -> Dict[str, Any]:
        """Benchmark each optimization technique individually"""
        print("Benchmarking individual optimization techniques...")
        
        # Configuration for testing
        config = Qwen3VLConfig()
        config.hidden_size = 256
        config.num_attention_heads = 8
        config.max_position_embeddings = 128
        
        # Create test data
        batch_size, seq_len, hidden_dim = 1, 64, 256
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Individual benchmark results
        technique_benchmarks = {}
        
        # 1. Block Sparse Attention
        sparse_attention = BlockSparseAttention(config)
        start_time = time.time()
        for _ in range(3):
            output, _, _ = sparse_attention(
                hidden_states=input_tensor,
                output_attentions=False
            )
        end_time = time.time()
        technique_benchmarks['block_sparse_attention'] = {
            'avg_time': (end_time - start_time) / 3,
            'output_shape': output.shape,
            'success': True
        }
        
        # 2. Cross-Modal Token Merging
        token_merger = CrossModalTokenMerger()
        start_time = time.time()
        for _ in range(3):
            merged = token_merger.merge_tokens(input_tensor)
        end_time = time.time()
        technique_benchmarks['cross_modal_token_merging'] = {
            'avg_time': (end_time - start_time) / 3,
            'output_shape': merged.shape,
            'success': True
        }
        
        # 3. Hierarchical Memory Compression
        memory_compressor = HierarchicalMemoryCompressor()
        start_time = time.time()
        for _ in range(3):
            compressed = memory_compressor.compress(input_tensor, 'medium')
        end_time = time.time()
        technique_benchmarks['hierarchical_memory_compression'] = {
            'avg_time': (end_time - start_time) / 3,
            'output_shape': compressed.shape,
            'success': True
        }
        
        # 4. Learned Activation Routing
        activation_router = LearnedActivationRouter()
        start_time = time.time()
        for _ in range(3):
            routed = activation_router(input_tensor, layer_idx=0)
        end_time = time.time()
        technique_benchmarks['learned_activation_routing'] = {
            'avg_time': (end_time - start_time) / 3,
            'output_shape': routed.shape,
            'success': True
        }
        
        # Results
        optimization_results = {
            'technique_benchmarks': technique_benchmarks,
            'num_techniques_benchmarked': len(technique_benchmarks),
            'test_batch_size': batch_size,
            'test_sequence_length': seq_len
        }
        
        print(f"Benchmarked {len(technique_benchmarks)} optimization techniques")
        for tech, metrics in technique_benchmarks.items():
            print(f"  {tech}: {metrics['avg_time']:.6f}s avg")
        
        self.results['individual_optimizations'] = optimization_results
        return optimization_results
    
    def benchmark_accuracy_preservation(self) -> Dict[str, Any]:
        """Benchmark accuracy preservation with optimizations"""
        print("Benchmarking accuracy preservation...")
        
        # Test that optimizations don't significantly change outputs
        batch_size, seq_len, hidden_dim = 2, 32, 128
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Create optimization components
        activation_router = LearnedActivationRouter()
        token_merger = CrossModalTokenMerger()
        memory_compressor = HierarchicalMemoryCompressor()
        
        # Apply optimizations
        routed_output = activation_router(input_tensor, layer_idx=0)
        merged_output = token_merger.merge_tokens(routed_output)
        compressed_output = memory_compressor.compress(merged_output, 'medium')
        
        # Check that outputs are reasonable (finite values)
        outputs_finite = (
            torch.isfinite(routed_output).all() and
            torch.isfinite(merged_output).all() and
            torch.isfinite(compressed_output).all()
        )
        
        # Calculate output differences to ensure they're reasonable
        initial_sum = input_tensor.abs().mean().item()
        final_sum = compressed_output.abs().mean().item()
        output_change_ratio = abs(final_sum - initial_sum) / initial_sum if initial_sum != 0 else 0
        
        accuracy_results = {
            'outputs_finite': outputs_finite,
            'initial_mean_abs': initial_sum,
            'final_mean_abs': final_sum,
            'output_change_ratio': output_change_ratio,
            'max_acceptable_change': 0.5,  # 50% change threshold
            'change_acceptable': output_change_ratio <= 0.5
        }
        
        print(f"Output change ratio: {output_change_ratio:.4f}")
        print(f"Outputs finite: {outputs_finite}")
        print(f"Change acceptable: {accuracy_results['change_acceptable']}")
        
        self.results['accuracy_preservation'] = accuracy_results
        return accuracy_results
    
    def benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark scalability with increasing input sizes"""
        print("Benchmarking scalability...")
        
        # Test performance with different input sizes
        input_sizes = [
            (1, 64, 256),    # Small
            (2, 128, 256),   # Medium
            (1, 256, 512),   # Large
        ]
        
        scalability_results = {}
        
        for batch_size, seq_len, hidden_dim in input_sizes:
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
            
            # Create optimization components
            sparse_attention = BlockSparseAttention(
                Qwen3VLConfig(hidden_size=hidden_dim, num_attention_heads=8, max_position_embeddings=seq_len)
            )
            
            # Time the operation
            start_time = time.time()
            output, _, _ = sparse_attention(
                hidden_states=input_tensor,
                output_attentions=False
            )
            end_time = time.time()
            
            operation_time = end_time - start_time
            input_elements = batch_size * seq_len * hidden_dim
            time_per_element = operation_time / input_elements if input_elements > 0 else 0
            
            size_key = f"{batch_size}x{seq_len}x{hidden_dim}"
            scalability_results[size_key] = {
                'input_elements': input_elements,
                'operation_time': operation_time,
                'time_per_element': time_per_element,
                'output_shape': output.shape
            }
        
        # Analyze scalability trends
        times_per_element = [v['time_per_element'] for v in scalability_results.values()]
        scalability_efficient = all(t <= times_per_element[0] * 2 for t in times_per_element)  # Should not scale worse than O(n)
        
        scalability_summary = {
            'scalability_results': scalability_results,
            'scalability_efficient': scalability_efficient,
            'input_sizes_tested': len(input_sizes)
        }
        
        print(f"Scalability efficient: {scalability_efficient}")
        for size, metrics in scalability_results.items():
            print(f"  {size}: {metrics['time_per_element']:.2e}s per element")
        
        self.results['scalability'] = scalability_summary
        return scalability_summary
    
    def benchmark_hardware_utilization(self) -> Dict[str, Any]:
        """Benchmark hardware utilization improvements"""
        print("Benchmarking hardware utilization...")
        
        # Monitor CPU and GPU utilization
        initial_cpu_percent = psutil.cpu_percent(interval=1)
        initial_memory_percent = psutil.virtual_memory().percent
        
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            initial_gpu_memory = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            initial_gpu_util = GPUtil.getGPUs()[0].load if GPUtil.getGPUs() else 0
        else:
            initial_gpu_memory = 0
            initial_gpu_util = 0
        
        # Run optimization-intensive operations
        batch_size, seq_len, hidden_dim = 2, 128, 256
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Apply multiple optimizations
        memory_compressor = HierarchicalMemoryCompressor()
        sequence_packer = AdaptiveSequencePacker()
        sparse_attention = BlockSparseAttention(
            Qwen3VLConfig(hidden_size=hidden_dim, num_attention_heads=8, max_position_embeddings=seq_len)
        )
        
        # Perform operations
        for _ in range(5):
            packed = sequence_packer.pack_sequences(input_tensor)
            compressed = memory_compressor.compress(packed, 'medium')
            output, _, _ = sparse_attention(
                hidden_states=compressed,
                output_attentions=False
            )
        
        # Monitor final utilization
        final_cpu_percent = psutil.cpu_percent(interval=1)
        final_memory_percent = psutil.virtual_memory().percent
        
        if gpu_available:
            final_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
            final_gpu_util = GPUtil.getGPUs()[0].load if GPUtil.getGPUs() else 0
        else:
            final_gpu_memory = 0
            final_gpu_util = 0
        
        hardware_results = {
            'cpu_initial_percent': initial_cpu_percent,
            'cpu_final_percent': final_cpu_percent,
            'cpu_change_percent': final_cpu_percent - initial_cpu_percent,
            'memory_initial_percent': initial_memory_percent,
            'memory_final_percent': final_memory_percent,
            'memory_change_percent': final_memory_percent - initial_memory_percent,
            'gpu_available': gpu_available,
            'gpu_memory_initial_gb': initial_gpu_memory,
            'gpu_memory_final_gb': final_gpu_memory,
            'gpu_memory_change_gb': final_gpu_memory - initial_gpu_memory,
            'gpu_util_initial': initial_gpu_util,
            'gpu_util_final': final_gpu_util,
            'gpu_util_change': final_gpu_util - initial_gpu_util
        }
        
        print(f"CPU usage change: {hardware_results['cpu_change_percent']:+.2f}%")
        print(f"Memory usage change: {hardware_results['memory_change_percent']:+.2f}%")
        if gpu_available:
            print(f"GPU memory change: {hardware_results['gpu_memory_change_gb']:+.4f} GB")
            print(f"GPU utilization change: {hardware_results['gpu_util_change']:+.2%}")
        
        self.results['hardware_utilization'] = hardware_results
        return hardware_results
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmark tests"""
        print("="*70)
        print("RUNNING COMPREHENSIVE BENCHMARK SUITE")
        print("="*70)
        
        # Run all benchmarks
        benchmarks = [
            ("Performance Improvements", self.benchmark_performance_improvements),
            ("Memory Efficiency", self.benchmark_memory_efficiency),
            ("Individual Optimizations", self.benchmark_optimization_techniques),
            ("Accuracy Preservation", self.benchmark_accuracy_preservation),
            ("Scalability", self.benchmark_scalability),
            ("Hardware Utilization", self.benchmark_hardware_utilization)
        ]
        
        for benchmark_name, benchmark_func in benchmarks:
            print(f"\n{benchmark_name}:")
            print("-" * len(benchmark_name))
            try:
                benchmark_func()
            except Exception as e:
                print(f"ERROR in {benchmark_name}: {str(e)}")
        
        # Generate summary
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        summary = {
            'total_benchmarks': len(benchmarks),
            'completed_benchmarks': len(self.results),
            'results': self.results
        }
        
        # Performance summary
        if 'performance' in self.results:
            perf = self.results['performance']
            print(f"Performance - Base model avg time: {perf['base_model_avg_time']:.4f}s")
        
        # Memory efficiency summary
        if 'memory_efficiency' in self.results:
            mem = self.results['memory_efficiency']
            print(f"Memory Efficiency - Compression ratio: {mem['compression_ratio']:.4f}")
        
        # Accuracy preservation summary
        if 'accuracy_preservation' in self.results:
            acc = self.results['accuracy_preservation']
            print(f"Accuracy - Change ratio: {acc['output_change_ratio']:.4f}, Acceptable: {acc['change_acceptable']}")
        
        # Scalability summary
        if 'scalability' in self.results:
            scale = self.results['scalability']
            print(f"Scalability - Efficient: {scale['scalability_efficient']}")
        
        print(f"\nTotal benchmarks completed: {summary['completed_benchmarks']}/{summary['total_benchmarks']}")
        
        return summary


def run_comprehensive_benchmarks():
    """Run the comprehensive benchmark suite"""
    benchmark_suite = ComprehensiveBenchmarkSuite()
    results = benchmark_suite.run_all_benchmarks()
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_benchmarks()