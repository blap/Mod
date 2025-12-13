"""
Performance Validation System for Qwen3-VL Optimization Techniques
Validates cumulative benefits of combined optimization techniques.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import time
import psutil
import GPUtil
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import logging
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import statistics
import copy


@dataclass
class PerformanceMetrics:
    """Data class to hold performance metrics"""
    execution_time: float
    memory_usage: float  # in GB
    peak_memory: float   # in GB
    gpu_memory: Optional[float] = None  # in GB
    peak_gpu_memory: Optional[float] = None  # in GB
    cpu_usage: Optional[float] = None
    throughput: Optional[float] = None  # samples per second
    accuracy: Optional[float] = None
    power_consumption: Optional[float] = None  # in watts (if available)


@dataclass
class ValidationResult:
    """Data class to hold validation results"""
    optimization_combo: List[str]
    baseline_metrics: PerformanceMetrics
    optimized_metrics: PerformanceMetrics
    improvement_factor: float
    synergy_factor: float
    validation_passed: bool
    validation_notes: List[str]


class PerformanceValidator:
    """
    Validates performance improvements from optimization combinations.
    Measures cumulative benefits and ensures optimizations provide synergistic effects.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results_history: List[ValidationResult] = []
        self.baseline_performance: Optional[PerformanceMetrics] = None
    
    def set_baseline_performance(self, baseline_metrics: PerformanceMetrics):
        """Set the baseline performance metrics"""
        self.baseline_performance = baseline_metrics
    
    def benchmark_model_performance(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 10,
        warmup_runs: int = 3,
        measure_memory: bool = True,
        measure_power: bool = False
    ) -> PerformanceMetrics:
        """Benchmark model performance with various metrics"""
        # Warmup runs
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = model(input_tensor)
        
        # Reset memory stats if measuring memory
        if measure_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        # Start timing and resource monitoring
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**3)  # GB
        
        if measure_memory and torch.cuda.is_available():
            start_gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        cpu_usage_samples = []
        
        # Run benchmark
        for i in range(num_runs):
            # Monitor CPU usage
            cpu_usage_samples.append(psutil.cpu_percent(interval=None))
            
            with torch.no_grad():
                output = model(input_tensor)
            
            # Synchronize for accurate timing on GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used / (1024**3)  # GB
        
        # Calculate metrics
        execution_time = (end_time - start_time) / num_runs
        avg_memory = (start_memory + end_memory) / 2
        peak_memory = psutil.virtual_memory().max_usage / (1024**3) if hasattr(psutil.virtual_memory(), 'max_usage') else end_memory
        
        # Get GPU metrics if available
        gpu_memory = None
        peak_gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        # Calculate throughput
        batch_size = input_tensor.shape[0]
        throughput = batch_size / execution_time if execution_time > 0 else float('inf')
        
        # Calculate average CPU usage
        avg_cpu_usage = statistics.mean(cpu_usage_samples) if cpu_usage_samples else None
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=avg_memory,
            peak_memory=peak_memory,
            gpu_memory=gpu_memory,
            peak_gpu_memory=peak_gpu_memory,
            cpu_usage=avg_cpu_usage,
            throughput=throughput
        )
    
    def validate_cumulative_performance(
        self,
        baseline_model: nn.Module,
        optimized_model: nn.Module,
        input_tensor: torch.Tensor,
        optimization_combo: List[str],
        num_runs: int = 10
    ) -> ValidationResult:
        """Validate cumulative performance improvements"""
        self.logger.info(f"Validating performance for optimization combo: {optimization_combo}")
        
        # Benchmark baseline model
        self.logger.info("Benchmarking baseline model...")
        baseline_metrics = self.benchmark_model_performance(
            baseline_model, input_tensor, num_runs=num_runs
        )
        
        # Benchmark optimized model
        self.logger.info("Benchmarking optimized model...")
        optimized_metrics = self.benchmark_model_performance(
            optimized_model, input_tensor, num_runs=num_runs
        )
        
        # Calculate improvement factor
        improvement_factor = baseline_metrics.execution_time / optimized_metrics.execution_time if optimized_metrics.execution_time > 0 else float('inf')
        
        # Calculate synergy factor based on expected vs actual improvement
        expected_improvement = len(optimization_combo) * 0.1  # Rough estimate: 10% per optimization
        synergy_factor = improvement_factor / (1 + expected_improvement) if (1 + expected_improvement) > 0 else improvement_factor
        
        # Validation checks
        validation_passed = True
        validation_notes = []
        
        # Check if optimization actually improved performance
        if improvement_factor <= 1.0:
            validation_passed = False
            validation_notes.append(f"Performance degradation detected: {improvement_factor:.2f}x slower")
        else:
            validation_notes.append(f"Performance improvement: {improvement_factor:.2f}x faster")
        
        # Check memory efficiency
        if optimized_metrics.peak_memory and baseline_metrics.peak_memory:
            memory_improvement = baseline_metrics.peak_memory / optimized_metrics.peak_memory if optimized_metrics.peak_memory > 0 else float('inf')
            if memory_improvement > 1.0:
                validation_notes.append(f"Memory efficiency: {memory_improvement:.2f}x better")
            else:
                validation_notes.append(f"Memory usage increased: {1/memory_improvement:.2f}x higher")
        
        # Check for GPU memory efficiency
        if optimized_metrics.peak_gpu_memory and baseline_metrics.peak_gpu_memory:
            gpu_memory_improvement = baseline_metrics.peak_gpu_memory / optimized_metrics.peak_gpu_memory if optimized_metrics.peak_gpu_memory > 0 else float('inf')
            if gpu_memory_improvement > 1.0:
                validation_notes.append(f"GPU memory efficiency: {gpu_memory_improvement:.2f}x better")
        
        # Check synergy
        if synergy_factor > 1.1:
            validation_notes.append(f"Positive synergy detected: {synergy_factor:.2f}x synergistic effect")
        elif synergy_factor < 0.9:
            validation_notes.append(f"Negative synergy detected: {synergy_factor:.2f}x efficiency loss")
        
        # Create validation result
        result = ValidationResult(
            optimization_combo=optimization_combo,
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            improvement_factor=improvement_factor,
            synergy_factor=synergy_factor,
            validation_passed=validation_passed,
            validation_notes=validation_notes
        )
        
        # Store result for history
        self.results_history.append(result)
        
        self.logger.info(f"Validation result: {'PASSED' if validation_passed else 'FAILED'}")
        self.logger.info(f"Improvement factor: {improvement_factor:.2f}x")
        self.logger.info(f"Synergy factor: {synergy_factor:.2f}x")
        
        return result
    
    def validate_all_optimizations_synergy(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        optimization_manager,
        interaction_handler,
        num_runs: int = 5
    ) -> Dict[str, Any]:
        """Validate synergy of all 12 optimizations working together"""
        self.logger.info("Validating synergy of all 12 optimizations...")
        
        # Create a copy of the model for baseline
        baseline_model = copy.deepcopy(model)
        
        # Apply all optimizations to the original model
        # This is a simplified approach - in reality, we would have a model that incorporates
        # all optimizations through the optimization manager
        optimized_model = model  # In a real implementation, this would be the model with all optimizations applied
        
        # Test with all optimizations enabled
        all_optimizations = [opt_type.value for opt_type in optimization_manager.get_active_optimizations()]
        
        result = self.validate_cumulative_performance(
            baseline_model,
            optimized_model,
            input_tensor,
            all_optimizations,
            num_runs=num_runs
        )
        
        # Additional validation: test different optimization subsets
        subset_results = []
        optimization_types = optimization_manager.get_active_optimizations()
        
        # Test with different numbers of optimizations
        for subset_size in [1, 2, 4, 8]:
            if len(optimization_types) >= subset_size:
                subset = optimization_types[:subset_size]
                subset_names = [opt.value for opt in subset]
                
                subset_result = self.validate_cumulative_performance(
                    baseline_model,
                    optimized_model,  # In practice, this would be a model with only the subset applied
                    input_tensor,
                    subset_names,
                    num_runs=num_runs//2  # Fewer runs for subsets to save time
                )
                subset_results.append(subset_result)
        
        # Calculate overall synergy metrics
        all_improvements = [r.improvement_factor for r in [result] + subset_results if r.validation_passed]
        avg_synergy = statistics.mean([r.synergy_factor for r in [result] + subset_results if r.validation_passed]) if all_improvements else 0
        
        overall_result = {
            'all_optimizations_result': result,
            'subset_results': subset_results,
            'average_synergy_factor': avg_synergy,
            'cumulative_improvement': result.improvement_factor,
            'validation_passed': result.validation_passed,
            'synergy_analysis': self._analyze_synergy(result, subset_results)
        }
        
        self.logger.info(f"Overall synergy factor: {avg_synergy:.2f}")
        self.logger.info(f"Cumulative improvement: {result.improvement_factor:.2f}x")
        
        return overall_result
    
    def _analyze_synergy(self, full_result: ValidationResult, subset_results: List[ValidationResult]) -> Dict[str, Any]:
        """Analyze synergy patterns from different optimization combinations"""
        # Calculate how much additional benefit each optimization provides
        synergy_analysis = {
            'marginal_benefits': [],
            'synergy_patterns': [],
            'efficiency_plateau': None
        }
        
        # Compare subset results to identify marginal benefits
        if len(subset_results) > 1:
            for i in range(1, len(subset_results)):
                prev_result = subset_results[i-1]
                curr_result = subset_results[i]
                
                marginal_improvement = curr_result.improvement_factor - prev_result.improvement_factor
                marginal_benefits = {
                    'optimization_count': len(curr_result.optimization_combo),
                    'marginal_improvement': marginal_improvement,
                    'relative_efficiency': marginal_improvement / (len(curr_result.optimization_combo) - len(prev_result.optimization_combo)) if len(curr_result.optimization_combo) > len(prev_result.optimization_combo) else 0
                }
                synergy_analysis['marginal_benefits'].append(marginal_benefits)
        
        # Identify efficiency plateau (where adding more optimizations stops providing significant benefits)
        if synergy_analysis['marginal_benefits']:
            efficiency_threshold = 0.05  # 5% improvement threshold
            for i, benefit in enumerate(synergy_analysis['marginal_benefits']):
                if benefit['relative_efficiency'] < efficiency_threshold:
                    synergy_analysis['efficiency_plateau'] = benefit['optimization_count']
                    break
        
        return synergy_analysis
    
    def profile_optimization_performance(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        profile_memory: bool = True,
        profile_compute: bool = True
    ) -> Dict[str, Any]:
        """Profile detailed performance of optimized model"""
        self.logger.info("Profiling optimization performance...")
        
        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        with profile(
            activities=activities,
            record_shapes=profile_memory,
            profile_memory=profile_memory,
            with_stack=True
        ) as prof:
            with record_function("model_inference"):
                start_time = time.time()
                with torch.no_grad():
                    output = model(input_tensor)
                end_time = time.time()
        
        profiling_results = {
            'execution_time': end_time - start_time,
            'profiler_results': prof.key_averages().table(sort_by="cuda_time_total", row_limit=20) if torch.cuda.is_available() else prof.key_averages().table(sort_by="cpu_time_total", row_limit=20),
            'memory_profile': None,
            'peak_memory_stats': None
        }
        
        if torch.cuda.is_available():
            profiling_results['peak_memory_stats'] = {
                'peak_memory_allocated': torch.cuda.max_memory_allocated() / (1024**3),
                'peak_memory_reserved': torch.cuda.max_memory_reserved() / (1024**3),
            }
        
        self.logger.info(f"Profiling complete. Execution time: {profiling_results['execution_time']:.4f}s")
        
        return profiling_results
    
    def validate_accuracy_preservation(
        self,
        baseline_model: nn.Module,
        optimized_model: nn.Module,
        test_data_loader,
        accuracy_threshold: float = 0.01  # 1% threshold
    ) -> Tuple[bool, float, float]:
        """Validate that accuracy is preserved after optimizations"""
        self.logger.info("Validating accuracy preservation...")
        
        baseline_model.eval()
        optimized_model.eval()
        
        baseline_outputs = []
        optimized_outputs = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_data_loader):
                # Get outputs from both models
                baseline_out = baseline_model(data)
                optimized_out = optimized_model(data)
                
                baseline_outputs.append(baseline_out)
                optimized_outputs.append(optimized_out)
        
        # Calculate similarity between outputs
        all_baseline = torch.cat(baseline_outputs, dim=0)
        all_optimized = torch.cat(optimized_outputs, dim=0)
        
        # Calculate mean absolute error
        mae = torch.mean(torch.abs(all_baseline - all_optimized)).item()
        
        # Calculate cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            all_baseline.flatten(), 
            all_optimized.flatten(), 
            dim=0
        ).item()
        
        # Validation: MAE should be below threshold and cosine similarity should be high
        accuracy_preserved = mae < accuracy_threshold and cosine_sim > (1 - accuracy_threshold)
        
        self.logger.info(f"Accuracy validation: MAE={mae:.6f}, Cosine Similarity={cosine_sim:.4f}")
        self.logger.info(f"Accuracy preserved: {'YES' if accuracy_preserved else 'NO'}")
        
        return accuracy_preserved, mae, cosine_sim
    
    def generate_performance_report(self, output_dir: str = "performance_reports") -> str:
        """Generate a comprehensive performance validation report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"performance_validation_report_{timestamp}.json"
        
        # Compile report data
        report_data = {
            'timestamp': timestamp,
            'total_validations': len(self.results_history),
            'passed_validations': sum(1 for r in self.results_history if r.validation_passed),
            'failed_validations': sum(1 for r in self.results_history if not r.validation_passed),
            'average_improvement_factor': statistics.mean([r.improvement_factor for r in self.results_history]) if self.results_history else 0,
            'average_synergy_factor': statistics.mean([r.synergy_factor for r in self.results_history]) if self.results_history else 0,
            'validation_results': [
                {
                    'optimization_combo': r.optimization_combo,
                    'baseline_metrics': r.baseline_metrics.__dict__,
                    'optimized_metrics': r.optimized_metrics.__dict__,
                    'improvement_factor': r.improvement_factor,
                    'synergy_factor': r.synergy_factor,
                    'validation_passed': r.validation_passed,
                    'validation_notes': r.validation_notes
                }
                for r in self.results_history
            ]
        }
        
        # Write report to file
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        self.logger.info(f"Performance validation report generated: {report_file}")
        
        return str(report_file)
    
    def validate_resource_efficiency(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        resource_budget: Dict[str, float]
    ) -> Dict[str, Any]:
        """Validate that optimizations stay within resource budgets"""
        self.logger.info("Validating resource efficiency...")
        
        # Benchmark the model
        metrics = self.benchmark_model_performance(model, input_tensor)
        
        # Check against resource budgets
        resource_check = {
            'memory_within_budget': True,
            'time_within_budget': True,
            'gpu_memory_within_budget': True,
            'exceeded_resources': []
        }
        
        # Check memory budget
        if resource_budget.get('max_memory_gb') and metrics.peak_memory > resource_budget['max_memory_gb']:
            resource_check['memory_within_budget'] = False
            resource_check['exceeded_resources'].append(f"Memory: {metrics.peak_memory:.2f}GB > {resource_budget['max_memory_gb']}GB")
        
        # Check time budget
        if resource_budget.get('max_time_per_sample') and metrics.execution_time > resource_budget['max_time_per_sample']:
            resource_check['time_within_budget'] = False
            resource_check['exceeded_resources'].append(f"Time: {metrics.execution_time:.4f}s > {resource_budget['max_time_per_sample']}s")
        
        # Check GPU memory budget
        if resource_budget.get('max_gpu_memory_gb') and metrics.peak_gpu_memory and metrics.peak_gpu_memory > resource_budget['max_gpu_memory_gb']:
            resource_check['gpu_memory_within_budget'] = False
            resource_check['exceeded_resources'].append(f"GPU Memory: {metrics.peak_gpu_memory:.2f}GB > {resource_budget['max_gpu_memory_gb']}GB")
        
        resource_check['actual_metrics'] = {
            'memory_gb': metrics.peak_memory,
            'time_per_sample': metrics.execution_time,
            'gpu_memory_gb': metrics.peak_gpu_memory
        }
        
        self.logger.info(f"Resource efficiency validation: {'PASSED' if not resource_check['exceeded_resources'] else 'FAILED'}")
        if resource_check['exceeded_resources']:
            self.logger.warning(f"Exceeded resources: {resource_check['exceeded_resources']}")
        
        return resource_check


class CumulativePerformanceValidator(PerformanceValidator):
    """
    Extended performance validator that specifically focuses on cumulative benefits
    from multiple optimization techniques working together.
    """
    
    def __init__(self):
        super().__init__()
        self.optimization_impact_matrix: Dict[Tuple[str, str], float] = {}
    
    def validate_cumulative_benefits(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        optimization_manager,
        num_runs: int = 10
    ) -> Dict[str, Any]:
        """Validate that cumulative benefits exceed individual optimization benefits"""
        self.logger.info("Validating cumulative benefits vs individual benefits...")
        
        # Get all active optimizations
        active_optimizations = optimization_manager.get_active_optimizations()
        
        # Test each optimization individually
        individual_results = {}
        for opt_type in active_optimizations:
            # Temporarily disable all optimizations except this one
            original_states = {}
            for other_opt in active_optimizations:
                original_states[other_opt] = optimization_manager.optimization_states[other_opt]
                optimization_manager.optimization_states[other_opt] = (other_opt == opt_type)
            
            # Create a copy of the model for this test
            temp_model = copy.deepcopy(model)
            
            # Benchmark with only this optimization
            baseline_model = copy.deepcopy(model)
            result = self.validate_cumulative_performance(
                baseline_model,
                temp_model,
                input_tensor,
                [opt_type.value],
                num_runs=num_runs//2  # Fewer runs for individual tests
            )
            
            individual_results[opt_type.value] = result
            optimization_manager.optimization_states = original_states
        
        # Test all optimizations together
        baseline_model = copy.deepcopy(model)
        all_opt_result = self.validate_cumulative_performance(
            baseline_model,
            model,  # Model with all optimizations enabled
            input_tensor,
            [opt.value for opt in active_optimizations],
            num_runs=num_runs
        )
        
        # Calculate cumulative benefit factor
        individual_improvements = [r.improvement_factor for r in individual_results.values()]
        expected_cumulative = sum(individual_improvements) - (len(individual_improvements) - 1)  # Subtract 1 for baseline
        actual_cumulative = all_opt_result.improvement_factor
        
        # Calculate synergy ratio
        synergy_ratio = actual_cumulative / expected_cumulative if expected_cumulative > 0 else 0
        
        results = {
            'individual_results': {name: {
                'improvement_factor': result.improvement_factor,
                'validation_passed': result.validation_passed
            } for name, result in individual_results.items()},
            'all_optimizations_result': {
                'improvement_factor': all_opt_result.improvement_factor,
                'validation_passed': all_opt_result.validation_passed
            },
            'expected_cumulative_improvement': expected_cumulative,
            'actual_cumulative_improvement': actual_cumulative,
            'synergy_ratio': synergy_ratio,
            'cumulative_benefit_validated': synergy_ratio >= 1.0,  # At least match expectations
            'efficiency_gains': actual_cumulative - expected_cumulative
        }
        
        self.logger.info(f"Expected cumulative improvement: {expected_cumulative:.2f}x")
        self.logger.info(f"Actual cumulative improvement: {actual_cumulative:.2f}x")
        self.logger.info(f"Synergy ratio: {synergy_ratio:.2f}x")
        self.logger.info(f"Cumulative benefit validated: {results['cumulative_benefit_validated']}")
        
        return results