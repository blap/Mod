"""Performance Monitoring System for Qwen3-VL Optimization Techniques
Tracks and evaluates the impact of each optimization combination to ensure
synergistic effects and proper performance improvements.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
import time
import math
import psutil
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, deque
import json


@dataclass
class PerformanceMetric:
    """Single performance metric for an optimization."""
    name: str
    execution_time: float
    memory_usage: float
    accuracy_change: float
    throughput: float
    latency: float
    timestamp: float
    optimization_combination: List[str]


class PerformanceMonitor:
    """Main performance monitoring system that tracks optimization impacts."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Store performance metrics
        self.metrics_history: List[PerformanceMetric] = []
        self.current_metrics: Dict[str, PerformanceMetric] = {}
        
        # Track optimization-specific metrics
        self.optimization_metrics: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        
        # Track combination metrics
        self.combination_metrics: Dict[Tuple[str, ...], List[PerformanceMetric]] = defaultdict(list)
        
        # Performance baselines
        self.baseline_execution_time = None
        self.baseline_memory_usage = None
        self.baseline_accuracy = None
        
        # Resource monitoring
        self.gpu_memory_monitoring = torch.cuda.is_available()
        self.system_memory_monitoring = True
        
        # Initialize monitoring intervals
        self.monitoring_interval = getattr(config, 'monitoring_interval', 10)  # Monitor every 10 steps
        self.step_counter = 0
        
        self.logger.info("Performance Monitor initialized")
    
    def start_monitoring(self, baseline_model: nn.Module, baseline_input: torch.Tensor):
        """Start monitoring by establishing baseline performance."""
        # Establish baseline performance metrics
        start_time = time.time()
        with torch.no_grad():
            baseline_output = baseline_model(baseline_input)
        baseline_time = time.time() - start_time
        
        baseline_memory = self._get_current_memory_usage()
        
        self.baseline_execution_time = baseline_time
        self.baseline_memory_usage = baseline_memory
        self.baseline_accuracy = 1.0  # Placeholder for accuracy
        
        self.logger.info(f"Baseline established: Execution time {baseline_time:.4f}s, Memory usage {baseline_memory:.2f}MB")
    
    def record_metric(self, 
                      optimization_name: str,
                      execution_time: float,
                      memory_usage: float,
                      accuracy_change: float,
                      throughput: float,
                      latency: float,
                      optimization_combination: List[str]):
        """Record performance metrics for an optimization."""
        metric = PerformanceMetric(
            name=optimization_name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            accuracy_change=accuracy_change,
            throughput=throughput,
            latency=latency,
            timestamp=time.time(),
            optimization_combination=optimization_combination
        )
        
        self.metrics_history.append(metric)
        self.current_metrics[optimization_name] = metric
        self.optimization_metrics[optimization_name].append(metric)
        
        # Sort combination for consistent key
        combo_key = tuple(sorted(optimization_combination))
        self.combination_metrics[combo_key].append(metric)
        
        # Log metric if verbose
        if self.config.get('verbose_monitoring', False):
            self.logger.info(
                f"Performance metric recorded: {optimization_name} - "
                f"Time: {execution_time:.4f}s, Memory: {memory_usage:.2f}MB, "
                f"Accuracy change: {accuracy_change:.4f}, Throughput: {throughput:.2f} samples/s"
            )
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.gpu_memory_monitoring:
            return torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        else:
            # Use system memory if GPU not available
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)  # Convert to MB
    
    def get_latest_metrics(self, optimization_name: str) -> Optional[PerformanceMetric]:
        """Get the latest metrics for a specific optimization."""
        if optimization_name in self.current_metrics:
            return self.current_metrics[optimization_name]
        return None
    
    def get_average_metrics(self, optimization_name: str) -> Optional[Dict[str, float]]:
        """Get average metrics for a specific optimization."""
        if optimization_name in self.optimization_metrics:
            metrics = self.optimization_metrics[optimization_name]
            if metrics:
                avg_exec_time = np.mean([m.execution_time for m in metrics])
                avg_memory = np.mean([m.memory_usage for m in metrics])
                avg_accuracy = np.mean([m.accuracy_change for m in metrics])
                avg_throughput = np.mean([m.throughput for m in metrics])
                avg_latency = np.mean([m.latency for m in metrics])
                
                return {
                    'avg_execution_time': avg_exec_time,
                    'avg_memory_usage': avg_memory,
                    'avg_accuracy_change': avg_accuracy,
                    'avg_throughput': avg_throughput,
                    'avg_latency': avg_latency,
                    'count': len(metrics)
                }
        return None
    
    def get_combination_metrics(self, optimization_combination: List[str]) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific combination of optimizations."""
        combo_key = tuple(sorted(optimization_combination))
        if combo_key in self.combination_metrics:
            metrics = self.combination_metrics[combo_key]
            if metrics:
                avg_exec_time = np.mean([m.execution_time for m in metrics])
                avg_memory = np.mean([m.memory_usage for m in metrics])
                avg_accuracy = np.mean([m.accuracy_change for m in metrics])
                avg_throughput = np.mean([m.throughput for m in metrics])
                avg_latency = np.mean([m.latency for m in metrics])
                
                # Calculate improvement ratios compared to baseline
                time_improvement = (self.baseline_execution_time - avg_exec_time) / self.baseline_execution_time * 100 if self.baseline_execution_time else 0
                memory_improvement = (self.baseline_memory_usage - avg_memory) / self.baseline_memory_usage * 100 if self.baseline_memory_usage else 0
                
                return {
                    'optimization_combination': optimization_combination,
                    'avg_execution_time': avg_exec_time,
                    'avg_memory_usage': avg_memory,
                    'avg_accuracy_change': avg_accuracy,
                    'avg_throughput': avg_throughput,
                    'avg_latency': avg_latency,
                    'time_improvement_percent': time_improvement,
                    'memory_improvement_percent': memory_improvement,
                    'count': len(metrics)
                }
        return None
    
    def get_synergy_analysis(self) -> Dict[str, Any]:
        """Analyze synergistic effects between optimizations."""
        if len(self.metrics_history) < 2:
            return {'synergy_analysis': 'insufficient_data'}
        
        # Analyze each pair of optimizations for synergistic effects
        synergy_results = {}
        
        # Get unique optimization names
        unique_optimizations = set([m.name for m in self.metrics_history])
        
        # Calculate baseline performance for each optimization individually
        individual_performance = {}
        for opt_name in unique_optimizations:
            opt_metrics = [m for m in self.metrics_history if m.name == opt_name and len(m.optimization_combination) == 1]
            if opt_metrics:
                avg_time = np.mean([m.execution_time for m in opt_metrics])
                avg_memory = np.mean([m.memory_usage for m in opt_metrics])
                individual_performance[opt_name] = {'time': avg_time, 'memory': avg_memory}
        
        # Analyze performance when optimizations are used together
        for combo, metrics in self.combination_metrics.items():
            if len(combo) > 1:  # Only analyze combinations
                combo_avg_time = np.mean([m.execution_time for m in metrics])
                combo_avg_memory = np.mean([m.memory_usage for m in metrics])
                
                # Calculate expected performance if optimizations were independent
                expected_time = 0
                expected_memory = 0
                for opt_name in combo:
                    if opt_name in individual_performance:
                        expected_time += individual_performance[opt_name]['time']
                        expected_memory += individual_performance[opt_name]['memory']
                
                # Normalize by number of optimizations in the combo
                expected_time /= len(combo)
                expected_memory /= len(combo)
                
                # Calculate synergy score
                time_synergy = (expected_time - combo_avg_time) / expected_time if expected_time > 0 else 0
                memory_synergy = (expected_memory - combo_avg_memory) / expected_memory if expected_memory > 0 else 0
                
                synergy_results[str(combo)] = {
                    'time_synergy': time_synergy,
                    'memory_synergy': memory_synergy,
                    'combo_avg_time': combo_avg_time,
                    'combo_avg_memory': combo_avg_memory,
                    'expected_time': expected_time,
                    'expected_memory': expected_memory
                }
        
        return {
            'synergy_results': synergy_results,
            'unique_optimizations': list(unique_optimizations),
            'individual_performance': individual_performance,
            'total_combinations_analyzed': len(synergy_results)
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary."""
        summary = {
            'total_metrics_recorded': len(self.metrics_history),
            'optimization_names': list(self.optimization_metrics.keys()),
            'combination_count': len(self.combination_metrics),
            'baseline_execution_time': self.baseline_execution_time,
            'baseline_memory_usage': self.baseline_memory_usage,
            'baseline_accuracy': self.baseline_accuracy
        }
        
        # Add average metrics for each optimization
        for opt_name in self.optimization_metrics.keys():
            avg_metrics = self.get_average_metrics(opt_name)
            if avg_metrics:
                summary[f'avg_metrics_{opt_name}'] = avg_metrics
        
        # Add synergy analysis
        summary['synergy_analysis'] = self.get_synergy_analysis()
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Export performance metrics to a file."""
        summary = self.get_performance_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)  # default=str to handle non-serializable objects like tensors
        
        self.logger.info(f"Performance metrics exported to {filepath}")


class OptimizationImpactTracker:
    """Tracks the impact of each optimization on model performance."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Impact tracking
        self.optimization_impacts = defaultdict(lambda: {'time': [], 'memory': [], 'accuracy': []})
        self.impact_history = deque(maxlen=getattr(config, 'impact_history_size', 100))
        
        # Initialize impact thresholds
        self.time_improvement_threshold = getattr(config, 'time_improvement_threshold', 0.05)  # 5% improvement needed
        self.memory_improvement_threshold = getattr(config, 'memory_improvement_threshold', 0.05)  # 5% improvement needed
        self.accuracy_threshold = getattr(config, 'accuracy_threshold', -0.01)  # Allow up to 1% accuracy drop
        
        self.logger.info("Optimization Impact Tracker initialized")
    
    def record_impact(self, optimization_name: str, 
                     time_impact: float, 
                     memory_impact: float, 
                     accuracy_impact: float):
        """Record the impact of an optimization."""
        self.optimization_impacts[optimization_name]['time'].append(time_impact)
        self.optimization_impacts[optimization_name]['memory'].append(memory_impact)
        self.optimization_impacts[optimization_name]['accuracy'].append(accuracy_impact)
        
        # Add to impact history
        self.impact_history.append({
            'optimization': optimization_name,
            'time_impact': time_impact,
            'memory_impact': memory_impact,
            'accuracy_impact': accuracy_impact,
            'timestamp': time.time()
        })
    
    def get_impact_summary(self, optimization_name: str) -> Dict[str, Any]:
        """Get impact summary for a specific optimization."""
        if optimization_name in self.optimization_impacts:
            impacts = self.optimization_impacts[optimization_name]
            return {
                'optimization': optimization_name,
                'avg_time_impact': np.mean(impacts['time']),
                'avg_memory_impact': np.mean(impacts['memory']),
                'avg_accuracy_impact': np.mean(impacts['accuracy']),
                'time_impact_std': np.std(impacts['time']),
                'memory_impact_std': np.std(impacts['memory']),
                'accuracy_impact_std': np.std(impacts['accuracy']),
                'sample_count': len(impacts['time']),
                'positive_time_impacts': sum(1 for x in impacts['time'] if x < 0),  # Negative impact is improvement
                'positive_memory_impacts': sum(1 for x in impacts['memory'] if x < 0),  # Negative impact is improvement
                'positive_accuracy_impacts': sum(1 for x in impacts['accuracy'] if x >= 0)  # Positive impact is good
            }
        return {}
    
    def is_optimization_beneficial(self, optimization_name: str) -> bool:
        """Check if an optimization is beneficial based on impact thresholds."""
        if optimization_name not in self.optimization_impacts:
            return False
        
        impacts = self.optimization_impacts[optimization_name]
        avg_time_impact = np.mean(impacts['time'])
        avg_memory_impact = np.mean(impacts['memory'])
        avg_accuracy_impact = np.mean(impacts['accuracy'])
        
        # Check if time and memory improvements meet thresholds and accuracy doesn't drop too much
        time_benefit = avg_time_impact < -self.time_improvement_threshold  # Negative is improvement
        memory_benefit = avg_memory_impact < -self.memory_improvement_threshold  # Negative is improvement
        accuracy_acceptable = avg_accuracy_impact > self.accuracy_threshold  # Positive or small negative is acceptable
        
        return time_benefit or memory_benefit and accuracy_acceptable
    
    def get_all_beneficial_optimizations(self) -> List[str]:
        """Get list of all optimizations that are currently beneficial."""
        beneficial = []
        for opt_name in self.optimization_impacts.keys():
            if self.is_optimization_beneficial(opt_name):
                beneficial.append(opt_name)
        return beneficial
    
    def get_optimization_rankings(self) -> List[Tuple[str, float]]:
        """Rank optimizations by their overall benefit."""
        rankings = []
        for opt_name in self.optimization_impacts.keys():
            impacts = self.optimization_impacts[opt_name]
            avg_time_impact = np.mean(impacts['time'])
            avg_memory_impact = np.mean(impacts['memory'])
            avg_accuracy_impact = np.mean(impacts['accuracy'])
            
            # Calculate composite benefit score (time and memory improvements, accuracy maintenance)
            benefit_score = -(avg_time_impact + avg_memory_impact) + avg_accuracy_impact * 0.5  # Weight accuracy less
            rankings.append((opt_name, benefit_score))
        
        # Sort by benefit score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


class ResourceUtilizationMonitor:
    """Monitors resource utilization across optimizations."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Resource tracking
        self.gpu_memory_usage_history = deque(maxlen=100)
        self.cpu_memory_usage_history = deque(maxlen=100)
        self.gpu_compute_utilization = deque(maxlen=100)
        self.cpu_compute_utilization = deque(maxlen=100)
        self.io_bandwidth_usage = deque(maxlen=100)
        
        # Resource allocation tracking
        self.resource_allocation_patterns = {}
        
        self.logger.info("Resource Utilization Monitor initialized")
    
    def record_resource_usage(self, gpu_memory: Optional[float] = None, 
                             cpu_memory: Optional[float] = None,
                             gpu_compute: Optional[float] = None,
                             cpu_compute: Optional[float] = None,
                             io_bandwidth: Optional[float] = None):
        """Record resource utilization."""
        if gpu_memory is not None:
            self.gpu_memory_usage_history.append(gpu_memory)
        if cpu_memory is not None:
            self.cpu_memory_usage_history.append(cpu_memory)
        if gpu_compute is not None:
            self.gpu_compute_utilization.append(gpu_compute)
        if cpu_compute is not None:
            self.cpu_compute_utilization.append(cpu_compute)
        if io_bandwidth is not None:
            self.io_bandwidth_usage.append(io_bandwidth)
    
    def get_current_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        utilization = {}
        
        if self.gpu_memory_usage_history:
            utilization['gpu_memory_current'] = self.gpu_memory_usage_history[-1]
            utilization['gpu_memory_avg'] = np.mean(self.gpu_memory_usage_history)
            utilization['gpu_memory_peak'] = max(self.gpu_memory_usage_history)
        
        if self.cpu_memory_usage_history:
            utilization['cpu_memory_current'] = self.cpu_memory_usage_history[-1]
            utilization['cpu_memory_avg'] = np.mean(self.cpu_memory_usage_history)
            utilization['cpu_memory_peak'] = max(self.cpu_memory_usage_history)
        
        if self.gpu_compute_utilization:
            utilization['gpu_compute_avg'] = np.mean(self.gpu_compute_utilization)
            utilization['gpu_compute_peak'] = max(self.gpu_compute_utilization)
        
        if self.cpu_compute_utilization:
            utilization['cpu_compute_avg'] = np.mean(self.cpu_compute_utilization)
            utilization['cpu_compute_peak'] = max(self.cpu_compute_utilization)
        
        if self.io_bandwidth_usage:
            utilization['io_bandwidth_avg'] = np.mean(self.io_bandwidth_usage)
            utilization['io_bandwidth_peak'] = max(self.io_bandwidth_usage)
        
        return utilization
    
    def analyze_resource_allocation_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in resource allocation."""
        # For now, return a placeholder - in a real implementation this would analyze patterns
        return {
            'allocation_patterns': 'analyzed',
            'peak_memory_usage': max(self.gpu_memory_usage_history) if self.gpu_memory_usage_history else 0,
            'average_memory_usage': np.mean(self.gpu_memory_usage_history) if self.gpu_memory_usage_history else 0,
            'memory_efficiency_score': 0.85  # Placeholder
        }


class PerformanceValidator:
    """Validates performance improvements from optimization combinations."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize monitors
        self.performance_monitor = PerformanceMonitor(config)
        self.impact_tracker = OptimizationImpactTracker(config)
        self.resource_monitor = ResourceUtilizationMonitor(config)
        
        self.logger.info("Performance Validator initialized")
    
    def validate_single_optimization(self, 
                                   model: nn.Module, 
                                   input_tensor: torch.Tensor,
                                   optimization_name: str,
                                   optimization_func: Callable) -> Dict[str, Any]:
        """Validate performance impact of a single optimization."""
        # Record baseline
        baseline_start_time = time.time()
        baseline_memory = self.performance_monitor._get_current_memory_usage()
        
        with torch.no_grad():
            baseline_output = model(input_tensor)
        
        baseline_time = time.time() - baseline_start_time
        
        # Apply optimization
        optimized_start_time = time.time()
        optimized_memory = self.performance_monitor._get_current_memory_usage()
        
        with torch.no_grad():
            optimized_output = optimization_func(model, input_tensor)
        
        optimized_time = time.time() - optimized_start_time
        
        # Calculate accuracy preservation (using cosine similarity as a proxy)
        accuracy_change = torch.cosine_similarity(
            baseline_output.flatten(), 
            optimized_output.flatten(), 
            dim=0
        ).mean().item() - 1.0  # Difference from perfect similarity
        
        # Calculate throughput (samples per second)
        batch_size = input_tensor.shape[0]
        baseline_throughput = batch_size / baseline_time if baseline_time > 0 else float('inf')
        optimized_throughput = batch_size / optimized_time if optimized_time > 0 else float('inf')
        
        # Calculate latency (time per sample)
        baseline_latency = baseline_time / batch_size if batch_size > 0 else 0
        optimized_latency = optimized_time / batch_size if batch_size > 0 else 0
        
        # Record metrics
        self.performance_monitor.record_metric(
            optimization_name=optimization_name,
            execution_time=optimized_time,
            memory_usage=optimized_memory,
            accuracy_change=accuracy_change,
            throughput=optimized_throughput,
            latency=optimized_latency,
            optimization_combination=[optimization_name]
        )
        
        # Record impact
        time_impact = optimized_time - baseline_time
        memory_impact = optimized_memory - baseline_memory
        self.impact_tracker.record_impact(optimization_name, time_impact, memory_impact, accuracy_change)
        
        # Record resource usage
        self.resource_monitor.record_resource_usage(
            gpu_memory=optimized_memory if torch.cuda.is_available() else None,
            cpu_memory=None if torch.cuda.is_available() else optimized_memory
        )
        
        return {
            'optimization_name': optimization_name,
            'baseline_time': baseline_time,
            'optimized_time': optimized_time,
            'time_improvement': (baseline_time - optimized_time) / baseline_time * 100 if baseline_time > 0 else 0,
            'baseline_memory': baseline_memory,
            'optimized_memory': optimized_memory,
            'memory_improvement': (baseline_memory - optimized_memory) / baseline_memory * 100 if baseline_memory > 0 else 0,
            'accuracy_preserved': accuracy_change > -0.01,  # Allow 1% drop
            'accuracy_change': accuracy_change,
            'throughput_improvement': (optimized_throughput - baseline_throughput) / baseline_throughput * 100 if baseline_throughput > 0 else 0,
            'latency_improvement': (baseline_latency - optimized_latency) / baseline_latency * 100 if baseline_latency > 0 else 0
        }
    
    def validate_optimization_combination(self, 
                                        model: nn.Module,
                                        input_tensor: torch.Tensor,
                                        optimization_combo: List[str],
                                        optimization_funcs: Dict[str, Callable]) -> Dict[str, Any]:
        """Validate performance impact of an optimization combination."""
        # Record baseline
        baseline_start_time = time.time()
        baseline_memory = self.performance_monitor._get_current_memory_usage()
        
        with torch.no_grad():
            baseline_output = model(input_tensor)
        
        baseline_time = time.time() - baseline_start_time
        
        # Apply all optimizations in the combination
        optimized_start_time = time.time()
        current_output = input_tensor
        
        for opt_name in optimization_combo:
            if opt_name in optimization_funcs:
                current_output = optimization_funcs[opt_name](model, current_output)
        
        optimized_time = time.time() - optimized_start_time
        optimized_memory = self.performance_monitor._get_current_memory_usage()
        
        # Calculate accuracy preservation
        accuracy_change = torch.cosine_similarity(
            baseline_output.flatten(), 
            current_output.flatten(), 
            dim=0
        ).mean().item() - 1.0  # Difference from perfect similarity
        
        # Calculate throughput and latency
        batch_size = input_tensor.shape[0]
        optimized_throughput = batch_size / optimized_time if optimized_time > 0 else float('inf')
        optimized_latency = optimized_time / batch_size if batch_size > 0 else 0
        
        # Record metrics
        self.performance_monitor.record_metric(
            optimization_name=f"combo_{len(optimization_combo)}",
            execution_time=optimized_time,
            memory_usage=optimized_memory,
            accuracy_change=accuracy_change,
            throughput=optimized_throughput,
            latency=optimized_latency,
            optimization_combination=optimization_combo
        )
        
        # Record impact for the combination
        time_impact = optimized_time - baseline_time
        memory_impact = optimized_memory - baseline_memory
        combo_name = f"combo_{'_'.join(sorted(optimization_combo))}"
        self.impact_tracker.record_impact(combo_name, time_impact, memory_impact, accuracy_change)
        
        # Record resource usage
        self.resource_monitor.record_resource_usage(
            gpu_memory=optimized_memory if torch.cuda.is_available() else None,
            cpu_memory=None if torch.cuda.is_available() else optimized_memory
        )
        
        return {
            'optimization_combo': optimization_combo,
            'baseline_time': baseline_time,
            'combo_time': optimized_time,
            'time_improvement': (baseline_time - optimized_time) / baseline_time * 100 if baseline_time > 0 else 0,
            'baseline_memory': baseline_memory,
            'combo_memory': optimized_memory,
            'memory_improvement': (baseline_memory - optimized_memory) / baseline_memory * 100 if baseline_memory > 0 else 0,
            'accuracy_preserved': accuracy_change > -0.01,  # Allow 1% drop
            'accuracy_change': accuracy_change,
            'throughput_improvement': (optimized_throughput - batch_size/baseline_time) / (batch_size/baseline_time) * 100 if baseline_time > 0 else 0,
            'synergy_score': self._calculate_synergy_score(optimization_combo),
            'beneficial_optimizations': self.impact_tracker.get_all_beneficial_optimizations()
        }
    
    def _calculate_synergy_score(self, optimization_combo: List[str]) -> float:
        """Calculate synergy score for an optimization combination."""
        # Placeholder implementation - in a real system, this would analyze how optimizations work together
        # For now, return a score based on the number of optimizations and their individual benefits
        individual_benefits = []
        for opt_name in optimization_combo:
            avg_metrics = self.performance_monitor.get_average_metrics(opt_name)
            if avg_metrics:
                # Time and memory improvements contribute positively to synergy
                individual_benefits.append(avg_metrics['avg_execution_time'] + avg_metrics['avg_memory_usage'])
        
        if individual_benefits:
            avg_individual_benefit = np.mean(individual_benefits)
            combo_metrics = self.performance_monitor.get_combination_metrics(optimization_combo)
            if combo_metrics:
                combo_benefit = combo_metrics['avg_execution_time'] + combo_metrics['avg_memory_usage']
                # Calculate synergy as the difference between expected and actual performance
                synergy = (avg_individual_benefit - combo_benefit) / avg_individual_benefit if avg_individual_benefit != 0 else 0
                return synergy
        return 0.0
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation report."""
        return {
            'performance_summary': self.performance_monitor.get_performance_summary(),
            'impact_summary': {
                opt: self.impact_tracker.get_impact_summary(opt) 
                for opt in self.impact_tracker.optimization_impacts.keys()
            },
            'resource_utilization': self.resource_monitor.get_current_resource_utilization(),
            'optimization_rankings': self.impact_tracker.get_optimization_rankings(),
            'beneficial_optimizations': self.impact_tracker.get_all_beneficial_optimizations()
        }


def create_performance_monitor(config) -> PerformanceMonitor:
    """Factory function to create a performance monitor."""
    return PerformanceMonitor(config)


def create_performance_validator(config) -> PerformanceValidator:
    """Factory function to create a performance validator."""
    return PerformanceValidator(config)


# Example usage and testing
if __name__ == "__main__":
    import torch.nn.functional as F
    
    # Mock config for testing
    class MockConfig:
        def __init__(self):
            self.monitoring_interval = 5
            self.impact_history_size = 20
            self.time_improvement_threshold = 0.05
            self.memory_improvement_threshold = 0.05
            self.accuracy_threshold = -0.01
            self.verbose_monitoring = True
    
    config = MockConfig()
    
    # Create performance validator
    validator = create_performance_validator(config)
    
    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self, hidden_size=512, num_heads=8):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_heads = num_heads
            self.head_dim = hidden_size // num_heads
            
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        def forward(self, x):
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            batch_size, seq_len, _ = x.shape
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
            
            return self.o_proj(attn_output)
    
    # Create test model and input
    model = SimpleModel(hidden_size=512, num_heads=8)
    test_input = torch.randn(2, 64, 512)  # [batch_size, seq_len, hidden_size]
    
    # Establish baseline
    validator.performance_monitor.start_monitoring(model, test_input)
    
    # Define mock optimization functions
    def mock_optimization_func1(model, input_tensor):
        """Mock optimization 1: Block sparse attention"""
        return input_tensor  # Placeholder - in real implementation, this would apply optimization
    
    def mock_optimization_func2(model, input_tensor):
        """Mock optimization 2: KV cache optimization"""
        return input_tensor  # Placeholder - in real implementation, this would apply optimization
    
    optimization_funcs = {
        'block_sparse_attention': mock_optimization_func1,
        'kv_cache_optimization': mock_optimization_func2
    }
    
    # Validate single optimization
    print("Validating single optimization...")
    single_result = validator.validate_single_optimization(
        model, test_input, 'block_sparse_attention', mock_optimization_func1
    )
    print(f"Single optimization result: {single_result}")
    
    # Validate optimization combination
    print("\nValidating optimization combination...")
    combo_result = validator.validate_optimization_combination(
        model, test_input, ['block_sparse_attention', 'kv_cache_optimization'], optimization_funcs
    )
    print(f"Combo optimization result: {combo_result}")
    
    # Get validation report
    print("\nGenerating validation report...")
    report = validator.get_validation_report()
    print(f"Report keys: {list(report.keys())}")
    
    print("\nPerformance monitoring system tests completed successfully!")