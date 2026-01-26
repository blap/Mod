"""
Performance Regression Testing Framework

This module provides a framework for detecting performance regressions in model inference.
It compares current performance metrics against historical baselines to identify degradation.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import statistics
from dataclasses import dataclass
import hashlib


@dataclass
class PerformanceBaseline:
    """Represents a performance baseline for a specific model and metric."""
    model_name: str
    metric_name: str
    value: float
    unit: str
    timestamp: str
    metadata: Dict[str, Any]
    std_dev: Optional[float] = None


@dataclass
class PerformanceRegressionResult:
    """Result of a performance regression test."""
    model_name: str
    metric_name: str
    current_value: float
    baseline_value: float
    unit: str
    regression_detected: bool
    regression_percentage: float
    threshold_used: float
    metadata: Dict[str, Any]


class PerformanceRegressionTracker:
    """Tracks and detects performance regressions."""
    
    def __init__(self, baseline_file: str = "performance_baselines.json", 
                 history_file: str = "performance_history.json"):
        """
        Initialize the performance regression tracker.
        
        Args:
            baseline_file: File to store performance baselines
            history_file: File to store historical performance data
        """
        self.baseline_file = Path(baseline_file)
        self.history_file = Path(history_file)
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.history: List[Dict[str, Any]] = []
        
        # Load existing baselines and history
        self.load_baselines()
        self.load_history()
    
    def load_baselines(self) -> None:
        """Load performance baselines from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    data = json.load(f)
                    
                for baseline_data in data:
                    key = f"{baseline_data['model_name']}_{baseline_data['metric_name']}"
                    self.baselines[key] = PerformanceBaseline(
                        model_name=baseline_data['model_name'],
                        metric_name=baseline_data['metric_name'],
                        value=baseline_data['value'],
                        unit=baseline_data['unit'],
                        timestamp=baseline_data['timestamp'],
                        metadata=baseline_data.get('metadata', {}),
                        std_dev=baseline_data.get('std_dev')
                    )
            except Exception as e:
                print(f"Warning: Could not load baselines from {self.baseline_file}: {e}")
    
    def save_baselines(self) -> None:
        """Save performance baselines to file."""
        data = []
        for baseline in self.baselines.values():
            data.append({
                'model_name': baseline.model_name,
                'metric_name': baseline.metric_name,
                'value': baseline.value,
                'unit': baseline.unit,
                'timestamp': baseline.timestamp,
                'metadata': baseline.metadata,
                'std_dev': baseline.std_dev
            })
        
        with open(self.baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_history(self) -> None:
        """Load historical performance data from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load history from {self.history_file}: {e}")
    
    def save_history(self) -> None:
        """Save historical performance data to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def set_baseline(self, model_name: str, metric_name: str, value: float, 
                     unit: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Set a performance baseline for a specific model and metric.
        
        Args:
            model_name: Name of the model
            metric_name: Name of the performance metric
            value: Baseline value
            unit: Unit of measurement
            metadata: Additional metadata about the baseline
        """
        key = f"{model_name}_{metric_name}"
        self.baselines[key] = PerformanceBaseline(
            model_name=model_name,
            metric_name=metric_name,
            value=value,
            unit=unit,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.save_baselines()
    
    def get_baseline(self, model_name: str, metric_name: str) -> Optional[PerformanceBaseline]:
        """
        Get the performance baseline for a specific model and metric.
        
        Args:
            model_name: Name of the model
            metric_name: Name of the performance metric
            
        Returns:
            Performance baseline or None if not found
        """
        key = f"{model_name}_{metric_name}"
        return self.baselines.get(key)
    
    def record_performance(self, model_name: str, metric_name: str, value: float, 
                          unit: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a performance measurement.
        
        Args:
            model_name: Name of the model
            metric_name: Name of the performance metric
            value: Measured value
            unit: Unit of measurement
            metadata: Additional metadata about the measurement
        """
        record = {
            'model_name': model_name,
            'metric_name': metric_name,
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.history.append(record)
        self.save_history()
    
    def detect_regression(self, model_name: str, metric_name: str, current_value: float,
                         threshold_percentage: float = 5.0) -> PerformanceRegressionResult:
        """
        Detect performance regression for a specific model and metric.
        
        Args:
            model_name: Name of the model
            metric_name: Name of the performance metric
            current_value: Current measured value
            threshold_percentage: Threshold percentage for regression detection
            
        Returns:
            Performance regression result
        """
        baseline = self.get_baseline(model_name, metric_name)
        
        if not baseline:
            # No baseline exists, so no regression can be detected
            return PerformanceRegressionResult(
                model_name=model_name,
                metric_name=metric_name,
                current_value=current_value,
                baseline_value=float('nan'),
                unit=baseline.unit if baseline else "",
                regression_detected=False,
                regression_percentage=0.0,
                threshold_used=threshold_percentage,
                metadata={}
            )
        
        # Calculate percentage difference
        # For performance metrics where higher is better (e.g., tokens/sec), 
        # a decrease indicates regression
        if metric_name in ['inference_speed', 'throughput', 'tokens_per_second']:
            # Higher is better, so lower current value is regression
            if baseline.value != 0:
                percentage_change = ((current_value - baseline.value) / abs(baseline.value)) * 100
            else:
                percentage_change = float('inf') if current_value > 0 else 0.0
        else:
            # Lower is better (e.g., latency, memory usage), so higher current value is regression
            if baseline.value != 0:
                percentage_change = ((current_value - baseline.value) / abs(baseline.value)) * 100
            else:
                percentage_change = float('inf') if current_value > 0 else 0.0
        
        # For metrics where higher is better, negative percentage means regression
        # For metrics where lower is better, positive percentage means regression
        if metric_name in ['inference_speed', 'throughput', 'tokens_per_second']:
            regression_detected = percentage_change < -abs(threshold_percentage)
        else:
            regression_detected = percentage_change > abs(threshold_percentage)
        
        # Record the current performance
        self.record_performance(
            model_name=model_name,
            metric_name=metric_name,
            value=current_value,
            unit=baseline.unit,
            metadata={'regression_detected': regression_detected, 'percentage_change': percentage_change}
        )
        
        return PerformanceRegressionResult(
            model_name=model_name,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline.value,
            unit=baseline.unit,
            regression_detected=regression_detected,
            regression_percentage=percentage_change,
            threshold_used=threshold_percentage,
            metadata={'baseline_timestamp': baseline.timestamp}
        )
    
    def get_performance_trend(self, model_name: str, metric_name: str, 
                             window_size: int = 10) -> List[Dict[str, Any]]:
        """
        Get the performance trend for a specific model and metric.
        
        Args:
            model_name: Name of the model
            metric_name: Name of the performance metric
            window_size: Number of recent measurements to include
            
        Returns:
            List of recent performance measurements
        """
        relevant_records = [
            record for record in self.history
            if record['model_name'] == model_name and record['metric_name'] == metric_name
        ]
        
        # Sort by timestamp (most recent first)
        relevant_records.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return relevant_records[:window_size]
    
    def calculate_statistics(self, model_name: str, metric_name: str) -> Dict[str, float]:
        """
        Calculate statistics for a specific model and metric.
        
        Args:
            model_name: Name of the model
            metric_name: Name of the performance metric
            
        Returns:
            Statistics dictionary
        """
        relevant_values = [
            record['value'] for record in self.history
            if record['model_name'] == model_name and record['metric_name'] == metric_name
        ]
        
        if not relevant_values:
            return {}
        
        return {
            'mean': statistics.mean(relevant_values),
            'median': statistics.median(relevant_values),
            'stdev': statistics.stdev(relevant_values) if len(relevant_values) > 1 else 0.0,
            'min': min(relevant_values),
            'max': max(relevant_values),
            'count': len(relevant_values)
        }
    
    def export_regression_report(self, output_file: str = "regression_report.json") -> None:
        """
        Export a comprehensive regression report.
        
        Args:
            output_file: Output file for the regression report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'baselines': {k: {
                'model_name': v.model_name,
                'metric_name': v.metric_name,
                'value': v.value,
                'unit': v.unit,
                'timestamp': v.timestamp,
                'metadata': v.metadata
            } for k, v in self.baselines.items()},
            'recent_measurements': self.history[-50:],  # Last 50 measurements
            'statistics': {}
        }
        
        # Calculate statistics for each unique model-metric combination
        unique_combinations = set((record['model_name'], record['metric_name']) for record in self.history)
        for model_name, metric_name in unique_combinations:
            stats = self.calculate_statistics(model_name, metric_name)
            if stats:
                key = f"{model_name}_{metric_name}"
                report['statistics'][key] = {
                    'model_name': model_name,
                    'metric_name': metric_name,
                    'statistics': stats
                }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)


class PerformanceRegressionTestSuite:
    """Suite for running performance regression tests."""
    
    def __init__(self, tracker: PerformanceRegressionTracker):
        """
        Initialize the test suite.
        
        Args:
            tracker: Performance regression tracker instance
        """
        self.tracker = tracker
    
    def run_regression_tests(self, test_data: List[Dict[str, Any]], 
                           threshold_percentage: float = 5.0) -> List[PerformanceRegressionResult]:
        """
        Run performance regression tests on a set of test data.
        
        Args:
            test_data: List of test data with model_name, metric_name, and value
            threshold_percentage: Threshold percentage for regression detection
            
        Returns:
            List of regression test results
        """
        results = []
        
        for test_case in test_data:
            result = self.tracker.detect_regression(
                model_name=test_case['model_name'],
                metric_name=test_case['metric_name'],
                current_value=test_case['value'],
                threshold_percentage=threshold_percentage
            )
            results.append(result)
        
        return results
    
    def generate_regression_summary(self, results: List[PerformanceRegressionResult]) -> str:
        """
        Generate a summary of regression test results.
        
        Args:
            results: List of regression test results
            
        Returns:
            Summary string
        """
        total_tests = len(results)
        regressions_detected = sum(1 for r in results if r.regression_detected)
        
        summary = f"Performance Regression Test Summary\n"
        summary += f"================================\n"
        summary += f"Total Tests: {total_tests}\n"
        summary += f"Regressions Detected: {regressions_detected}\n"
        summary += f"Success Rate: {(total_tests - regressions_detected) / total_tests * 100:.2f}%\n\n"
        
        if regressions_detected > 0:
            summary += "Detected Regressions:\n"
            summary += "-" * 50 + "\n"
            for result in results:
                if result.regression_detected:
                    direction = "DECREASE" if result.current_value < result.baseline_value else "INCREASE"
                    summary += f"  {result.model_name} - {result.metric_name}: "
                    summary += f"Current={result.current_value:.2f}, Baseline={result.baseline_value:.2f}, "
                    summary += f"Change={result.regression_percentage:.2f}% ({direction})\n"
        
        return summary


# Utility functions for common performance metrics
def calculate_tokens_per_second(total_tokens: int, elapsed_time: float) -> float:
    """
    Calculate tokens per second.
    
    Args:
        total_tokens: Total number of tokens processed
        elapsed_time: Elapsed time in seconds
        
    Returns:
        Tokens per second
    """
    return total_tokens / elapsed_time if elapsed_time > 0 else float('inf')


def calculate_latency_ms(start_time: float, end_time: float) -> float:
    """
    Calculate latency in milliseconds.
    
    Args:
        start_time: Start time in seconds
        end_time: End time in seconds
        
    Returns:
        Latency in milliseconds
    """
    return (end_time - start_time) * 1000


def calculate_memory_increase_mb(before_mb: float, after_mb: float) -> float:
    """
    Calculate memory increase in MB.
    
    Args:
        before_mb: Memory usage before in MB
        after_mb: Memory usage after in MB
        
    Returns:
        Memory increase in MB
    """
    return after_mb - before_mb


def create_default_baselines(tracker: PerformanceRegressionTracker, model_name: str) -> None:
    """
    Create default baselines for a model.
    
    Args:
        tracker: Performance regression tracker
        model_name: Name of the model
    """
    # Example baselines - these would normally come from actual measurements
    default_baselines = [
        {"metric_name": "inference_speed", "value": 100.0, "unit": "tokens/sec"},
        {"metric_name": "memory_usage", "value": 512.0, "unit": "MB"},
        {"metric_name": "model_loading_time", "value": 2.5, "unit": "seconds"},
        {"metric_name": "latency", "value": 150.0, "unit": "ms"},
    ]
    
    for baseline in default_baselines:
        tracker.set_baseline(
            model_name=model_name,
            metric_name=baseline["metric_name"],
            value=baseline["value"],
            unit=baseline["unit"],
            metadata={"source": "default_baseline", "created_at": datetime.now().isoformat()}
        )


# Example usage
if __name__ == "__main__":
    # Create a tracker instance
    tracker = PerformanceRegressionTracker()
    
    # Set a baseline for a model
    tracker.set_baseline(
        model_name="test_model",
        metric_name="inference_speed",
        value=100.0,
        unit="tokens/sec",
        metadata={"device": "CPU", "input_length": 50}
    )
    
    # Test for regression
    result = tracker.detect_regression(
        model_name="test_model",
        metric_name="inference_speed",
        current_value=90.0,  # Lower than baseline, indicating regression
        threshold_percentage=5.0
    )
    
    print(f"Regression detected: {result.regression_detected}")
    print(f"Percentage change: {result.regression_percentage:.2f}%")
    
    # Export report
    tracker.export_regression_report()