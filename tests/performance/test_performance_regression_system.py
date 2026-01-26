"""
Integration Tests for Performance Regression System

This module tests the performance regression tracking system to ensure
it properly detects and reports performance regressions.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import time

from src.inference_pio.common.performance_regression_tracker import (
    PerformanceRegressionTracker,
    PerformanceMetric,
    RegressionAlert,
    RegressionSeverity
)


class TestPerformanceRegressionSystem(unittest.TestCase):
    """Test the performance regression tracking system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test storage
        self.temp_dir = Path(tempfile.mkdtemp())
        self.tracker = PerformanceRegressionTracker(
            storage_dir=self.temp_dir,
            regression_threshold=5.0  # 5% threshold
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_basic_metric_tracking(self):
        """Test basic metric tracking functionality."""
        # Add a few metrics
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        
        self.tracker.add_metric(metric1)
        
        # Verify the metric was added
        self.assertIn("test_model:inference_speed", self.tracker.metrics_history)
        self.assertEqual(len(self.tracker.metrics_history["test_model:inference_speed"]), 1)
        self.assertEqual(self.tracker.metrics_history["test_model:inference_speed"][0].value, 100.0)
    
    def test_regression_detection_performance_metric(self):
        """Test regression detection for performance metrics (higher is better)."""
        # Add initial high-performance metric
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,  # Good performance: 100 tokens/sec
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        self.tracker.add_metric(metric1)
        
        # Add degraded performance metric (>5% worse)
        metric2 = PerformanceMetric(
            name="inference_speed",
            value=94.0,  # Degraded: ~6% worse
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        self.tracker.add_metric(metric2)
        
        # Verify regression was detected
        self.assertEqual(len(self.tracker.alerts), 1)
        alert = self.tracker.alerts[0]
        self.assertEqual(alert.severity, RegressionSeverity.WARNING)
        self.assertIn("Performance regression detected", alert.message)
        self.assertAlmostEqual(alert.previous_value, 100.0, places=1)
        self.assertAlmostEqual(alert.current_value, 94.0, places=1)
    
    def test_regression_detection_time_metric(self):
        """Test regression detection for time metrics (lower is better)."""
        # Add initial low-time metric
        metric1 = PerformanceMetric(
            name="inference_time",
            value=0.1,  # Fast: 0.1 seconds
            unit="seconds",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        self.tracker.add_metric(metric1)
        
        # Add degraded time metric (>5% worse)
        metric2 = PerformanceMetric(
            name="inference_time",
            value=0.11,  # Slower: 10% worse
            unit="seconds",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        self.tracker.add_metric(metric2)
        
        # Verify regression was detected
        self.assertEqual(len(self.tracker.alerts), 1)
        alert = self.tracker.alerts[0]
        self.assertEqual(alert.severity, RegressionSeverity.WARNING)
        self.assertIn("Performance regression detected", alert.message)
    
    def test_no_regression_below_threshold(self):
        """Test that regressions below threshold are not detected."""
        # Add initial metric
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        self.tracker.add_metric(metric1)
        
        # Add slightly degraded metric (<5% worse)
        metric2 = PerformanceMetric(
            name="inference_speed",
            value=98.0,  # Only 2% worse, below threshold
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        self.tracker.add_metric(metric2)
        
        # Verify no regression was detected
        self.assertEqual(len(self.tracker.alerts), 0)
    
    def test_improvement_not_flagged_as_regression(self):
        """Test that performance improvements are not flagged as regressions."""
        # Add initial metric
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        self.tracker.add_metric(metric1)
        
        # Add improved metric
        metric2 = PerformanceMetric(
            name="inference_speed",
            value=110.0,  # Better performance
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        self.tracker.add_metric(metric2)
        
        # Verify no regression was detected
        self.assertEqual(len(self.tracker.alerts), 0)
    
    def test_critical_regression_detection(self):
        """Test detection of critical regressions (>2x threshold)."""
        # Add initial metric
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        self.tracker.add_metric(metric1)
        
        # Add severely degraded metric (>10% worse, which is 2x the 5% threshold)
        metric2 = PerformanceMetric(
            name="inference_speed",
            value=85.0,  # 15% worse, >2x threshold
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        self.tracker.add_metric(metric2)
        
        # Verify critical regression was detected
        self.assertEqual(len(self.tracker.alerts), 1)
        alert = self.tracker.alerts[0]
        self.assertEqual(alert.severity, RegressionSeverity.CRITICAL)
    
    def test_memory_regression_detection(self):
        """Test regression detection for memory usage (lower is better)."""
        # Add initial low memory usage
        metric1 = PerformanceMetric(
            name="memory_usage",
            value=500.0,  # 500 MB
            unit="MB",
            timestamp=time.time(),
            model_name="test_model",
            category="memory"
        )
        self.tracker.add_metric(metric1)
        
        # Add increased memory usage (>5% worse)
        metric2 = PerformanceMetric(
            name="memory_usage",
            value=550.0,  # 10% increase, >5% threshold
            unit="MB",
            timestamp=time.time(),
            model_name="test_model",
            category="memory"
        )
        self.tracker.add_metric(metric2)
        
        # Verify regression was detected
        self.assertEqual(len(self.tracker.alerts), 1)
        alert = self.tracker.alerts[0]
        self.assertEqual(alert.severity, RegressionSeverity.WARNING)
        self.assertIn("Performance regression detected", alert.message)
    
    def test_historical_stats_calculation(self):
        """Test calculation of historical statistics."""
        # Add several metrics
        for i, value in enumerate([100, 105, 95, 110, 90]):
            metric = PerformanceMetric(
                name="inference_speed",
                value=value,
                unit="tokens/sec",
                timestamp=time.time() + i,
                model_name="test_model",
                category="performance"
            )
            self.tracker.add_metric(metric)
        
        # Get historical stats
        stats = self.tracker.get_historical_stats("test_model:inference_speed")
        
        # Verify stats were calculated
        self.assertIn('mean', stats)
        self.assertIn('median', stats)
        self.assertIn('stdev', stats)
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('count', stats)
        self.assertIn('latest', stats)
        
        # Verify specific values
        self.assertAlmostEqual(stats['mean'], 100.0, places=1)
        self.assertEqual(stats['min'], 90)
        self.assertEqual(stats['max'], 110)
        self.assertEqual(stats['count'], 5)
        self.assertEqual(stats['latest'], 90)
    
    def test_save_and_load_history(self):
        """Test saving and loading of historical data."""
        # Add some metrics
        for i, value in enumerate([100, 105, 95]):
            metric = PerformanceMetric(
                name="inference_speed",
                value=value,
                unit="tokens/sec",
                timestamp=time.time() + i,
                model_name="test_model",
                category="performance"
            )
            self.tracker.add_metric(metric)
        
        # Save history
        self.tracker.save_history()
        
        # Create new tracker and load history
        new_tracker = PerformanceRegressionTracker(
            storage_dir=self.temp_dir,
            regression_threshold=5.0
        )
        
        # Verify history was loaded
        self.assertIn("test_model:inference_speed", new_tracker.metrics_history)
        self.assertEqual(len(new_tracker.metrics_history["test_model:inference_speed"]), 3)
        self.assertEqual(new_tracker.metrics_history["test_model:inference_speed"][0].value, 100)
        self.assertEqual(new_tracker.metrics_history["test_model:inference_speed"][2].value, 95)
    
    def test_report_generation(self):
        """Test generation of performance reports."""
        # Add some metrics and alerts
        metric1 = PerformanceMetric(
            name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            timestamp=time.time(),
            model_name="test_model",
            category="performance"
        )
        self.tracker.add_metric(metric1)
        
        # Generate report
        report_path = self.tracker.generate_report(output_dir=self.temp_dir / "reports")
        
        # Verify report was created
        self.assertTrue(Path(report_path).exists())
        
        # Read and verify content
        with open(report_path, 'r') as f:
            content = f.read()
        
        self.assertIn("# Performance Report", content)
        self.assertIn("test_model:inference_speed", content)
        self.assertIn("Total metrics tracked: 1", content)
    
    def test_csv_export(self):
        """Test CSV export functionality."""
        # Add some metrics
        for i, value in enumerate([100, 105, 95]):
            metric = PerformanceMetric(
                name="inference_speed",
                value=value,
                unit="tokens/sec",
                timestamp=time.time() + i,
                model_name="test_model",
                category="performance"
            )
            self.tracker.add_metric(metric)
        
        # Export to CSV
        csv_path = self.tracker.export_csv(output_dir=self.temp_dir / "exports")
        
        # Verify CSV was created
        self.assertTrue(Path(csv_path).exists())
        
        # Read and verify content
        with open(csv_path, 'r') as f:
            content = f.read()
        
        self.assertIn("Model Name,Metric Name,Value,Unit,Category,Timestamp,Date Time", content)
        self.assertIn("test_model,inference_speed,95.0,tokens/sec,performance,", content)


class TestPerformanceRegressionIntegration(unittest.TestCase):
    """Test integration with the broader system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_global_tracker_access(self):
        """Test access to the global tracker."""
        from src.inference_pio.common.performance_regression_tracker import (
            get_default_tracker,
            record_performance_metric,
            get_regression_alerts,
            save_regression_data
        )
        
        # Get the default tracker
        tracker = get_default_tracker()
        
        # Verify it's accessible and functional
        self.assertIsInstance(tracker, PerformanceRegressionTracker)
        
        # Record a metric using the global function
        record_performance_metric(
            name="test_metric",
            value=42.0,
            unit="units",
            model_name="test_model",
            category="test"
        )
        
        # Verify the metric was recorded
        alerts = get_regression_alerts()
        # No regression should be detected with just one metric
        self.assertEqual(len(alerts), 0)
        
        # Save the data
        save_regression_data()


if __name__ == "__main__":
    unittest.main()