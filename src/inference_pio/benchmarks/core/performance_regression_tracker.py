"""
Performance Regression Tracking System

This module implements a comprehensive system to track performance metrics over time,
detect performance regressions, and provide alerts when performance degrades.
"""

import csv
import json
import logging
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class RegressionSeverity(Enum):
    """Enumeration for regression severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Data class to represent a single performance metric."""

    name: str
    value: float
    unit: str
    timestamp: float
    model_name: str
    category: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RegressionAlert:
    """Data class to represent a performance regression alert."""

    metric_name: str
    model_name: str
    previous_value: float
    current_value: float
    threshold_percentage: float
    severity: RegressionSeverity
    timestamp: float
    message: str


class PerformanceRegressionTracker:
    """Main class for tracking performance metrics and detecting regressions."""

    def __init__(
        self,
        storage_dir: str = "performance_history",
        regression_threshold: float = 5.0,
    ):
        """
        Initialize the performance regression tracker.

        Args:
            storage_dir: Directory to store historical performance data
            regression_threshold: Threshold percentage for detecting regressions
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        self.regression_threshold = regression_threshold  # Percentage
        self.metrics_history: Dict[str, List[PerformanceMetric]] = {}
        self.alerts: List[RegressionAlert] = []

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Load existing history if available
        self.load_history()

    def add_metric(self, metric: PerformanceMetric) -> None:
        """
        Add a new performance metric to the tracker.

        Args:
            metric: PerformanceMetric object to add
        """
        key = f"{metric.model_name}:{metric.name}"

        if key not in self.metrics_history:
            self.metrics_history[key] = []

        self.metrics_history[key].append(metric)

        # Check for regressions
        self.check_regression(key, metric)

    def check_regression(
        self, metric_key: str, current_metric: PerformanceMetric
    ) -> None:
        """
        Check if the current metric represents a regression compared to historical data.

        Args:
            metric_key: Key identifying the metric (model_name:name)
            current_metric: Current metric value to check
        """
        if metric_key not in self.metrics_history:
            return

        history = self.metrics_history[metric_key]

        # Need at least 2 data points to compare
        if len(history) < 2:
            return

        # Get the previous value (most recent before current)
        previous_metric = history[-2]
        current_value = current_metric.value
        previous_value = previous_metric.value

        # Calculate percentage change
        if previous_value != 0:
            percentage_change = (
                (current_value - previous_value) / abs(previous_value)
            ) * 100
        else:
            percentage_change = 0

        # Determine if this is a regression based on the metric type
        is_regression = False
        severity = RegressionSeverity.INFO

        # For performance metrics like tokens/sec, higher is better
        # For time metrics like seconds, lower is better
        if current_metric.category == "performance":
            if current_metric.unit in ["tokens/sec", "requests/sec", "throughput"]:
                # Higher values are better, so negative change is regression
                if percentage_change < -self.regression_threshold:
                    is_regression = True
                    if percentage_change < -2 * self.regression_threshold:
                        severity = RegressionSeverity.CRITICAL
                    else:
                        severity = RegressionSeverity.WARNING
            elif current_metric.unit in ["seconds", "ms", "time"]:
                # Lower values are better, so positive change is regression
                if percentage_change > self.regression_threshold:
                    is_regression = True
                    if percentage_change > 2 * self.regression_threshold:
                        severity = RegressionSeverity.CRITICAL
                    else:
                        severity = RegressionSeverity.WARNING
        elif current_metric.category == "memory":
            # Lower memory usage is better, so positive change is regression
            if percentage_change > self.regression_threshold:
                is_regression = True
                if percentage_change > 2 * self.regression_threshold:
                    severity = RegressionSeverity.CRITICAL
                else:
                    severity = RegressionSeverity.WARNING

        if is_regression:
            message = (
                f"Performance regression detected for {current_metric.model_name}:{current_metric.name}. "
                f"Previous: {previous_value:.2f}{current_metric.unit}, "
                f"Current: {current_value:.2f}{current_metric.unit}, "
                f"Change: {percentage_change:+.2f}%"
            )

            alert = RegressionAlert(
                metric_name=current_metric.name,
                model_name=current_metric.model_name,
                previous_value=previous_value,
                current_value=current_value,
                threshold_percentage=self.regression_threshold,
                severity=severity,
                timestamp=time.time(),
                message=message,
            )

            self.alerts.append(alert)
            self.logger.warning(message)

            # Also log to file
            self.log_alert(alert)

    def get_historical_stats(
        self, metric_key: str, window_size: int = 10
    ) -> Dict[str, float]:
        """
        Get statistical information about a metric's historical performance.

        Args:
            metric_key: Key identifying the metric (model_name:name)
            window_size: Number of recent data points to consider

        Returns:
            Dictionary with statistical information
        """
        if metric_key not in self.metrics_history:
            return {}

        history = self.metrics_history[metric_key][-window_size:]

        if not history:
            return {}

        values = [m.value for m in history]

        stats = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
            "latest": values[-1],
        }

        return stats

    def save_history(self) -> None:
        """Save the current metrics history to storage."""
        # Save metrics history
        history_file = self.storage_dir / "metrics_history.json"
        serializable_history = {}

        for key, metrics in self.metrics_history.items():
            serializable_history[key] = [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "timestamp": m.timestamp,
                    "model_name": m.model_name,
                    "category": m.category,
                    "metadata": m.metadata,
                }
                for m in metrics
            ]

        with open(history_file, "w") as f:
            json.dump(serializable_history, f, indent=2)

        # Save alerts
        alerts_file = self.storage_dir / "regression_alerts.json"
        serializable_alerts = [
            {
                "metric_name": a.metric_name,
                "model_name": a.model_name,
                "previous_value": a.previous_value,
                "current_value": a.current_value,
                "threshold_percentage": a.threshold_percentage,
                "severity": a.severity.value,
                "timestamp": a.timestamp,
                "message": a.message,
            }
            for a in self.alerts
        ]

        with open(alerts_file, "w") as f:
            json.dump(serializable_alerts, f, indent=2)

    def load_history(self) -> None:
        """Load metrics history from storage."""
        history_file = self.storage_dir / "metrics_history.json"

        if history_file.exists():
            with open(history_file, "r") as f:
                serializable_history = json.load(f)

            for key, metrics_data in serializable_history.items():
                self.metrics_history[key] = [
                    PerformanceMetric(
                        name=m["name"],
                        value=m["value"],
                        unit=m["unit"],
                        timestamp=m["timestamp"],
                        model_name=m["model_name"],
                        category=m["category"],
                        metadata=m["metadata"],
                    )
                    for m in metrics_data
                ]

        alerts_file = self.storage_dir / "regression_alerts.json"

        if alerts_file.exists():
            with open(alerts_file, "r") as f:
                serializable_alerts = json.load(f)

            for alert_data in serializable_alerts:
                self.alerts.append(
                    RegressionAlert(
                        metric_name=alert_data["metric_name"],
                        model_name=alert_data["model_name"],
                        previous_value=alert_data["previous_value"],
                        current_value=alert_data["current_value"],
                        threshold_percentage=alert_data["threshold_percentage"],
                        severity=RegressionSeverity(alert_data["severity"]),
                        timestamp=alert_data["timestamp"],
                        message=alert_data["message"],
                    )
                )

    def log_alert(self, alert: RegressionAlert) -> None:
        """Log a regression alert to a dedicated file."""
        alerts_log_file = self.storage_dir / "regression_alerts.log"

        with open(alerts_log_file, "a") as f:
            f.write(
                f"[{datetime.fromtimestamp(alert.timestamp)}] {alert.severity.value.upper()}: {alert.message}\n"
            )

    def generate_report(self, output_dir: str = "performance_reports") -> str:
        """
        Generate a comprehensive performance report.

        Args:
            output_dir: Directory to save the report

        Returns:
            Path to the generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"performance_report_{timestamp}.md"

        with open(report_file, "w") as f:
            f.write("# Performance Report\n\n")
            f.write(f"Generated on: {datetime.now().isoformat()}\n\n")

            # Summary section
            f.write("## Summary\n\n")
            f.write(
                f"- Total metrics tracked: {sum(len(metrics) for metrics in self.metrics_history.values())}\n"
            )
            f.write(f"- Unique metrics: {len(self.metrics_history)}\n")
            f.write(f"- Regression alerts: {len(self.alerts)}\n")
            f.write(
                f"- Critical alerts: {len([a for a in self.alerts if a.severity == RegressionSeverity.CRITICAL])}\n\n"
            )

            # Recent alerts section
            if self.alerts:
                f.write("## Recent Regression Alerts\n\n")
                for alert in reversed(self.alerts[-10:]):  # Last 10 alerts
                    f.write(f"- **{alert.severity.value.upper()}**: {alert.message}\n")
                f.write("\n")

            # Metrics history section
            f.write("## Metrics History\n\n")
            for metric_key, metrics in self.metrics_history.items():
                if metrics:  # Only show metrics that have data
                    latest = metrics[-1]
                    stats = self.get_historical_stats(metric_key)

                    f.write(f"### {metric_key}\n")
                    f.write(f"- Latest value: {latest.value:.2f} {latest.unit}\n")
                    if stats:
                        f.write(
                            f"- Historical mean: {stats['mean']:.2f} {latest.unit}\n"
                        )
                        f.write(
                            f"- Historical stdev: {stats['stdev']:.2f} {latest.unit}\n"
                        )
                        f.write(f"- Min: {stats['min']:.2f}, Max: {stats['max']:.2f}\n")
                    f.write(f"- Total measurements: {stats['count']}\n\n")

        return str(report_file)

    def reset_alerts(self) -> None:
        """Reset the alerts list."""
        self.alerts = []

    def export_csv(self, output_dir: str = "performance_exports") -> str:
        """
        Export all metrics to a CSV file.

        Args:
            output_dir: Directory to save the CSV export

        Returns:
            Path to the exported CSV file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = output_path / f"performance_metrics_export_{timestamp}.csv"

        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "Model Name",
                    "Metric Name",
                    "Value",
                    "Unit",
                    "Category",
                    "Timestamp",
                    "Date Time",
                ]
            )

            # Write data rows
            for metric_key, metrics in self.metrics_history.items():
                for metric in metrics:
                    writer.writerow(
                        [
                            metric.model_name,
                            metric.name,
                            metric.value,
                            metric.unit,
                            metric.category,
                            metric.timestamp,
                            datetime.fromtimestamp(metric.timestamp).isoformat(),
                        ]
                    )

        return str(csv_file)


class PerformanceRegressionTestMixin:
    """Mixin class to add performance regression testing capabilities to test classes."""

    def __init__(self, regression_tracker: PerformanceRegressionTracker = None):
        self.regression_tracker = regression_tracker or PerformanceRegressionTracker()

    def record_performance_metric(
        self,
        name: str,
        value: float,
        unit: str,
        model_name: str,
        category: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a performance metric for regression tracking.

        Args:
            name: Name of the metric
            value: Value of the metric
            unit: Unit of measurement
            model_name: Name of the model being tested
            category: Category of the metric (e.g., 'performance', 'memory')
            metadata: Additional metadata about the measurement
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            model_name=model_name,
            category=category,
            metadata=metadata,
        )

        self.regression_tracker.add_metric(metric)

    def get_regression_alerts(self) -> List[RegressionAlert]:
        """Get any regression alerts that have been triggered."""
        return self.regression_tracker.alerts

    def save_regression_data(self) -> None:
        """Save the regression tracking data."""
        self.regression_tracker.save_history()


# Global instance for easy access
_default_tracker = PerformanceRegressionTracker()


def get_default_tracker() -> PerformanceRegressionTracker:
    """Get the default global performance regression tracker."""
    return _default_tracker


def record_performance_metric(
    name: str,
    value: float,
    unit: str,
    model_name: str,
    category: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Record a performance metric using the default tracker.

    Args:
        name: Name of the metric
        value: Value of the metric
        unit: Unit of measurement
        model_name: Name of the model being tested
        category: Category of the metric (e.g., 'performance', 'memory')
        metadata: Additional metadata about the measurement
    """
    metric = PerformanceMetric(
        name=name,
        value=value,
        unit=unit,
        timestamp=time.time(),
        model_name=model_name,
        category=category,
        metadata=metadata,
    )

    _default_tracker.add_metric(metric)


def get_regression_alerts() -> List[RegressionAlert]:
    """Get regression alerts from the default tracker."""
    return _default_tracker.alerts


def save_regression_data() -> None:
    """Save regression data from the default tracker."""
    _default_tracker.save_history()
