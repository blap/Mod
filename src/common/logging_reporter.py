"""
Comprehensive Logging and Reporting for Tests and Benchmarks

This module provides comprehensive logging and reporting capabilities
for tests and benchmarks in the Inference-PIO project.
"""

import csv
import json
import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


class ComprehensiveLogger:
    """
    Comprehensive logging system for tests and benchmarks.
    """

    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.INFO):
        """
        Initialize the comprehensive logger.

        Args:
            name: Logger name
            log_dir: Directory to store logs
            level: Logging level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Prevent adding handlers multiple times
        if not self.logger.handlers:
            # Create file handler
            log_file = (
                self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)

            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


class TestReporter:
    """
    Comprehensive reporting system for tests.
    """

    def __init__(self, report_dir: str = "test_reports"):
        """
        Initialize the test reporter.

        Args:
            report_dir: Directory to store reports
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        self.test_results = []
        self.start_time = None
        self.end_time = None

    def start_test_suite(self, suite_name: str):
        """
        Start a test suite.

        Args:
            suite_name: Name of the test suite
        """
        self.start_time = time.time()
        self.logger = ComprehensiveLogger(f"test_suite_{suite_name}")
        self.logger.info(f"Starting test suite: {suite_name}")

    def add_test_result(
        self,
        test_name: str,
        status: str,
        duration: float,
        error_message: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        """
        Add a test result.

        Args:
            test_name: Name of the test
            status: Test status ('PASS', 'FAIL', 'SKIP')
            duration: Test duration in seconds
            error_message: Error message if test failed
            details: Additional test details
        """
        result = {
            "test_name": test_name,
            "status": status,
            "duration": duration,
            "error_message": error_message,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.test_results.append(result)

        # Log the result
        if hasattr(self, "logger"):
            status_msg = f"Test {test_name} {status} in {duration:.4f}s"
            if status == "FAIL":
                self.logger.error(f"{status_msg} - {error_message}")
            else:
                self.logger.info(status_msg)

    def end_test_suite(self):
        """
        End the test suite and generate reports.
        """
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time if self.start_time else 0

        # Generate reports
        self.generate_json_report(total_duration)
        self.generate_csv_report()
        self.generate_summary_report(total_duration)

    def generate_json_report(self, total_duration: float):
        """
        Generate JSON report.

        Args:
            total_duration: Total test suite duration
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "total_tests": len(self.test_results),
            "passed_tests": len(
                [r for r in self.test_results if r["status"] == "PASS"]
            ),
            "failed_tests": len(
                [r for r in self.test_results if r["status"] == "FAIL"]
            ),
            "skipped_tests": len(
                [r for r in self.test_results if r["status"] == "SKIP"]
            ),
            "results": self.test_results,
        }

        report_file = (
            self.report_dir
            / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        if hasattr(self, "logger"):
            self.logger.info(f"JSON report generated: {report_file}")

    def generate_csv_report(self):
        """
        Generate CSV report.
        """
        report_file = (
            self.report_dir
            / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        with open(report_file, "w", newline="") as f:
            fieldnames = [
                "test_name",
                "status",
                "duration",
                "error_message",
                "timestamp",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.test_results:
                writer.writerow(
                    {
                        "test_name": result["test_name"],
                        "status": result["status"],
                        "duration": f"{result['duration']:.4f}",
                        "error_message": result.get("error_message", ""),
                        "timestamp": result["timestamp"],
                    }
                )

        if hasattr(self, "logger"):
            self.logger.info(f"CSV report generated: {report_file}")

    def generate_summary_report(self, total_duration: float):
        """
        Generate summary report.

        Args:
            total_duration: Total test suite duration
        """
        passed = len([r for r in self.test_results if r["status"] == "PASS"])
        failed = len([r for r in self.test_results if r["status"] == "FAIL"])
        skipped = len([r for r in self.test_results if r["status"] == "SKIP"])
        total = len(self.test_results)

        success_rate = (passed / total * 100) if total > 0 else 0.0
        summary = f"""
Test Suite Summary
==================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Duration: {total_duration:.4f}s
Total Tests: {total}
Passed: {passed}
Failed: {failed}
Skipped: {skipped}
Success Rate: {success_rate:.2f}%

Failed Tests:
"""
        for result in self.test_results:
            if result["status"] == "FAIL":
                summary += f"  - {result['test_name']}: {result.get('error_message', 'Unknown error')}\n"

        summary_file = (
            self.report_dir
            / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(summary_file, "w") as f:
            f.write(summary)

        if hasattr(self, "logger"):
            self.logger.info(f"Summary report generated: {summary_file}")


class BenchmarkReporter:
    """
    Comprehensive reporting system for benchmarks.
    """

    def __init__(self, report_dir: str = "benchmark_reports"):
        """
        Initialize the benchmark reporter.

        Args:
            report_dir: Directory to store reports
        """
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        self.benchmark_results = []
        self.system_info = self._collect_system_info()
        self.start_time = None
        self.end_time = None

    def _collect_system_info(self) -> Dict[str, Any]:
        """
        Collect system information for benchmark context.

        Returns:
            Dictionary with system information
        """
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(
                psutil.virtual_memory().available / (1024**3), 2
            ),
            "timestamp": datetime.now().isoformat(),
        }

    def start_benchmark_suite(self, suite_name: str):
        """
        Start a benchmark suite.

        Args:
            suite_name: Name of the benchmark suite
        """
        self.start_time = time.time()
        self.logger = ComprehensiveLogger(f"benchmark_suite_{suite_name}")
        self.logger.info(f"Starting benchmark suite: {suite_name}")
        self.logger.info(f"System Info: {self.system_info}")

    def add_benchmark_result(
        self,
        benchmark_name: str,
        model_name: str,
        metric_name: str,
        value: float,
        unit: str,
        duration: float,
        details: Optional[Dict] = None,
    ):
        """
        Add a benchmark result.

        Args:
            benchmark_name: Name of the benchmark
            model_name: Name of the model
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            duration: Benchmark duration in seconds
            details: Additional benchmark details
        """
        result = {
            "benchmark_name": benchmark_name,
            "model_name": model_name,
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "duration": duration,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "system_info": self.system_info,
        }
        self.benchmark_results.append(result)

        # Log the result
        if hasattr(self, "logger"):
            self.logger.info(
                f"Benchmark {benchmark_name} for {model_name}: "
                f"{value:.4f} {unit} in {duration:.4f}s"
            )

    def end_benchmark_suite(self):
        """
        End the benchmark suite and generate reports.
        """
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time if self.start_time else 0

        # Generate reports
        self.generate_json_report(total_duration)
        self.generate_csv_report()
        self.generate_comparison_report()

    def generate_json_report(self, total_duration: float):
        """
        Generate JSON report.

        Args:
            total_duration: Total benchmark suite duration
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_duration": total_duration,
            "system_info": self.system_info,
            "total_benchmarks": len(self.benchmark_results),
            "results": self.benchmark_results,
        }

        report_file = (
            self.report_dir
            / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2)

        if hasattr(self, "logger"):
            self.logger.info(f"JSON benchmark report generated: {report_file}")

    def generate_csv_report(self):
        """
        Generate CSV report.
        """
        report_file = (
            self.report_dir
            / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        with open(report_file, "w", newline="") as f:
            fieldnames = [
                "benchmark_name",
                "model_name",
                "metric_name",
                "value",
                "unit",
                "duration",
                "timestamp",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.benchmark_results:
                writer.writerow(
                    {
                        "benchmark_name": result["benchmark_name"],
                        "model_name": result["model_name"],
                        "metric_name": result["metric_name"],
                        "value": result["value"],
                        "unit": result["unit"],
                        "duration": f"{result['duration']:.4f}",
                        "timestamp": result["timestamp"],
                    }
                )

        if hasattr(self, "logger"):
            self.logger.info(f"CSV benchmark report generated: {report_file}")

    def generate_comparison_report(self):
        """
        Generate comparison report across models and metrics.
        """
        # Group results by model
        model_results = {}
        for result in self.benchmark_results:
            model = result["model_name"]
            if model not in model_results:
                model_results[model] = []
            model_results[model].append(result)

        comparison = f"""
Benchmark Comparison Report
=========================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System: {self.system_info['platform']}
CPU: {self.system_info['cpu_count']} cores
Memory: {self.system_info['memory_total_gb']} GB

Model Comparisons:
"""

        # Compare models by metric
        metrics = set(r["metric_name"] for r in self.benchmark_results)

        for metric in metrics:
            comparison += f"\n{metric.upper()} Comparison:\n"
            metric_results = [
                r for r in self.benchmark_results if r["metric_name"] == metric
            ]

            # Sort by value (descending for performance metrics, ascending for latency)
            reverse_sort = metric.lower() in [
                "tokens_per_second",
                "throughput",
                "accuracy",
                "ops_per_sec",
            ]
            sorted_results = sorted(
                metric_results, key=lambda x: x["value"], reverse=reverse_sort
            )

            for result in sorted_results:
                comparison += f"  {result['model_name']}: {result['value']:.4f} {result['unit']}\n"

        comparison_file = (
            self.report_dir
            / f"benchmark_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(comparison_file, "w") as f:
            f.write(comparison)

        if hasattr(self, "logger"):
            self.logger.info(f"Comparison report generated: {comparison_file}")


# Context managers for easy usage
from contextlib import contextmanager


@contextmanager
def test_reporting(suite_name: str, report_dir: str = "test_reports"):
    """
    Context manager for test reporting.

    Args:
        suite_name: Name of the test suite
        report_dir: Directory to store reports
    """
    reporter = TestReporter(report_dir)
    reporter.start_test_suite(suite_name)

    try:
        yield reporter
    finally:
        reporter.end_test_suite()


@contextmanager
def benchmark_reporting(suite_name: str, report_dir: str = "benchmark_reports"):
    """
    Context manager for benchmark reporting.

    Args:
        suite_name: Name of the benchmark suite
        report_dir: Directory to store reports
    """
    reporter = BenchmarkReporter(report_dir)
    reporter.start_benchmark_suite(suite_name)

    try:
        yield reporter
    finally:
        reporter.end_benchmark_suite()


# Example usage functions
def log_test_step(
    logger: ComprehensiveLogger, step_name: str, details: Dict[str, Any] = None
):
    """
    Log a test step with details.

    Args:
        logger: Comprehensive logger instance
        step_name: Name of the test step
        details: Additional step details
    """
    logger.info(f"Executing test step: {step_name}")
    if details:
        logger.debug(f"Step details: {details}")


def log_benchmark_phase(
    logger: ComprehensiveLogger,
    phase_name: str,
    model_name: str,
    details: Dict[str, Any] = None,
):
    """
    Log a benchmark phase with details.

    Args:
        logger: Comprehensive logger instance
        phase_name: Name of the benchmark phase
        model_name: Name of the model being benchmarked
        details: Additional phase details
    """
    logger.info(f"Starting {phase_name} for {model_name}")
    if details:
        logger.debug(f"Phase details: {details}")


def format_metric(value: float, unit: str, precision: int = 4) -> str:
    """
    Format a metric value with unit.

    Args:
        value: Metric value
        unit: Unit of measurement
        precision: Decimal precision

    Returns:
        Formatted metric string
    """
    return f"{value:.{precision}f} {unit}"


__all__ = [
    "ComprehensiveLogger",
    "TestReporter",
    "BenchmarkReporter",
    "test_reporting",
    "benchmark_reporting",
    "log_test_step",
    "log_benchmark_phase",
    "format_metric",
]
