"""
Standalone Pytest Plugin for Comprehensive Test Reporting

This plugin extends pytest with comprehensive reporting capabilities that generate
detailed reports in multiple formats (JSON, XML, HTML, Markdown) for CI/CD integration.
"""

import json
import os
import statistics
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from xml.dom import minidom

import cpuinfo
import psutil
import pytest


@dataclass
class TestResult:
    """Data class to represent a single test result."""

    name: str
    duration: float
    success: bool
    error_message: Optional[str] = None
    traceback_info: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    tags: list = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TestSuiteMetrics:
    """Data class to represent test suite metrics."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    start_time: float
    end_time: float
    average_duration: float
    min_duration: float
    max_duration: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None


@dataclass
class SystemInfo:
    """Data class to represent system information."""

    os_name: str
    os_version: str
    cpu_info: str
    cpu_count: int
    memory_total: float
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: Optional[str] = None
    gpu_count: Optional[int] = None
    gpu_names: Optional[list] = None
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None


class TestReportGenerator:
    """Main class for generating comprehensive test reports."""

    def __init__(self, output_dir: str = "test_reports"):
        """
        Initialize the test report generator.

        Args:
            output_dir: Directory to store generated reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.system_info = self._collect_system_info()

    def _collect_system_info(self) -> SystemInfo:
        """Collect system information for the report."""
        try:
            import torch

            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else None
            gpu_count = torch.cuda.device_count() if cuda_available else None
            gpu_names = (
                [torch.cuda.get_device_name(i) for i in range(gpu_count)]
                if gpu_count
                else None
            )

            # Handle GPU utilization separately as it may fail even if CUDA is available
            gpu_usage = None
            gpu_memory = None
            if cuda_available and gpu_count and gpu_count > 0:
                try:
                    gpu_usage = (
                        torch.cuda.utilization()
                        if hasattr(torch.cuda, "utilization")
                        else None
                    )
                except:
                    gpu_usage = None  # GPU utilization not available

                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (
                        1024**3
                    )
                except:
                    gpu_memory = None  # GPU memory info not available
        except ImportError:
            torch_version = "Not installed"
            cuda_available = False
            cuda_version = None
            gpu_count = None
            gpu_names = None
            gpu_usage = None
            gpu_memory = None

        info = SystemInfo(
            os_name=f"{os.name} ({sys.platform})",
            os_version=(
                getattr(
                    os, "uname", lambda: type("", (), {"version": "Unknown"})()
                )().version
                if hasattr(os, "uname")
                else "Unknown"
            ),
            cpu_info=(
                cpuinfo.get_cpu_info()["brand_raw"]
                if hasattr(cpuinfo.get_cpu_info(), "brand_raw")
                else "Unknown"
            ),
            cpu_count=psutil.cpu_count(logical=True),
            memory_total=psutil.virtual_memory().total / (1024**3),  # GB
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            torch_version=torch_version,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_count=gpu_count,
            gpu_names=gpu_names,
        )

        if gpu_usage is not None:
            info.gpu_usage = gpu_usage
        if gpu_memory is not None:
            info.gpu_memory = gpu_memory

        return info

    def calculate_metrics(self, test_results: list) -> TestSuiteMetrics:
        """
        Calculate comprehensive metrics for test suite execution.

        Args:
            test_results: List of test results

        Returns:
            TestSuiteMetrics object with calculated metrics
        """
        durations = [result.duration for result in test_results]

        passed_tests = sum(1 for result in test_results if result.success)
        failed_tests = sum(1 for result in test_results if not result.success)
        skipped_tests = sum(1 for result in test_results if result.duration == 0)

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        metrics = TestSuiteMetrics(
            total_tests=len(test_results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            execution_time=sum(durations),
            start_time=(
                min(result.start_time for result in test_results if result.start_time)
                if test_results
                else 0
            ),
            end_time=(
                max(result.end_time for result in test_results if result.end_time)
                if test_results
                else 0
            ),
            average_duration=statistics.mean(durations) if durations else 0,
            min_duration=min(durations) if durations else 0,
            max_duration=max(durations) if durations else 0,
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            gpu_usage=self.system_info.gpu_usage,
            gpu_memory=self.system_info.gpu_memory,
        )

        return metrics

    def generate_json_report(
        self, test_results: list, metrics: TestSuiteMetrics
    ) -> str:
        """
        Generate a JSON report with test results and metrics.

        Args:
            test_results: List of test results
            metrics: Calculated test suite metrics

        Returns:
            Path to the generated JSON report
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": asdict(self.system_info),
            "metrics": asdict(metrics),
            "test_results": [asdict(result) for result in test_results],
            "summary": {
                "total_tests": metrics.total_tests,
                "passed": metrics.passed_tests,
                "failed": metrics.failed_tests,
                "skipped": metrics.skipped_tests,
                "pass_rate": (
                    (metrics.passed_tests / metrics.total_tests * 100)
                    if metrics.total_tests > 0
                    else 0
                ),
            },
        }

        json_path = self.output_dir / "test_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, default=str)

        return str(json_path)

    def generate_xml_report(self, test_results: list, metrics: TestSuiteMetrics) -> str:
        """
        Generate an XML JUnit-style report for CI/CD integration.

        Args:
            test_results: List of test results
            metrics: Calculated test suite metrics

        Returns:
            Path to the generated XML report
        """
        root = ET.Element("testsuite")
        root.set("name", "Inference-PIO Test Suite")
        root.set("tests", str(metrics.total_tests))
        root.set("failures", str(metrics.failed_tests))
        root.set("errors", "0")  # In our system, failures are not errors
        root.set("skipped", str(metrics.skipped_tests))
        root.set("time", str(metrics.execution_time))
        root.set("timestamp", datetime.now().isoformat())

        # Add properties element with system info
        properties = ET.SubElement(root, "properties")
        for key, value in asdict(self.system_info).items():
            prop = ET.SubElement(properties, "property")
            prop.set("name", key)
            prop.set("value", str(value))

        # Add test cases
        for result in test_results:
            testcase = ET.SubElement(root, "testcase")
            testcase.set("name", result.name)
            testcase.set("classname", result.name.split(".")[0])
            testcase.set("time", str(result.duration))

            if not result.success:
                failure = ET.SubElement(testcase, "failure")
                failure.set("message", result.error_message or "Test failed")
                failure.text = result.traceback_info or ""

            # Add tags as properties
            if result.tags:
                tags_prop = ET.SubElement(testcase, "properties")
                for tag in result.tags:
                    tag_prop = ET.SubElement(tags_prop, "property")
                    tag_prop.set("name", f"tag_{tag}")
                    tag_prop.set("value", "true")

        # Format the XML nicely
        rough_string = ET.tostring(root, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        xml_path = self.output_dir / "test_report.xml"
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(pretty_xml)

        return str(xml_path)

    def generate_html_report(
        self, test_results: list, metrics: TestSuiteMetrics
    ) -> str:
        """
        Generate an HTML report with visualizations and detailed information.

        Args:
            test_results: List of test results
            metrics: Calculated test suite metrics

        Returns:
            Path to the generated HTML report
        """
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inference-PIO Test Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .summary-card {{
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        .card {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px;
            min-width: 200px;
            text-align: center;
        }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .skipped {{ color: #ffc107; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .success {{ background-color: #d4edda; }}
        .failure {{ background-color: #f8d7da; }}
        .details {{
            margin-top: 10px;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #007bff;
            display: none;
        }}
        .toggle-details {{
            cursor: pointer;
            color: #007bff;
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Inference-PIO Test Report</h1>
        
        <h2>Summary</h2>
        <div class="summary-card">
            <div class="card">
                <h3>Total Tests</h3>
                <p>{metrics.total_tests}</p>
            </div>
            <div class="card passed">
                <h3>Passed</h3>
                <p>{metrics.passed_tests}</p>
            </div>
            <div class="card failed">
                <h3>Failed</h3>
                <p>{metrics.failed_tests}</p>
            </div>
            <div class="card skipped">
                <h3>Skipped</h3>
                <p>{metrics.skipped_tests}</p>
            </div>
            <div class="card">
                <h3>Pass Rate</h3>
                <p>{(metrics.passed_tests / metrics.total_tests * 100):.2f}%</p>
            </div>
            <div class="card">
                <h3>Execution Time</h3>
                <p>{metrics.execution_time:.2f}s</p>
            </div>
        </div>

        <h2>System Information</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>OS</td><td>{self.system_info.os_name}</td></tr>
            <tr><td>Python Version</td><td>{self.system_info.python_version}</td></tr>
            <tr><td>Torch Version</td><td>{self.system_info.torch_version}</td></tr>
            <tr><td>CUDA Available</td><td>{'Yes' if self.system_info.cuda_available else 'No'}</td></tr>
            <tr><td>CPU Count</td><td>{self.system_info.cpu_count}</td></tr>
            <tr><td>Memory Total</td><td>{self.system_info.memory_total:.2f} GB</td></tr>
        </table>

        <h2>Performance Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Average Duration</td><td>{metrics.average_duration:.2f}s</td></tr>
            <tr><td>Min Duration</td><td>{metrics.min_duration:.2f}s</td></tr>
            <tr><td>Max Duration</td><td>{metrics.max_duration:.2f}s</td></tr>
            <tr><td>CPU Usage</td><td>{metrics.cpu_usage}%</td></tr>
            <tr><td>Memory Usage</td><td>{metrics.memory_usage}%</td></tr>
        </table>

        <h2>Test Results</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Status</th>
                <th>Duration (s)</th>
                <th>Error Message</th>
                <th>Details</th>
            </tr>
        """

        for result in test_results:
            status_class = "success" if result.success else "failure"
            status_text = "PASS" if result.success else "FAIL"

            error_msg = (
                result.error_message[:100] + "..."
                if result.error_message and len(result.error_message) > 100
                else (result.error_message or "")
            )

            html_content += f"""
            <tr class="{status_class}">
                <td>{result.name}</td>
                <td>{status_text}</td>
                <td>{result.duration:.2f}</td>
                <td>{error_msg}</td>
                <td>
                    <span class="toggle-details" onclick="toggleDetails('{result.name}')">View Details</span>
                    <div id="details-{result.name}" class="details">
                        <strong>Full Error:</strong><br>
                        {result.error_message or 'None'}<br><br>
                        <strong>Traceback:</strong><br>
                        {result.traceback_info or 'None'}<br><br>
                        <strong>Tags:</strong> {', '.join(result.tags) or 'None'}
                    </div>
                </td>
            </tr>
            """

        html_content += """
        </table>
    </div>

    <script>
        function toggleDetails(testName) {
            const detailsDiv = document.getElementById('details-' + testName);
            detailsDiv.style.display = detailsDiv.style.display === 'none' ? 'block' : 'none';
        }
    </script>
</body>
</html>
        """

        html_path = self.output_dir / "test_report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return str(html_path)

    def generate_markdown_report(
        self, test_results: list, metrics: TestSuiteMetrics
    ) -> str:
        """
        Generate a markdown report for documentation and CI/CD logs.

        Args:
            test_results: List of test results
            metrics: Calculated test suite metrics

        Returns:
            Path to the generated markdown report
        """
        md_content = f"""# Inference-PIO Test Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {metrics.total_tests} |
| Passed | {metrics.passed_tests} |
| Failed | {metrics.failed_tests} |
| Skipped | {metrics.skipped_tests} |
| Pass Rate | {(metrics.passed_tests / metrics.total_tests * 100):.2f}% |
| Execution Time | {metrics.execution_time:.2f}s |

## System Information

- OS: {self.system_info.os_name}
- Python Version: {self.system_info.python_version}
- Torch Version: {self.system_info.torch_version}
- CUDA Available: {'Yes' if self.system_info.cuda_available else 'No'}
- CPU Count: {self.system_info.cpu_count}
- Memory Total: {self.system_info.memory_total:.2f} GB

## Performance Metrics

- Average Duration: {metrics.average_duration:.2f}s
- Min Duration: {metrics.min_duration:.2f}s
- Max Duration: {metrics.max_duration:.2f}s
- CPU Usage: {metrics.cpu_usage}%
- Memory Usage: {metrics.memory_usage}%

## Test Results

| Test Name | Status | Duration (s) | Error Message |
|-----------|--------|--------------|---------------|
"""

        for result in test_results:
            status = "PASS" if result.success else "FAIL"
            error_msg = (
                (result.error_message[:50] + "...")
                if result.error_message and len(result.error_message) > 50
                else (result.error_message or "")
            )

            md_content += (
                f"| {result.name} | {status} | {result.duration:.2f} | {error_msg} |\n"
            )

        md_content += "\n## Failed Tests Details\n\n"

        failed_tests = [r for r in test_results if not r.success]
        for result in failed_tests:
            md_content += f"### {result.name}\n\n"
            md_content += f"**Error:** {result.error_message}\n\n"
            md_content += f"**Traceback:**\n```\n{result.traceback_info}\n```\n\n"

        md_path = self.output_dir / "test_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return str(md_path)

    def generate_all_reports(self, test_results: list) -> Dict[str, str]:
        """
        Generate all types of reports (JSON, XML, HTML, Markdown).

        Args:
            test_results: List of test results

        Returns:
            Dictionary mapping report type to file path
        """
        metrics = self.calculate_metrics(test_results)

        report_paths = {
            "json": self.generate_json_report(test_results, metrics),
            "xml": self.generate_xml_report(test_results, metrics),
            "html": self.generate_html_report(test_results, metrics),
            "markdown": self.generate_markdown_report(test_results, metrics),
        }

        return report_paths


class ComprehensiveTestReporter:
    """Pytest plugin for generating comprehensive test reports."""

    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = output_dir
        self.test_results = []
        self.start_time = None
        self.report_generator = TestReportGenerator(output_dir)

    def pytest_configure(self, config):
        """Configure the plugin."""
        self.config = config

    def pytest_sessionstart(self, session):
        """Called when test session starts."""
        self.start_time = time.time()

    def pytest_runtest_logreport(self, report):
        """Process test results."""
        if report.when == "call":  # Only process the call phase, not setup/teardown
            duration = getattr(report, "duration", 0)

            # Create test result object
            test_result = TestResult(
                name=report.nodeid,
                duration=duration,
                success=report.passed,
                error_message=str(report.longrepr) if report.failed else None,
                traceback_info=(
                    traceback.format_tb(report.longrepr.tb)
                    if report.failed and hasattr(report.longrepr, "tb")
                    else None
                ),
                start_time=time.time() - duration if duration > 0 else None,
                end_time=time.time() if duration > 0 else None,
            )

            # Extract tags from test name or markers
            tags = []
            if hasattr(report, "keywords"):
                tags.extend(list(report.keywords.keys()))

            # Add common tags based on test path
            if "unit" in report.nodeid:
                tags.append("unit")
            elif "integration" in report.nodeid:
                tags.append("integration")
            elif "performance" in report.nodeid:
                tags.append("performance")

            test_result.tags = tags

            self.test_results.append(test_result)

    def pytest_sessionfinish(self, session, exitstatus):
        """Called when test session finishes."""
        # Generate all reports
        if self.test_results:
            report_paths = self.report_generator.generate_all_reports(self.test_results)

            print("\n" + "=" * 60)
            print("COMPREHENSIVE TEST REPORTS GENERATED")
            print("=" * 60)
            for report_type, path in report_paths.items():
                print(f"{report_type.upper()} Report: {path}")
            print("=" * 60)


def pytest_configure(config):
    """Register the plugin."""
    if config.getoption("--generate-reports", default=False):
        reporter = ComprehensiveTestReporter(
            output_dir=config.getoption("--reports-dir", default="test_reports")
        )
        config.pluginmanager.register(reporter, "comprehensive-test-reporter")


def pytest_addoption(parser):
    """Add options to pytest."""
    group = parser.getgroup("test-reporting")
    group.addoption(
        "--generate-reports",
        action="store_true",
        default=False,
        help="Generate comprehensive test reports",
    )
    group.addoption(
        "--reports-dir",
        action="store",
        default="test_reports",
        help="Directory to store generated reports",
    )
