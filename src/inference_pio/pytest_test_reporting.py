"""
Pytest Plugin for Comprehensive Test Reporting

This plugin extends pytest with comprehensive reporting capabilities that generate
detailed reports in multiple formats (JSON, XML, HTML, Markdown) for CI/CD integration.
"""

import pytest
import time
import traceback
from pathlib import Path
import sys
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference_pio.test_reporting import TestReportGenerator, TestResult


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
            duration = getattr(report, 'duration', 0)
            
            # Create test result object
            test_result = TestResult(
                name=report.nodeid,
                duration=duration,
                success=report.passed,
                error_message=str(report.longrepr) if report.failed else None,
                traceback_info=traceback.format_tb(report.longrepr.tb) if report.failed and hasattr(report.longrepr, 'tb') else None,
                start_time=time.time() - duration if duration > 0 else None,
                end_time=time.time() if duration > 0 else None
            )
            
            # Extract tags from test name or markers
            tags = []
            if hasattr(report, 'keywords'):
                tags.extend(list(report.keywords.keys()))
            
            # Add common tags based on test path
            if 'unit' in report.nodeid:
                tags.append('unit')
            elif 'integration' in report.nodeid:
                tags.append('integration')
            elif 'performance' in report.nodeid:
                tags.append('performance')
            
            test_result.tags = tags
            
            self.test_results.append(test_result)

    def pytest_sessionfinish(self, session, exitstatus):
        """Called when test session finishes."""
        # Generate all reports
        if self.test_results:
            report_paths = self.report_generator.generate_all_reports(self.test_results)
            
            print("\n" + "="*60)
            print("COMPREHENSIVE TEST REPORTS GENERATED")
            print("="*60)
            for report_type, path in report_paths.items():
                print(f"{report_type.upper()} Report: {path}")
            print("="*60)


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
        help="Generate comprehensive test reports"
    )
    group.addoption(
        "--reports-dir",
        action="store",
        default="test_reports",
        help="Directory to store generated reports"
    )