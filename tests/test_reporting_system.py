"""
Test suite for the comprehensive test reporting system.
"""

import pytest
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.utils.reporting import TestReportGenerator, TestResult


def test_basic_functionality():
    """Test basic functionality of the reporting system."""
    # Simulate some test results
    test_results = [
        TestResult(
            name="test_basic_addition",
            duration=0.1,
            success=True
        ),
        TestResult(
            name="test_string_concatenation",
            duration=0.2,
            success=True
        ),
        TestResult(
            name="test_failing_case",
            duration=0.15,
            success=False,
            error_message="AssertionError: Expected 5, got 4",
            traceback_info="Traceback...\nAssertionError: Expected 5, got 4"
        )
    ]
    
    # Create report generator
    generator = TestReportGenerator(output_dir="test_output/reports")
    
    # Generate all reports
    report_paths = generator.generate_all_reports(test_results)
    
    # Verify all report types were generated
    assert 'json' in report_paths
    assert 'xml' in report_paths
    assert 'html' in report_paths
    assert 'markdown' in report_paths
    
    # Verify files exist
    for path in report_paths.values():
        assert Path(path).exists(), f"Report file does not exist: {path}"


def test_performance_metrics():
    """Test that performance metrics are calculated correctly."""
    test_results = [
        TestResult(name="fast_test", duration=0.1, success=True),
        TestResult(name="medium_test", duration=0.5, success=True),
        TestResult(name="slow_test", duration=1.2, success=True),
    ]
    
    generator = TestReportGenerator(output_dir="test_output/reports")
    metrics = generator.calculate_metrics(test_results)
    
    assert metrics.total_tests == 3
    assert metrics.passed_tests == 3
    assert metrics.failed_tests == 0
    assert metrics.average_duration == 0.6  # (0.1 + 0.5 + 1.2) / 3
    assert metrics.min_duration == 0.1
    assert metrics.max_duration == 1.2


def test_failed_test_handling():
    """Test handling of failed tests in reports."""
    test_results = [
        TestResult(
            name="failing_test",
            duration=0.3,
            success=False,
            error_message="ValueError: Invalid input",
            traceback_info="Traceback (most recent call):\n  File \"test.py\", line 10, in failing_test\n    raise ValueError(\"Invalid input\")"
        )
    ]
    
    generator = TestReportGenerator(output_dir="test_output/reports")
    report_paths = generator.generate_all_reports(test_results)
    
    # Verify reports were generated even with failed tests
    for path in report_paths.values():
        assert Path(path).exists()


def test_tags_and_metadata():
    """Test inclusion of tags and metadata in reports."""
    test_results = [
        TestResult(
            name="tagged_test",
            duration=0.2,
            success=True,
            tags=["unit", "core", "critical"],
            metadata={"priority": "high", "component": "tokenizer"}
        )
    ]
    
    generator = TestReportGenerator(output_dir="test_output/reports")
    report_paths = generator.generate_all_reports(test_results)
    
    # Verify reports were generated with tags
    for path in report_paths.values():
        assert Path(path).exists()


def test_long_running_test():
    """Test handling of longer running tests."""
    # Simulate a longer test
    start_time = time.time()
    time.sleep(0.1)  # Sleep briefly to simulate work
    duration = time.time() - start_time
    
    test_results = [
        TestResult(
            name="long_running_test",
            duration=duration,
            success=True
        )
    ]
    
    generator = TestReportGenerator(output_dir="test_output/reports")
    metrics = generator.calculate_metrics(test_results)
    
    # The duration should be at least the sleep time (with some tolerance for system overhead)
    assert metrics.max_duration >= 0.09


if __name__ == "__main__":
    # Run the tests directly if this file is executed
    test_basic_functionality()
    test_performance_metrics()
    test_failed_test_handling()
    test_tags_and_metadata()
    test_long_running_test()
    print("All tests passed!")