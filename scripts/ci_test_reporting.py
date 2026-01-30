#!/usr/bin/env python3
"""
CI/CD Integration Script for Test Reporting

This script provides functionality to integrate test reporting with CI/CD pipelines,
including GitHub Actions, GitLab CI, Jenkins, and other popular CI/CD platforms.
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import shutil
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.utils.reporting import TestReportGenerator, TestResult


def run_pytest_with_reporting(pytest_args: List[str], reports_dir: str = "test_reports") -> Dict[str, Any]:
    """
    Run pytest with comprehensive reporting enabled.

    Args:
        pytest_args: Arguments to pass to pytest
        reports_dir: Directory to store reports

    Returns:
        Dictionary with test results and report paths
    """
    # Construct the pytest command with reporting options
    cmd = [
        sys.executable, "-m", "pytest",
        "--generate-reports",
        f"--reports-dir={reports_dir}",
        "--tb=short"
    ] + pytest_args

    print(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    # Return the result
    return {
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def upload_reports_to_artifacts(reports_dir: str = "test_reports"):
    """
    Prepare reports for CI/CD artifact upload.

    Args:
        reports_dir: Directory containing reports to upload
    """
    reports_path = Path(reports_dir)
    if not reports_path.exists():
        print(f"Reports directory {reports_dir} does not exist")
        return

    print(f"Preparing reports in {reports_dir} for artifact upload...")
    
    # In CI environments, artifacts are typically uploaded automatically
    # This function prepares the reports for upload by ensuring they're in the right place
    print(f"Reports ready for upload from: {reports_path.absolute()}")


def parse_junit_xml_report(xml_path: str) -> Dict[str, Any]:
    """
    Parse JUnit XML report to extract summary information.

    Args:
        xml_path: Path to the JUnit XML report

    Returns:
        Dictionary with parsed report information
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    summary = {
        "tests": int(root.attrib.get("tests", 0)),
        "failures": int(root.attrib.get("failures", 0)),
        "errors": int(root.attrib.get("errors", 0)),
        "skipped": int(root.attrib.get("skipped", 0)),
        "time": float(root.attrib.get("time", 0)),
        "timestamp": root.attrib.get("timestamp", ""),
    }

    # Extract test cases
    test_cases = []
    for testcase in root.findall(".//testcase"):
        tc_data = {
            "name": testcase.attrib.get("name", ""),
            "classname": testcase.attrib.get("classname", ""),
            "time": float(testcase.attrib.get("time", 0)),
        }

        # Check for failure
        failure = testcase.find("failure")
        if failure is not None:
            tc_data["status"] = "FAILED"
            tc_data["message"] = failure.attrib.get("message", "")
            tc_data["type"] = failure.attrib.get("type", "")
            tc_data["detail"] = failure.text or ""
        elif testcase.find("error") is not None:
            tc_data["status"] = "ERROR"
        elif testcase.find("skipped") is not None:
            tc_data["status"] = "SKIPPED"
        else:
            tc_data["status"] = "PASSED"

        test_cases.append(tc_data)

    summary["test_cases"] = test_cases
    return summary


def generate_ci_summary(summary: Dict[str, Any], output_file: str = "ci_summary.md"):
    """
    Generate a CI summary in Markdown format for GitHub Actions.

    Args:
        summary: Parsed test summary
        output_file: Output file for the summary
    """
    markdown_content = f"""# Test Results Summary

## Overview
- **Total Tests**: {summary['tests']}
- **Passed**: {summary['tests'] - summary['failures'] - summary['errors'] - summary['skipped']}
- **Failed**: {summary['failures']}
- **Errors**: {summary['errors']}
- **Skipped**: {summary['skipped']}
- **Execution Time**: {summary['time']:.2f}s

## Test Results
"""

    # Add details for failed tests
    failed_tests = [tc for tc in summary.get('test_cases', []) if tc['status'] in ['FAILED', 'ERROR']]
    if failed_tests:
        markdown_content += "\n### Failed Tests\n\n"
        for test_case in failed_tests:
            markdown_content += f"- **{test_case['name']}**\n"
            if 'message' in test_case:
                markdown_content += f"  - {test_case['message']}\n"
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"CI summary written to {output_file}")


def github_actions_integration(reports_dir: str = "test_reports"):
    """
    Special integration for GitHub Actions.

    Args:
        reports_dir: Directory containing test reports
    """
    xml_report = Path(reports_dir) / "test_report.xml"
    if xml_report.exists():
        # Parse the XML report
        summary = parse_junit_xml_report(str(xml_report))
        
        # Generate CI summary
        generate_ci_summary(summary)
        
        # Set GitHub Actions outputs
        if os.getenv("GITHUB_ACTIONS"):
            # Write to GitHub Actions step summary
            with open(os.environ.get("GITHUB_STEP_SUMMARY", "/dev/null"), "a") as f:
                f.write(f"\n## Test Results\n")
                f.write(f"- Total: {summary['tests']}\n")
                f.write(f"- Passed: {summary['tests'] - summary['failures'] - summary['errors'] - summary['skipped']}\n")
                f.write(f"- Failed: {summary['failures']}\n")
                f.write(f"- Errors: {summary['errors']}\n")
                
            # Set outputs for conditional steps
            with open(os.environ.get("GITHUB_OUTPUT", "/dev/null"), "a") as f:
                f.write(f"total_tests={summary['tests']}\n")
                f.write(f"passed_tests={summary['tests'] - summary['failures'] - summary['errors'] - summary['skipped']}\n")
                f.write(f"failed_tests={summary['failures']}\n")
                f.write(f"errors={summary['errors']}\n")
                f.write(f"has_failures={'true' if summary['failures'] > 0 else 'false'}\n")


def main():
    parser = argparse.ArgumentParser(description="CI/CD Test Reporting Integration")
    parser.add_argument(
        "--pytest-args",
        nargs="*",
        default=[],
        help="Arguments to pass to pytest (e.g., test directories, markers)"
    )
    parser.add_argument(
        "--reports-dir",
        default="test_reports",
        help="Directory to store test reports"
    )
    parser.add_argument(
        "--no-run-tests",
        action="store_true",
        help="Skip running tests, only process existing reports"
    )
    parser.add_argument(
        "--ci-platform",
        choices=["github-actions", "gitlab-ci", "jenkins", "generic"],
        default="github-actions",
        help="Target CI/CD platform for integration"
    )

    args = parser.parse_args()

    # Create reports directory
    Path(args.reports_dir).mkdir(exist_ok=True)

    if not args.no_run_tests:
        # Run tests with reporting
        result = run_pytest_with_reporting(args.pytest_args, args.reports_dir)
        
        if result["return_code"] != 0:
            print(f"Pytest failed with return code: {result['return_code']}")
            sys.exit(result["return_code"])

    # Perform CI/CD platform-specific integration
    if args.ci_platform == "github-actions":
        github_actions_integration(args.reports_dir)
    elif args.ci_platform == "gitlab-ci":
        # GitLab CI integration would go here
        print("GitLab CI integration placeholder")
    elif args.ci_platform == "jenkins":
        # Jenkins integration would go here
        print("Jenkins integration placeholder")
    else:
        print("Generic CI/CD integration")

    # Prepare reports for artifact upload
    upload_reports_to_artifacts(args.reports_dir)


if __name__ == "__main__":
    main()