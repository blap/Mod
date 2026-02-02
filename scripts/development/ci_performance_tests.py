#!/usr/bin/env python3
"""
CI/CD Integration Script for Performance Regression Testing

This script provides integration points for CI/CD pipelines to run performance
regression tests and handle alerts appropriately.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference_pio.common.performance_regression_tracker import (
    PerformanceRegressionTracker,
    RegressionAlert,
    get_regression_alerts,
    save_regression_data,
)
from src.inference_pio.common.test_performance_regression import (
    run_all_performance_regression_tests,
)


def send_alert(alerts: List[RegressionAlert], webhook_url: str = None):
    """
    Send alerts about performance regressions.

    Args:
        alerts: List of regression alerts to send
        webhook_url: Optional webhook URL to send alerts to
    """
    if not alerts:
        print("No performance regressions detected.")
        return

    print(f"Detected {len(alerts)} performance regression(s):")
    for alert in alerts:
        print(f"  - {alert.message}")

    # If webhook URL is provided, send alert there
    if webhook_url:
        import requests

        # Group alerts by severity
        critical_alerts = [a for a in alerts if a.severity.value == "critical"]
        warning_alerts = [a for a in alerts if a.severity.value == "warning"]

        message = {
            "text": f"Performance Regression Alert: {len(alerts)} issue(s) detected",
            "attachments": [],
        }

        if critical_alerts:
            critical_attachment = {
                "color": "danger",
                "title": f"Critical Regressions ({len(critical_alerts)})",
                "text": "\n".join([a.message for a in critical_alerts]),
            }
            message["attachments"].append(critical_attachment)

        if warning_alerts:
            warning_attachment = {
                "color": "warning",
                "title": f"Warning Regressions ({len(warning_alerts)})",
                "text": "\n".join([a.message for a in warning_alerts]),
            }
            message["attachments"].append(warning_attachment)

        try:
            response = requests.post(webhook_url, json=message)
            if response.status_code == 200:
                print(f"Alert sent successfully to {webhook_url}")
            else:
                print(f"Failed to send alert: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error sending alert: {e}")


def run_ci_performance_tests(
    threshold: float = 5.0, webhook_url: str = None, fail_on_regression: bool = True
) -> int:
    """
    Run performance tests in CI environment.

    Args:
        threshold: Regression detection threshold percentage
        webhook_url: Optional webhook URL to send alerts to
        fail_on_regression: Whether to fail the build on regression detection

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("Starting CI performance regression tests...")
    print(f"Threshold: {threshold}%")
    print(f"Fail on regression: {fail_on_regression}")

    # Set the threshold in the tracker
    tracker = PerformanceRegressionTracker(regression_threshold=threshold)

    # Run the performance regression tests
    results = run_all_performance_regression_tests()

    # Get any new alerts that were generated
    alerts = get_regression_alerts()

    # Send alerts if any were detected
    if alerts:
        send_alert(alerts, webhook_url)

        if fail_on_regression:
            print("\nPerformance regression detected. Failing build.")
            return 1
        else:
            print(
                "\nPerformance regression detected but continuing (fail_on_regression=False)"
            )
    else:
        print("\nNo performance regressions detected.")

    return 0


def generate_ci_report():
    """Generate a CI-friendly report of performance test results."""
    tracker = PerformanceRegressionTracker()

    # Generate detailed report
    report_path = tracker.generate_report(output_dir="ci_reports")

    # Also generate a summary for CI logs
    alerts = get_regression_alerts()

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_alerts": len(alerts),
        "critical_alerts": len([a for a in alerts if a.severity.value == "critical"]),
        "warning_alerts": len([a for a in alerts if a.severity.value == "warning"]),
        "report_path": report_path,
    }

    # Print summary to stdout for CI
    print("\n" + "=" * 60)
    print("PERFORMANCE REGRESSION TEST SUMMARY")
    print("=" * 60)
    print(f"Total alerts: {summary['total_alerts']}")
    print(f"Critical alerts: {summary['critical_alerts']}")
    print(f"Warning alerts: {summary['warning_alerts']}")
    print(f"Detailed report: {summary['report_path']}")
    print("=" * 60)

    # Save summary as JSON for CI tools to parse
    with open("performance_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="CI/CD Performance Regression Testing")
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Regression detection threshold percentage (default: 5.0)",
    )
    parser.add_argument("--webhook-url", type=str, help="Webhook URL to send alerts to")
    parser.add_argument(
        "--no-fail-on-regression",
        action="store_true",
        help="Don't fail the build on regression detection",
    )
    parser.add_argument(
        "--generate-report", action="store_true", help="Generate CI report"
    )

    args = parser.parse_args()

    if args.generate_report:
        generate_ci_report()
        return 0

    exit_code = run_ci_performance_tests(
        threshold=args.threshold,
        webhook_url=args.webhook_url,
        fail_on_regression=not args.no_fail_on_regression,
    )

    # Generate report even if tests failed
    try:
        generate_ci_report()
    except Exception as e:
        print(f"Error generating report: {e}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
