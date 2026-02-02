"""
Benchmark Results Handler Module

This module provides the logic for saving and manipulating benchmark results.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class BenchmarkResultsHandler:
    """
    A class to handle saving and manipulation of benchmark results.
    """

    def save_results(
        self, results: Dict[str, Any], output_dir: str = "benchmark_results"
    ) -> None:
        """
        Save benchmark results to JSON and CSV files.

        Args:
            results: Results dictionary to save
            output_dir: Directory to save results to
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        json_filename = output_path / f"benchmark_results_{timestamp}.json"
        with open(json_filename, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Save CSV summary
        csv_filename = output_path / f"benchmark_summary_{timestamp}.csv"
        with open(csv_filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                [
                    "Model",
                    "Total Benchmarks",
                    "Successful Runs",
                    "Failed Runs",
                    "Success Rate",
                    "Timestamp",
                ]
            )

            # Write summary row
            summary = results.get("summary", {})
            writer.writerow(
                [
                    results.get("model", "all"),
                    summary.get("total_benchmarks", 0),
                    summary.get("successful_runs", 0),
                    summary.get("failed_runs", 0),
                    f"{summary.get('success_rate', 0):.2%}",
                    results.get("timestamp", ""),
                ]
            )

        print(f"\nResults saved to:")
        print(f"  JSON: {json_filename}")
        print(f"  CSV: {csv_filename}")
