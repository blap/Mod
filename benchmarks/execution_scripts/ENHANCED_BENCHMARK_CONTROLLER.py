"""
Enhanced Comprehensive Benchmark Controller for Comparing Original vs Modified Models

This script manages the execution of all 7 benchmark categories for all 4 models
in both original and modified states. It collects data for comparison and
generates comprehensive reports with percentage differences and detailed analysis.
"""

import sys
import time
import json
import os
from pathlib import Path
import shutil
from datetime import datetime
from typing import Dict, List, Any
import gc
import torch
import psutil

# Import the enhanced utilities
from enhanced_benchmark_utils import (
    run_all_benchmarks_for_state,
    compare_results,
    save_results_json,
    save_results_csv,
    generate_detailed_report,
    backup_original_models,
    restore_original_models,
    apply_modifications,
    remove_modifications,
    cleanup_resources,
    check_system_resources
)


class EnhancedBenchmarkController:
    def __init__(self):
        self.models = [
            "glm_4_7",
            "qwen3_4b_instruct_2507",
            "qwen3_coder_30b",
            "qwen3_vl_2b"
        ]
        self.categories = [
            "accuracy",
            "inference_speed", 
            "memory_usage",
            "optimization_impact"
        ]
        self.states = ["original", "modified"]

    def run_enhanced_benchmarks(self):
        """Run enhanced benchmarks for both original and modified states."""
        print("ENHANCED COMPREHENSIVE BENCHMARK EXECUTION")
        print("="*80)
        print("Running all 4 categories for all 4 models in both original and modified states")
        print("With organized results, percentage differences, and detailed reports")
        print("="*80)

        # Check resources
        if not check_system_resources(min_memory_gb=8):
            print("Insufficient resources. Exiting.")
            return None

        print(f"\nModels: {self.models}")
        print(f"Categories: {self.categories}")
        print(f"States: {self.states}")
        print(f"Total States: {len(self.states)}")
        print(f"Total Categories per State: {len(self.categories)}")
        print(f"Total Model-Category-State Combinations: {len(self.models) * len(self.categories) * len(self.states)}")

        confirm = input("\nProceed with enhanced benchmark execution? (y/n): ")
        if confirm.lower() != 'y':
            print("Execution cancelled.")
            return None

        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory if it doesn't exist
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)

        # Backup original state
        backup_dir = backup_original_models(self.models)

        # Initialize comprehensive results
        comprehensive_results = {
            "execution_metadata": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "total_duration_seconds": None,
                "models": self.models,
                "categories": self.categories,
                "states": self.states,
                "platform": sys.platform,
                "python_version": sys.version,
                "torch_version": torch.__version__ if 'torch' in sys.modules else "N/A",
                "timestamp": timestamp
            },
            "state_results": {},
            "comparison_results": {},
            "summary": {
                "original": {
                    "successful_categories": 0,
                    "failed_categories": 0,
                    "timed_out_categories": 0,
                    "total_execution_time": 0
                },
                "modified": {
                    "successful_categories": 0,
                    "failed_categories": 0,
                    "timed_out_categories": 0,
                    "total_execution_time": 0
                }
            }
        }

        print(f"\nStarting execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Run benchmarks for each state
        for state_idx, state in enumerate(self.states):
            print(f"\n{'='*60}")
            print(f"PROCESSING {state.upper()} STATE")
            print(f"{'='*60}")
            
            if state == "original":
                # Ensure original state is active
                restore_original_models(backup_dir)
                print(f"Switched to ORIGINAL state")
            elif state == "modified":
                # Apply modifications
                restore_original_models(backup_dir)  # First restore original
                apply_modifications()  # Then apply modifications
                print(f"Applied MODIFICATIONS and switched to MODIFIED state")

            # Run all benchmarks for this state
            state_result = run_all_benchmarks_for_state(self.models, self.categories, state)

            # Store state results
            comprehensive_results["state_results"][state] = state_result

            # Update summary
            comp_summary = comprehensive_results["summary"][state]
            state_summary = state_result["summary"]
            comp_summary["successful_categories"] = sum(
                1 for model_results in state_result["details"].values()
                for cat_result in model_results.values()
                if cat_result.get("status") in ["success", "partial"]
            )
            comp_summary["failed_categories"] = sum(
                1 for model_results in state_result["details"].values()
                for cat_result in model_results.values()
                if cat_result.get("status") == "failed"
            )
            comp_summary["total_execution_time"] = state_summary["duration"]

            # Save intermediate results for this state
            state_filename = results_dir / f"benchmark_results_{state}_{timestamp}.json"
            save_results_json(state_result, str(state_filename))
            
            # Also save as CSV
            csv_filename = results_dir / f"benchmark_results_{state}_{timestamp}.csv"
            save_results_csv(state_result, str(csv_filename))

            # Cleanup resources between states (but not after the last one)
            if state_idx < len(self.states) - 1:
                cleanup_resources()
                print(f"Resources cleaned up. Waiting 5 seconds before next state...")
                time.sleep(5)

        # Perform comparison between original and modified results
        print(f"\n{'='*60}")
        print("PERFORMING COMPARISON ANALYSIS")
        print(f"{'='*60}")
        
        original_results = comprehensive_results["state_results"]["original"]
        modified_results = comprehensive_results["state_results"]["modified"]
        
        comparison_result = compare_results(original_results, modified_results)
        comprehensive_results["comparison_results"] = comparison_result

        # Finalize comprehensive results
        comprehensive_results["execution_metadata"]["end_time"] = datetime.now().isoformat()
        total_duration = time.time() - datetime.fromisoformat(
            comprehensive_results["execution_metadata"]["start_time"]
        ).timestamp()
        comprehensive_results["execution_metadata"]["total_duration_seconds"] = total_duration

        # Print comparison summary
        self.print_enhanced_summary(comprehensive_results)

        # Save comprehensive results
        self.save_enhanced_results(comprehensive_results, timestamp, results_dir)

        # Generate detailed reports
        self.generate_enhanced_reports(comprehensive_results, timestamp, results_dir)

        # Restore original state
        restore_original_models(backup_dir)
        remove_modifications()
        print(f"\nRestored original model state after benchmarking")

        return comprehensive_results

    def print_enhanced_summary(self, results: Dict[str, Any]):
        """Print an enhanced summary of the benchmark results."""
        print("\n" + "="*80)
        print("ENHANCED COMPREHENSIVE BENCHMARK SUMMARY")
        print("="*80)

        metadata = results["execution_metadata"]
        comp_summary = results["summary"]

        print(f"Start Time: {metadata['start_time']}")
        print(f"End Time:   {metadata['end_time']}")
        print(f"Duration:   {metadata['total_duration_seconds']:.2f} seconds ({metadata['total_duration_seconds']/60:.2f} minutes)")
        print(f"Models:     {len(metadata['models'])}")
        print(f"Categories: {len(metadata['categories'])}")
        print(f"States:     {len(metadata['states'])}")

        print(f"\nComparison Results:")
        for state in ["original", "modified"]:
            summary = comp_summary[state]
            print(f"\n{state.upper()} STATE:")
            print(f"  Successful: {summary['successful_categories']}")
            print(f"  Failed:     {summary['failed_categories']}")
            print(f"  Total:      {len(self.models) * len(self.categories)}")
            print(f"  Execution Time: {summary['total_execution_time']:.2f}s")

        # Calculate improvement/degradation
        orig_time = comp_summary["original"]["total_execution_time"]
        mod_time = comp_summary["modified"]["total_execution_time"]

        if orig_time > 0:
            time_change = ((mod_time - orig_time) / orig_time) * 100
            print(f"\nOverall Performance Change: {time_change:+.2f}%")
            if time_change < 0:
                print("  => Modification improved overall performance!")
            elif time_change > 0:
                print("  => Modification degraded overall performance.")
            else:
                print("  => No overall performance change.")

        # Show detailed comparison from the comparison results
        print(f"\nDetailed Category-by-Category Comparison:")
        comparison_data = results["comparison_results"]
        for model_name, model_comp in comparison_data["model_comparisons"].items():
            print(f"\n{model_name.upper()}:")
            for category_name, cat_comp in model_comp["category_comparisons"].items():
                dur_impr = cat_comp["metrics_comparison"].get("duration_improvement_pct", 0)
                print(f"  {category_name.upper()}: {dur_impr:+.2f}%")

    def save_enhanced_results(self, results: Dict[str, Any], timestamp: str, results_dir: Path):
        """Save enhanced results to organized files."""
        # Save detailed comprehensive results
        detailed_file = results_dir / f"comprehensive_benchmark_results_detailed_{timestamp}.json"
        save_results_json(results, str(detailed_file))

        # Save summary
        summary_file = results_dir / f"comprehensive_benchmark_results_summary_{timestamp}.json"
        summary_data = {
            "execution_summary": results["execution_metadata"],
            "comparison_summary": results["summary"],
            "state_comparison": {
                state: {
                    "successful": data["successful_categories"],
                    "failed": data["failed_categories"],
                    "execution_time": data["total_execution_time"]
                }
                for state, data in results["summary"].items()
            }
        }
        save_results_json(summary_data, str(summary_file))

        # Save comparison results separately
        comparison_file = results_dir / f"benchmark_comparison_results_{timestamp}.json"
        comparison_data = results["comparison_results"]
        save_results_json(comparison_data, str(comparison_file))

        print(f"\nEnhanced results saved to {results_dir}/ directory")

    def generate_enhanced_reports(self, results: Dict[str, Any], timestamp: str, results_dir: Path):
        """Generate enhanced reports in various formats."""
        # Create detailed comparison report
        comparison_report_file = results_dir / f"benchmark_comparison_report_{timestamp}.md"
        generate_detailed_report(results["comparison_results"], str(comparison_report_file))

        # Create summary report
        summary_report_file = results_dir / f"benchmark_summary_report_{timestamp}.md"
        generate_detailed_report(results, str(summary_report_file))

        # Create CSV report for easy analysis
        csv_comparison_file = results_dir / f"benchmark_comparison_results_{timestamp}.csv"
        save_results_csv(results["comparison_results"], str(csv_comparison_file))

        print(f"Enhanced reports generated in {results_dir}/ directory")


def main():
    """Main function."""
    controller = EnhancedBenchmarkController()

    print("Inference-PIO Enhanced Benchmark Controller")
    print("This will run all 4 benchmark categories for all 4 models in both original and modified states.")
    print("Results will be compared, analyzed, and reported with percentage differences.")

    try:
        results = controller.run_enhanced_benchmarks()
        if results:
            print(f"\nEnhanced benchmark execution completed successfully!")
            print(f"Results saved with timestamp {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    except KeyboardInterrupt:
        print(f"\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()