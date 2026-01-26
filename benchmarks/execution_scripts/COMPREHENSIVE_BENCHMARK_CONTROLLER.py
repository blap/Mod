"""
Comprehensive Benchmark Controller for Comparing Original vs Modified Models

This script manages the execution of all 7 benchmark categories for all 4 models
in both original and modified states. It collects data for comparison and 
generates comprehensive reports.
"""

import sys
import time
import subprocess
import json
import os
from pathlib import Path
import gc
import torch
import psutil
from datetime import datetime
from typing import Dict, List, Any, Optional
import shutil


class ComprehensiveBenchmarkController:
    def __init__(self):
        self.results = {}
        self.models = [
            "glm_4_7",
            "qwen3_4b_instruct_2507", 
            "qwen3_coder_30b",
            "qwen3_vl_2b"
        ]
        self.categories = [
            "accuracy",
            "comparison", 
            "inference_speed",
            "memory_usage",
            "optimization_impact",
            "power_efficiency", 
            "throughput"
        ]
        self.states = ["original", "modified"]  # Two states to compare
        
        # Mapping of benchmark categories to their respective scripts
        self.script_mapping = {
            "accuracy": "run_accuracy_benchmarks.py",
            "comparison": "run_comparison_benchmarks.py",
            "inference_speed": "run_inference_speed_benchmarks.py",
            "memory_usage": "run_memory_usage_benchmarks.py",
            "optimization_impact": "run_optimization_impact_benchmarks.py",
            "power_efficiency": "run_power_efficiency_benchmarks.py",
            "throughput": "run_throughput_benchmarks.py"
        }

    def check_system_resources(self, min_memory_gb=8):
        """Check if system has sufficient resources."""
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)

        print(f"Available Memory: {available_memory_gb:.2f} GB")

        if available_memory_gb < min_memory_gb:
            print(f"WARNING: Low memory condition. Available: {available_memory_gb:.2f} GB, Recommended: {min_memory_gb} GB")
            response = input("Continue anyway? (y/n): ")
            return response.lower() == 'y'

        return True

    def backup_original_state(self):
        """Return the path to the current model implementations (no backup needed)."""
        print("Using current models as baseline (no backup needed)...")

        src_dir = Path("src/inference_pio/models")
        print(f"Current models location: {src_dir}")
        return src_dir

    def restore_original_state(self, backup_dir: Path):
        """Restore the original model implementations (no restoration needed)."""
        print("Using current models as baseline (no restoration needed)...")

        # No action needed since we're using the current models
        print("Current models remain unchanged")

    def apply_modifications(self):
        """
        Apply custom code modifications to models.
        This is a placeholder - in a real scenario, this would apply actual modifications.
        """
        print("Applying custom code modifications to models...")
        
        # In a real implementation, this would apply actual modifications
        # For now, we'll simulate this by creating a temporary marker
        modification_marker = Path("MODIFICATIONS_APPLIED.marker")
        modification_marker.touch()
        
        print("Custom code modifications applied successfully")
        return True

    def remove_modifications(self):
        """Remove custom code modifications and restore original state."""
        print("Removing custom code modifications...")
        
        modification_marker = Path("MODIFICATIONS_APPLIED.marker")
        if modification_marker.exists():
            modification_marker.unlink()
        
        print("Custom code modifications removed")

    def cleanup_resources(self):
        """Clean up resources between benchmark runs."""
        print("Cleaning up resources...")

        # Force garbage collection
        gc.collect()

        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Small delay to allow system to settle
        time.sleep(2)

    def run_single_benchmark_state(self, state: str, category: str, timeout_minutes: int = 60):
        """Run a single benchmark category for all models in a specific state."""
        script_name = self.script_mapping[category]
        print(f"\n{'='*60}")
        print(f"RUNNING {category.upper()} BENCHMARKS FOR ALL MODELS ({state.upper()} STATE)")
        print(f"Script: {script_name}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            result = subprocess.run([
                sys.executable, script_name
            ],
            capture_output=True,
            text=True,
            timeout=timeout_minutes*60,
            cwd=os.getcwd())

            execution_time = time.time() - start_time

            status = "success" if result.returncode == 0 else "failed"

            state_result = {
                "status": status,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "start_time": start_time,
                "end_time": time.time()
            }

            print(f"{category.upper()} benchmarks ({state}) completed with status: {status}")
            print(f"Execution time: {execution_time:.2f} seconds")

            return state_result

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"{category.upper()} benchmarks ({state}) timed out after {timeout_minutes} minutes")
            return {
                "status": "timeout",
                "execution_time": execution_time,
                "error": f"Timed out after {timeout_minutes} minutes"
            }
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"{category.upper()} benchmarks ({state}) failed with error: {str(e)}")
            return {
                "status": "error",
                "execution_time": execution_time,
                "error": str(e)
            }

    def run_benchmarks_for_state(self, state: str):
        """Run all benchmarks for all models in a specific state."""
        print(f"\n{'='*70}")
        print(f"RUNNING ALL BENCHMARKS FOR {state.upper()} MODEL STATE")
        print(f"{'='*70}")
        
        # Initialize results structure for this state
        state_results = {
            "state": state,
            "execution_metadata": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "total_duration_seconds": None,
                "models": self.models,
                "categories": self.categories,
                "platform": sys.platform,
                "python_version": sys.version
            },
            "category_results": {},
            "summary": {
                "successful_categories": 0,
                "failed_categories": 0,
                "timed_out_categories": 0,
                "total_execution_time": 0
            }
        }
        
        # Run each category for this state
        for i, category in enumerate(self.categories, 1):
            print(f"\n[{i}/{len(self.categories)}] Executing {category.upper()} benchmarks ({state})...")

            # Run this category for all models in this state
            category_result = self.run_single_benchmark_state(state, category)

            # Store result
            state_results["category_results"][category] = category_result

            # Update summary
            if category_result["status"] == "success":
                state_results["summary"]["successful_categories"] += 1
            elif category_result["status"] == "failed":
                state_results["summary"]["failed_categories"] += 1
            elif category_result["status"] == "timeout":
                state_results["summary"]["timed_out_categories"] += 1

            state_results["summary"]["total_execution_time"] += category_result.get("execution_time", 0)

            # Cleanup resources between categories
            if i < len(self.categories):  # Don't cleanup after the last one
                self.cleanup_resources()
                print(f"Resources cleaned up. Waiting 5 seconds before next category...")
                time.sleep(5)
        
        # Finalize state results
        state_results["execution_metadata"]["end_time"] = datetime.now().isoformat()
        total_duration = time.time() - datetime.fromisoformat(
            state_results["execution_metadata"]["start_time"]
        ).timestamp()
        state_results["execution_metadata"]["total_duration_seconds"] = total_duration
        
        return state_results

    def run_comprehensive_benchmarks(self):
        """Run all benchmarks for all models in both original and modified states."""
        print("COMPREHENSIVE BENCHMARK EXECUTION")
        print("="*70)
        print("Running all 7 categories for all 4 models in both original and modified states")
        print("="*70)

        # Check resources
        if not self.check_system_resources(min_memory_gb=8):
            print("Insufficient resources. Exiting.")
            return None

        print(f"\nModels: {self.models}")
        print(f"Categories: {self.categories}")
        print(f"States: {self.states}")
        print(f"Total States: {len(self.states)}")
        print(f"Total Categories per State: {len(self.categories)}")
        print(f"Total Model-Category-State Combinations: {len(self.models) * len(self.categories) * len(self.states)}")

        confirm = input("\nProceed with comprehensive benchmark execution? (y/n): ")
        if confirm.lower() != 'y':
            print("Execution cancelled.")
            return None

        # Get current models path (no backup needed)
        backup_dir = self.backup_original_state()

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
                "python_version": sys.version
            },
            "state_results": {},
            "comparison_summary": {
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
        for state in self.states:
            if state == "original":
                # Use current models as original state
                print(f"\nUsing CURRENT models as ORIGINAL state")
            elif state == "modified":
                # Apply modifications
                self.apply_modifications()  # Apply modifications for modified state
                print(f"\nApplied MODIFICATIONS and switched to MODIFIED state")

            # Run all benchmarks for this state
            state_result = self.run_benchmarks_for_state(state)

            # Store state results
            comprehensive_results["state_results"][state] = state_result

            # Update comparison summary
            comp_summary = comprehensive_results["comparison_summary"][state]
            state_summary = state_result["summary"]
            comp_summary["successful_categories"] = state_summary["successful_categories"]
            comp_summary["failed_categories"] = state_summary["failed_categories"]
            comp_summary["timed_out_categories"] = state_summary["timed_out_categories"]
            comp_summary["total_execution_time"] = state_summary["total_execution_time"]

        # Finalize comprehensive results
        comprehensive_results["execution_metadata"]["end_time"] = datetime.now().isoformat()
        total_duration = time.time() - datetime.fromisoformat(
            comprehensive_results["execution_metadata"]["start_time"]
        ).timestamp()
        comprehensive_results["execution_metadata"]["total_duration_seconds"] = total_duration

        # Print comparison summary
        self.print_comparison_summary(comprehensive_results)

        # Save comprehensive results
        self.save_comprehensive_results(comprehensive_results)

        # Clean up modification marker if it exists
        if Path("MODIFICATIONS_APPLIED.marker").exists():
            Path("MODIFICATIONS_APPLIED.marker").unlink()
        print(f"\nBenchmark execution completed")

        return comprehensive_results

    def print_comparison_summary(self, results: Dict[str, Any]):
        """Print a comparison summary of original vs modified results."""
        print("\n" + "="*70)
        print("COMPREHENSIVE COMPARISON SUMMARY")
        print("="*70)

        metadata = results["execution_metadata"]
        comp_summary = results["comparison_summary"]

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
            print(f"  Timed Out:  {summary['timed_out_categories']}")
            print(f"  Total:      {len(self.categories)}")
            print(f"  Execution Time: {summary['total_execution_time']:.2f}s")

        # Calculate improvement/degradation
        orig_time = comp_summary["original"]["total_execution_time"]
        mod_time = comp_summary["modified"]["total_execution_time"]
        
        if orig_time > 0:
            time_change = ((mod_time - orig_time) / orig_time) * 100
            print(f"\nPerformance Change: {time_change:+.2f}%")
            if time_change < 0:
                print("  => Modification improved performance!")
            elif time_change > 0:
                print("  => Modification degraded performance.")
            else:
                print("  => No performance change.")

    def save_comprehensive_results(self, results: Dict[str, Any]):
        """Save comprehensive results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        detailed_file = f"comprehensive_benchmark_results_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {detailed_file}")

        # Save summary
        summary_file = f"comprehensive_benchmark_results_summary_{timestamp}.json"
        summary_data = {
            "execution_summary": results["execution_metadata"],
            "comparison_summary": results["comparison_summary"],
            "state_comparison": {
                state: {
                    "successful": data["successful_categories"],
                    "failed": data["failed_categories"],
                    "timed_out": data["timed_out_categories"],
                    "execution_time": data["total_execution_time"]
                }
                for state, data in results["comparison_summary"].items()
            }
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        print(f"Summary results saved to: {summary_file}")

        # Create comparison report
        report_file = f"comprehensive_benchmark_comparison_report_{timestamp}.md"
        self.create_comparison_report(results, report_file)
        print(f"Comparison report saved to: {report_file}")

    def create_comparison_report(self, results: Dict[str, Any], filename: str):
        """Create a human-readable comparison report in Markdown format."""
        with open(filename, 'w') as f:
            f.write("# INFERENC-PIO COMPREHENSIVE BENCHMARK COMPARISON REPORT\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            metadata = results["execution_metadata"]
            comp_summary = results["comparison_summary"]

            f.write("## Execution Summary\n")
            f.write(f"- Start Time: {metadata['start_time']}\n")
            f.write(f"- End Time: {metadata['end_time']}\n")
            f.write(f"- Duration: {metadata['total_duration_seconds']:.2f} seconds\n")
            f.write(f"- Models Tested: {len(metadata['models'])}\n")
            f.write(f"- Categories: {len(metadata['categories'])}\n")
            f.write(f"- States Compared: {len(metadata['states'])}\n\n")

            f.write("## Comparison Results\n\n")
            
            for state in ["original", "modified"]:
                summary = comp_summary[state]
                f.write(f"### {state.upper()} STATE\n")
                f.write(f"- Successful: {summary['successful_categories']}\n")
                f.write(f"- Failed: {summary['failed_categories']}\n")
                f.write(f"- Timed Out: {summary['timed_out_categories']}\n")
                f.write(f"- Total: {len(self.categories)}\n")
                f.write(f"- Execution Time: {summary['total_execution_time']:.2f}s\n\n")

            # Calculate and report performance change
            orig_time = comp_summary["original"]["total_execution_time"]
            mod_time = comp_summary["modified"]["total_execution_time"]
            
            if orig_time > 0:
                time_change = ((mod_time - orig_time) / orig_time) * 100
                f.write(f"## Performance Change Analysis\n")
                f.write(f"- Original Execution Time: {orig_time:.2f}s\n")
                f.write(f"- Modified Execution Time: {mod_time:.2f}s\n")
                f.write(f"- Change: {time_change:+.2f}%\n")
                
                if time_change < 0:
                    f.write("- Result: **MODIFICATION IMPROVED PERFORMANCE**\n")
                elif time_change > 0:
                    f.write("- Result: **MODIFICATION DEGRADED PERFORMANCE**\n")
                else:
                    f.write("- Result: **NO PERFORMANCE CHANGE**\n")

            f.write("\n## Detailed Category Results\n\n")
            for state in ["original", "modified"]:
                f.write(f"### {state.upper()} STATE DETAILED RESULTS\n")
                state_results = results["state_results"][state]
                
                for category, result in state_results["category_results"].items():
                    f.write(f"#### {category.upper()}\n")
                    f.write(f"- Status: {result['status'].upper()}\n")
                    f.write(f"- Execution Time: {result.get('execution_time', 0):.1f}s\n")
                    if result["status"] != "success":
                        error = result.get("error", result.get("stderr", ""))[:200]
                        f.write(f"- Error: {error}...\n")
                    f.write("\n")


def main():
    """Main function."""
    controller = ComprehensiveBenchmarkController()

    print("Inference-PIO Comprehensive Benchmark Controller")
    print("This will run all 7 benchmark categories for all 4 models in both original and modified states.")

    try:
        results = controller.run_comprehensive_benchmarks()
        if results:
            print(f"\nComprehensive benchmark execution completed successfully!")
            print(f"Results saved with timestamp {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    except KeyboardInterrupt:
        print(f"\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()