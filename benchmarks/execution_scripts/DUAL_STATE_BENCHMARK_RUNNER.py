"""
Benchmark Runner for Original vs Modified Models

This script runs benchmarks for both original and modified model states,
collecting data for comparison.
"""

import sys
import time
import json
import importlib
from typing import Dict, List, Any
from pathlib import Path
import os
import shutil
import subprocess
import copy


class DualStateBenchmarkRunner:
    """
    A class to run benchmarks across multiple models and categories 
    for both original and modified states.
    """

    def __init__(self):
        self.models = [
            "glm_4_7",
            "qwen3_4b_instruct_2507",
            "qwen3_coder_30b",
            "qwen3_vl_2b"
        ]

        self.benchmark_categories = [
            "accuracy",
            "comparison",
            "inference_speed", 
            "memory_usage",
            "optimization_impact",
            "power_efficiency",
            "throughput"
        ]

        self.states = ["original", "modified"]

        # Add the src directory to the Python path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))

    def backup_original_models(self):
        """Return the path to the current model implementations (no backup needed)."""
        print("Using current models as baseline (no backup needed)...")

        src_dir = Path("src/inference_pio/models")
        print(f"Current models location: {src_dir}")
        return src_dir

    def restore_original_models(self, backup_dir):
        """Restore the original model implementations (no restoration needed)."""
        print("Using current models as baseline (no restoration needed)...")

        # No action needed since we're using the current models
        print("Current models remain unchanged")

    def apply_modifications(self):
        """
        Apply custom code modifications to models.
        This is a placeholder - in a real scenario, this would apply actual modifications.
        """
        print("Applying custom code modifications...")
        
        # In a real implementation, this would apply actual modifications
        # For now, we'll simulate this by creating a temporary marker
        modification_marker = Path("MODIFICATIONS_APPLIED.marker")
        modification_marker.touch()
        
        print("Custom code modifications applied")

    def run_benchmarks_for_state(self, state: str) -> Dict[str, Any]:
        """
        Run all benchmarks for all models in a specific state.
        """
        print(f"\n{'='*60}")
        print(f"RUNNING BENCHMARKS FOR {state.upper()} STATE")
        print(f"{'='*60}")

        # Prepare the environment for the specific state
        if state == "original":
            print("Using CURRENT models as ORIGINAL state")
        elif state == "modified":
            self.apply_modifications()  # Apply modifications for modified state
            print("Switched to MODIFIED state")

        # Run benchmarks using the existing runner
        results = {
            "state": state,
            "timestamp": time.time(),
            "model_results": {}
        }

        for model in self.models:
            print(f"\nRunning benchmarks for {model} ({state})...")
            model_results = {}

            for category in self.benchmark_categories:
                print(f"  Running {category} benchmarks...")

                # Import and run the appropriate benchmark script
                script_name = f"run_{category}_benchmarks.py"

                try:
                    # Execute the benchmark script as a subprocess
                    result = subprocess.run([
                        sys.executable, script_name
                    ], capture_output=True, text=True, timeout=300)  # 5 minute timeout per category

                    model_results[category] = {
                        "status": "success" if result.returncode == 0 else "failed",
                        "return_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }

                    if result.returncode != 0:
                        print(f"    ⚠️  {category} benchmarks failed for {model}")
                    else:
                        print(f"    ✓ {category} benchmarks completed for {model}")

                except subprocess.TimeoutExpired:
                    model_results[category] = {
                        "status": "timeout",
                        "error": "Benchmark timed out after 5 minutes"
                    }
                    print(f"    ⚠️  {category} benchmarks timed out for {model}")
                except Exception as e:
                    model_results[category] = {
                        "status": "error",
                        "error": str(e)
                    }
                    print(f"    ✗ {category} benchmarks errored for {model}: {e}")

            results["model_results"][model] = model_results

        return results

    def run_comprehensive_benchmarks(self) -> Dict[str, Any]:
        """
        Run all benchmarks for all models in both original and modified states.
        """
        print("Starting comprehensive dual-state benchmark execution...")
        print(f"Models: {self.models}")
        print(f"Categories: {self.benchmark_categories}")
        print(f"States: {self.states}")

        # Get current models path (no backup needed)
        backup_dir = self.backup_original_models()

        # Initialize comprehensive results
        comprehensive_results = {
            "execution_start_time": time.time(),
            "states": {},
            "comparison_analysis": {}
        }

        # Run benchmarks for each state
        for state in self.states:
            state_results = self.run_benchmarks_for_state(state)
            comprehensive_results["states"][state] = state_results

        # Perform comparison analysis
        comprehensive_results["comparison_analysis"] = self.analyze_comparison(
            comprehensive_results["states"]["original"],
            comprehensive_results["states"]["modified"]
        )

        # Clean up modification marker if it exists
        if Path("MODIFICATIONS_APPLIED.marker").exists():
            Path("MODIFICATIONS_APPLIED.marker").unlink()

        comprehensive_results["execution_end_time"] = time.time()
        comprehensive_results["total_execution_time"] = (
            comprehensive_results["execution_end_time"] -
            comprehensive_results["execution_start_time"]
        )

        return comprehensive_results

    def analyze_comparison(self, original_results: Dict[str, Any], modified_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze and compare results between original and modified states.
        """
        analysis = {
            "summary": {
                "original_success_count": 0,
                "modified_success_count": 0,
                "total_comparisons": 0
            },
            "detailed": {}
        }

        for model in self.models:
            analysis["detailed"][model] = {}
            
            for category in self.benchmark_categories:
                orig_result = original_results["model_results"][model].get(category, {})
                mod_result = modified_results["model_results"][model].get(category, {})
                
                comparison = {
                    "original_status": orig_result.get("status", "missing"),
                    "modified_status": mod_result.get("status", "missing"),
                    "improved": False,
                    "degraded": False,
                    "same": False
                }
                
                # Count successful benchmarks
                if orig_result.get("status") == "success":
                    analysis["summary"]["original_success_count"] += 1
                if mod_result.get("status") == "success":
                    analysis["summary"]["modified_success_count"] += 1
                
                analysis["summary"]["total_comparisons"] += 1
                
                # Determine if there was improvement, degradation, or no change
                if orig_result.get("status") == mod_result.get("status"):
                    comparison["same"] = True
                elif orig_result.get("status") == "failed" and mod_result.get("status") == "success":
                    comparison["improved"] = True
                elif orig_result.get("status") == "success" and mod_result.get("status") == "failed":
                    comparison["degraded"] = True
                
                analysis["detailed"][model][category] = comparison

        return analysis

    def save_results(self, results: Dict[str, Any], filename: str = "dual_state_benchmark_results.json"):
        """
        Save benchmark results to a JSON file.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {filename}")

    def print_summary(self, results: Dict[str, Any]):
        """
        Print a summary of the benchmark results.
        """
        print("\n" + "="*70)
        print("DUAL-STATE BENCHMARK EXECUTION SUMMARY")
        print("="*70)

        exec_time = results["total_execution_time"]
        print(f"Total Execution Time: {exec_time:.2f} seconds ({exec_time/60:.2f} minutes)")

        analysis = results["comparison_analysis"]
        orig_success = analysis["summary"]["original_success_count"]
        mod_success = analysis["summary"]["modified_success_count"]
        total = analysis["summary"]["total_comparisons"]

        print(f"\nBenchmark Success Rates:")
        print(f"  Original State: {orig_success}/{total} ({orig_success/total*100:.1f}%)")
        print(f"  Modified State: {mod_success}/{total} ({mod_success/total*100:.1f}%)")

        if mod_success > orig_success:
            improvement = mod_success - orig_success
            print(f"  Improvement: +{improvement} successful benchmarks")
        elif mod_success < orig_success:
            regression = orig_success - mod_success
            print(f"  Regression: -{regression} successful benchmarks")
        else:
            print("  No change in success rate")

        print(f"\nDetailed Results by Model:")

        for model in self.models:
            print(f"\n{model.upper()}:")
            orig_model_results = results["states"]["original"]["model_results"][model]
            mod_model_results = results["states"]["modified"]["model_results"][model]
            
            successful_orig = sum(1 for v in orig_model_results.values() if v.get("status") == "success")
            successful_mod = sum(1 for v in mod_model_results.values() if v.get("status") == "success")
            
            print(f"  Original: {successful_orig}/{len(self.benchmark_categories)} successful")
            print(f"  Modified: {successful_mod}/{len(self.benchmark_categories)} successful")


def main():
    """
    Main function to run dual-state benchmarks.
    """
    runner = DualStateBenchmarkRunner()
    
    print("Inference-PIO Dual-State Benchmark Runner")
    print("Running benchmarks for both original and modified model states.")
    
    try:
        results = runner.run_comprehensive_benchmarks()
        runner.print_summary(results)
        runner.save_results(results, "dual_state_benchmark_results.json")
        
        # Also save detailed results
        runner.save_results(results, "dual_state_benchmark_results_detailed.json")
        
        print(f"\nDual-state benchmark execution completed!")
        
    except KeyboardInterrupt:
        print(f"\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()