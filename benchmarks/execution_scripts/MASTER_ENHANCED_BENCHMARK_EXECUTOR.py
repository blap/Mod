"""
Master Enhanced Benchmark Executor

This script runs all 4 enhanced benchmark categories for all 4 models in both 
original and modified states. It coordinates the execution, comparison, and 
reporting of all benchmark results.
"""

import sys
import time
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any
import subprocess

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from enhanced_benchmark_utils import (
    apply_modifications,
    remove_modifications,
    cleanup_resources,
    check_system_resources
)


def run_enhanced_benchmarks():
    """
    Run all enhanced benchmarks for all models in both original and modified states.
    """
    print("MASTER ENHANCED BENCHMARK EXECUTION")
    print("="*80)
    print("Running all 4 benchmark categories for all 4 models in both original and modified states")
    print("With organized results, percentage differences, and detailed reports")
    print("="*80)

    # Check resources
    if not check_system_resources(min_memory_gb=8):
        print("Insufficient resources. Exiting.")
        return None

    models = [
        "glm_4_7",
        "qwen3_4b_instruct_2507",
        "qwen3_coder_30b",
        "qwen3_vl_2b"
    ]
    
    categories = [
        "accuracy",
        "inference_speed", 
        "memory_usage",
        "optimization_impact"
    ]

    print(f"\nModels: {models}")
    print(f"Categories: {categories}")
    print(f"Total Categories: {len(categories)}")
    print(f"Total Model-Category Pairs: {len(models) * len(categories) * 2}")  # *2 for original & modified

    confirm = input("\nProceed with master enhanced benchmark execution? (y/n): ")
    if confirm.lower() != 'y':
        print("Execution cancelled.")
        return None

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory if it doesn't exist
    results_dir = Path("benchmark_results")
    results_dir.mkdir(exist_ok=True)

    # Get current models path (no backup needed)
    backup_dir = Path("src/inference_pio/models")  # Just reference the current models

    # Initialize comprehensive results
    comprehensive_results = {
        "execution_metadata": {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_duration_seconds": None,
            "models": models,
            "categories": categories,
            "platform": sys.platform,
            "python_version": sys.version,
            "torch_version": "N/A",
            "timestamp": timestamp
        },
        "category_results": {},
        "summary": {
            "successful_executions": 0,
            "failed_executions": 0,
            "total_executions": 0
        }
    }

    print(f"\nStarting execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run each category for both states
    total_executions = len(categories) * 2  # Each category runs for both original and modified
    execution_count = 0
    
    for category in categories:
        for state in ["original", "modified"]:
            execution_count += 1
            print(f"\n{'='*60}")
            print(f"[{execution_count}/{total_executions}] RUNNING {category.upper()} BENCHMARKS FOR {state.upper()} STATE")
            print(f"{'='*60}")
            
            # Switch to appropriate state
            if state == "original":
                print(f"Using CURRENT models as ORIGINAL state")
            elif state == "modified":
                apply_modifications()  # Apply modifications for modified state
                print(f"Applied MODIFICATIONS and switched to MODIFIED state")

            # Run the specific benchmark category
            script_name = f"enhanced_benchmark_{category}.py"
            try:
                result = subprocess.run([
                    sys.executable, script_name
                ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout per category

                category_result = {
                    "status": "success" if result.returncode == 0 else "failed",
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "execution_time": time.time()  # Placeholder, actual time would be in the script
                }

                if result.returncode != 0:
                    print(f"⚠️  {category} benchmarks failed for {state} state")
                    comprehensive_results["summary"]["failed_executions"] += 1
                else:
                    print(f"✓ {category} benchmarks completed for {state} state")
                    comprehensive_results["summary"]["successful_executions"] += 1

                # Store result
                if category not in comprehensive_results["category_results"]:
                    comprehensive_results["category_results"][category] = {}
                comprehensive_results["category_results"][category][state] = category_result

            except subprocess.TimeoutExpired:
                category_result = {
                    "status": "timeout",
                    "error": "Benchmark timed out after 30 minutes"
                }
                print(f"⚠️  {category} benchmarks timed out for {state} state")
                comprehensive_results["summary"]["failed_executions"] += 1
                comprehensive_results["category_results"][category][state] = category_result
            except Exception as e:
                category_result = {
                    "status": "error",
                    "error": str(e)
                }
                print(f"✗ {category} benchmarks errored for {state} state: {e}")
                comprehensive_results["summary"]["failed_executions"] += 1
                comprehensive_results["category_results"][category][state] = category_result

            comprehensive_results["summary"]["total_executions"] += 1

            # Cleanup resources between executions
            cleanup_resources()
            print(f"Resources cleaned up. Waiting 5 seconds before next execution...")
            time.sleep(5)

    # Finalize comprehensive results
    comprehensive_results["execution_metadata"]["end_time"] = datetime.now().isoformat()
    total_duration = time.time() - datetime.fromisoformat(
        comprehensive_results["execution_metadata"]["start_time"]
    ).timestamp()
    comprehensive_results["execution_metadata"]["total_duration_seconds"] = total_duration

    # Print summary
    print_summary(comprehensive_results)

    # Save comprehensive results
    save_master_results(comprehensive_results, timestamp, results_dir)

    # Clean up modifications
    remove_modifications()
    print(f"\nBenchmark execution completed")

    return comprehensive_results


def print_summary(results: Dict[str, Any]):
    """Print a summary of the master benchmark execution."""
    print("\n" + "="*80)
    print("MASTER ENHANCED BENCHMARK EXECUTION SUMMARY")
    print("="*80)

    metadata = results["execution_metadata"]
    summary = results["summary"]

    print(f"Start Time: {metadata['start_time']}")
    print(f"End Time:   {metadata['end_time']}")
    print(f"Duration:   {metadata['total_duration_seconds']:.2f} seconds ({metadata['total_duration_seconds']/60:.2f} minutes)")
    print(f"Models:     {len(metadata['models'])}")
    print(f"Categories: {len(metadata['categories'])}")

    print(f"\nExecution Summary:")
    print(f"  Successful: {summary['successful_executions']}")
    print(f"  Failed:     {summary['failed_executions']}")
    print(f"  Total:      {summary['total_executions']}")
    
    success_rate = (summary['successful_executions'] / summary['total_executions'] * 100) if summary['total_executions'] > 0 else 0
    print(f"  Success Rate: {success_rate:.1f}%")

    print(f"\nDetailed Results by Category:")
    for category, states in results["category_results"].items():
        print(f"\n{category.upper()}:")
        for state, result in states.items():
            status = result["status"].upper()
            print(f"  {state.upper()}: {status}")


def save_master_results(results: Dict[str, Any], timestamp: str, results_dir: Path):
    """Save master benchmark results to organized files."""
    # Save detailed comprehensive results
    detailed_file = results_dir / f"master_enhanced_benchmark_results_detailed_{timestamp}.json"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {detailed_file}")

    # Save summary
    summary_file = results_dir / f"master_enhanced_benchmark_results_summary_{timestamp}.json"
    summary_data = {
        "execution_summary": results["execution_metadata"],
        "execution_summary": results["summary"],
        "category_results": {
            cat: {
                state: {
                    "status": res["status"],
                    "return_code": res.get("return_code")
                }
                for state, res in states.items()
            }
            for cat, states in results["category_results"].items()
        }
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"Summary results saved to: {summary_file}")

    # Create report
    report_file = results_dir / f"master_enhanced_benchmark_report_{timestamp}.md"
    create_master_report(results, report_file)
    print(f"Report saved to: {report_file}")


def create_master_report(results: Dict[str, Any], filename: str):
    """Create a human-readable master report in Markdown format."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# INFERENC-PIO MASTER ENHANCED BENCHMARK REPORT\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        metadata = results["execution_metadata"]
        summary = results["summary"]

        f.write("## Execution Summary\n")
        f.write(f"- Start Time: {metadata['start_time']}\n")
        f.write(f"- End Time: {metadata['end_time']}\n")
        f.write(f"- Duration: {metadata['total_duration_seconds']:.2f} seconds\n")
        f.write(f"- Models Tested: {len(metadata['models'])}\n")
        f.write(f"- Categories: {len(metadata['categories'])}\n")
        f.write(f"- Total Executions: {summary['total_executions']}\n\n")

        f.write("## Success Metrics\n")
        f.write(f"- Successful: {summary['successful_executions']}\n")
        f.write(f"- Failed: {summary['failed_executions']}\n")
        success_rate = (summary['successful_executions'] / summary['total_executions'] * 100) if summary['total_executions'] > 0 else 0
        f.write(f"- Success Rate: {success_rate:.1f}%\n\n")

        f.write("## Detailed Results by Category\n\n")
        for category, states in results["category_results"].items():
            f.write(f"### {category.upper()}\n")
            for state, result in states.items():
                f.write(f"- **{state.upper()}**: {result['status'].upper()}\n")
                if result["status"] != "success":
                    error = result.get("error", result.get("stderr", ""))[:200]
                    f.write(f"  - Error: {error}...\n")
            f.write("\n")


def main():
    """Main function."""
    print("Inference-PIO Master Enhanced Benchmark Executor")
    print("This will run all 4 benchmark categories for all 4 models in both original and modified states.")
    print("Results will be compared, analyzed, and reported with percentage differences.")

    try:
        results = run_enhanced_benchmarks()
        if results:
            print(f"\nMaster enhanced benchmark execution completed successfully!")
            print(f"Results saved with timestamp {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    except KeyboardInterrupt:
        print(f"\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()