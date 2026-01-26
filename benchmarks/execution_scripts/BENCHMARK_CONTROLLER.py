"""
Resource-Efficient Benchmark Execution Controller

This script manages the execution of all 7 benchmark categories for all 4 models
with proper resource management and real data collection. It implements strategies
to handle computational intensity appropriately.
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
from typing import Dict, List, Any


class BenchmarkController:
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
        self.script_mapping = {
            "accuracy": "run_accuracy_benchmarks.py",
            "comparison": "run_comparison_benchmarks.py",
            "inference_speed": "run_inference_speed_benchmarks.py",
            "memory_usage": "run_memory_usage_benchmarks.py",
            "optimization_impact": "run_optimization_impact_benchmarks.py",
            "power_efficiency": "run_power_efficiency_benchmarks.py",
            "throughput": "run_throughput_benchmarks.py"
        }

    def check_system_resources(self, min_memory_gb=4):
        """Check if system has sufficient resources."""
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        
        print(f"Available Memory: {available_memory_gb:.2f} GB")
        
        if available_memory_gb < min_memory_gb:
            print(f"WARNING: Low memory condition. Available: {available_memory_gb:.2f} GB, Recommended: {min_memory_gb} GB")
            response = input("Continue anyway? (y/n): ")
            return response.lower() == 'y'
        
        return True

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

    def run_single_benchmark_category(self, category: str, timeout_minutes: int = 60):
        """Run a single benchmark category for all models."""
        script_name = self.script_mapping[category]
        print(f"\n{'='*60}")
        print(f"RUNNING {category.upper()} BENCHMARKS FOR ALL MODELS")
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
            
            category_result = {
                "status": status,
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "start_time": start_time,
                "end_time": time.time()
            }
            
            print(f"{category.upper()} benchmarks completed with status: {status}")
            print(f"Execution time: {execution_time:.2f} seconds")
            
            return category_result
            
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"{category.upper()} benchmarks timed out after {timeout_minutes} minutes")
            return {
                "status": "timeout",
                "execution_time": execution_time,
                "error": f"Timed out after {timeout_minutes} minutes"
            }
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"{category.upper()} benchmarks failed with error: {str(e)}")
            return {
                "status": "error", 
                "execution_time": execution_time,
                "error": str(e)
            }

    def run_all_benchmarks_resource_efficient(self):
        """Run all benchmarks with resource efficiency in mind."""
        print("RESOURCE-EFFICIENT BENCHMARK EXECUTION")
        print("="*70)
        print("Running all 7 categories for all 4 models with resource management")
        print("="*70)
        
        # Check resources
        if not self.check_system_resources(min_memory_gb=6):
            print("Insufficient resources. Exiting.")
            return None
        
        print(f"\nModels: {self.models}")
        print(f"Categories: {self.categories}")
        print(f"Total Categories: {len(self.categories)}")
        print(f"Total Model-Category Pairs: {len(self.models) * len(self.categories)}")
        
        confirm = input("\nProceed with benchmark execution? (y/n): ")
        if confirm.lower() != 'y':
            print("Execution cancelled.")
            return None
        
        # Initialize results structure
        results = {
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
        
        print(f"\nStarting execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run each category separately to manage resources
        for i, category in enumerate(self.categories, 1):
            print(f"\n[{i}/{len(self.categories)}] Executing {category.upper()} benchmarks...")
            
            # Run this category for all models
            category_result = self.run_single_benchmark_category(category)
            
            # Store result
            results["category_results"][category] = category_result
            
            # Update summary
            if category_result["status"] == "success":
                results["summary"]["successful_categories"] += 1
            elif category_result["status"] == "failed":
                results["summary"]["failed_categories"] += 1
            elif category_result["status"] == "timeout":
                results["summary"]["timed_out_categories"] += 1
                
            results["summary"]["total_execution_time"] += category_result.get("execution_time", 0)
            
            # Cleanup resources between categories
            if i < len(self.categories):  # Don't cleanup after the last one
                self.cleanup_resources()
                print(f"Resources cleaned up. Waiting 5 seconds before next category...")
                time.sleep(5)
        
        # Finalize results
        results["execution_metadata"]["end_time"] = datetime.now().isoformat()
        total_duration = time.time() - datetime.fromisoformat(results["execution_metadata"]["start_time"]).timestamp()
        results["execution_metadata"]["total_duration_seconds"] = total_duration
        
        # Print summary
        self.print_execution_summary(results)
        
        # Save results
        self.save_results(results)
        
        return results

    def print_execution_summary(self, results: Dict[str, Any]):
        """Print a summary of the execution."""
        print("\n" + "="*70)
        print("EXECUTION SUMMARY")
        print("="*70)
        
        metadata = results["execution_metadata"]
        summary = results["summary"]
        
        print(f"Start Time: {metadata['start_time']}")
        print(f"End Time:   {metadata['end_time']}")
        print(f"Duration:   {metadata['total_duration_seconds']:.2f} seconds ({metadata['total_duration_seconds']/60:.2f} minutes)")
        print(f"Models:     {len(metadata['models'])}")
        print(f"Categories: {len(metadata['categories'])}")
        
        print(f"\nCategory Results:")
        print(f"  Successful: {summary['successful_categories']}")
        print(f"  Failed:     {summary['failed_categories']}")
        print(f"  Timed Out:  {summary['timed_out_categories']}")
        print(f"  Total:      {len(self.categories)}")
        
        success_rate = (summary['successful_categories'] / len(self.categories) * 100) if len(self.categories) > 0 else 0
        print(f"  Success Rate: {success_rate:.1f}%")
        
        print(f"\nDetailed Results:")
        for category, result in results["category_results"].items():
            status = result["status"].upper()
            exec_time = result.get("execution_time", 0)
            print(f"  {category:<20} {status:<10} {exec_time:>6.1f}s")

    def save_results(self, results: Dict[str, Any]):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = f"resource_efficient_benchmark_results_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {detailed_file}")
        
        # Save summary
        summary_file = f"resource_efficient_benchmark_results_summary_{timestamp}.json"
        summary_data = {
            "execution_summary": results["execution_metadata"],
            "category_summary": results["summary"],
            "detailed_results": {
                cat: {
                    "status": res["status"],
                    "execution_time": res.get("execution_time", 0),
                    "return_code": res.get("return_code")
                } 
                for cat, res in results["category_results"].items()
            }
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        print(f"Summary results saved to: {summary_file}")
        
        # Create report
        report_file = f"resource_efficient_benchmark_report_{timestamp}.txt"
        self.create_report(results, report_file)
        print(f"Report saved to: {report_file}")

    def create_report(self, results: Dict[str, Any], filename: str):
        """Create a human-readable report."""
        with open(filename, 'w') as f:
            f.write("INFERENC-PIO RESOURCE-EFFICIENT BENCHMARK REPORT\n")
            f.write("="*60 + "\n\n")
            
            metadata = results["execution_metadata"]
            summary = results["summary"]
            
            f.write(f"Execution Summary:\n")
            f.write(f"  Start Time: {metadata['start_time']}\n")
            f.write(f"  End Time:   {metadata['end_time']}\n")
            f.write(f"  Duration:   {metadata['total_duration_seconds']:.2f} seconds\n")
            f.write(f"  Models:     {len(metadata['models'])}\n")
            f.write(f"  Categories: {len(metadata['categories'])}\n\n")
            
            f.write(f"Success Metrics:\n")
            f.write(f"  Successful: {summary['successful_categories']}\n")
            f.write(f"  Failed:     {summary['failed_categories']}\n")
            f.write(f"  Timed Out:  {summary['timed_out_categories']}\n")
            f.write(f"  Success Rate: {(summary['successful_categories'] / len(self.categories) * 100):.1f}%\n\n")
            
            f.write(f"Detailed Results:\n")
            for category, result in results["category_results"].items():
                f.write(f"  {category.upper()}:\n")
                f.write(f"    Status: {result['status'].upper()}\n")
                f.write(f"    Execution Time: {result.get('execution_time', 0):.1f}s\n")
                if result["status"] != "success":
                    error = result.get("error", result.get("stderr", ""))[:200]
                    f.write(f"    Error: {error}...\n")
                f.write(f"\n")


def main():
    """Main function."""
    controller = BenchmarkController()
    
    print("Inference-PIO Resource-Efficient Benchmark Controller")
    print("This will run all 7 benchmark categories for all 4 models with proper resource management.")
    
    try:
        results = controller.run_all_benchmarks_resource_efficient()
        if results:
            print(f"\nBenchmark execution completed successfully!")
            print(f"Results saved with timestamp {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    except KeyboardInterrupt:
        print(f"\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()