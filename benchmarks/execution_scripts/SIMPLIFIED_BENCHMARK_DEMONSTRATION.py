"""
SIMPLIFIED BENCHMARK EXECUTION DEMONSTRATION

This script demonstrates the execution of benchmark categories for models with real data collection.
Due to computational intensity, this version runs a focused subset to show the implementation.
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


class SimplifiedBenchmarkExecutor:
    def __init__(self):
        self.models = [
            "glm_4_7",
            "qwen3_4b_instruct_2507"
        ]
        self.categories = [
            "accuracy",
            "inference_speed",
            "memory_usage"
        ]
        self.script_mapping = {
            "accuracy": "run_accuracy_benchmarks.py",
            "inference_speed": "run_inference_speed_benchmarks.py",
            "memory_usage": "run_memory_usage_benchmarks.py"
        }
        self.results = {}

    def check_system_resources(self, min_memory_gb=2):
        """Check if system has sufficient resources to run benchmarks."""
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        
        print(f"System Memory Check:")
        print(f"  Available: {available_memory_gb:.2f} GB")
        print(f"  Required:  {min_memory_gb} GB")
        
        if available_memory_gb < min_memory_gb:
            print(f"  Status: INSUFFICIENT MEMORY - continuing with reduced expectations")
        else:
            print(f"  Status: SUFFICIENT MEMORY")
        
        # Check GPU if available
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU Memory: {gpu_memory:.2f} GB available")
        else:
            print(f"  GPU: Not available")
        
        return True

    def run_benchmark_with_monitoring(self, script_name: str, category: str, timeout_minutes: int = 15):
        """Run a benchmark script with system monitoring."""
        print(f"\n{'='*60}")
        print(f"EXECUTING {category.upper()} BENCHMARKS")
        print(f"Script: {script_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            # Run the benchmark with timeout
            result = subprocess.run([
                sys.executable, script_name
            ], 
            capture_output=True, 
            text=True, 
            timeout=timeout_minutes*60,
            cwd=os.getcwd())
            
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            status = "success" if result.returncode == 0 else "failed"
            
            benchmark_result = {
                "status": status,
                "execution_time": execution_time,
                "memory_delta": memory_delta,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "start_time": start_time,
                "end_time": end_time
            }
            
            print(f"  Status: {status.upper()}")
            print(f"  Execution Time: {execution_time:.2f}s")
            print(f"  Memory Change: {memory_delta:+.1f}%")
            
            return benchmark_result
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"  Status: TIMEOUT")
            print(f"  Execution Time: {execution_time:.2f}s (timeout)")
            
            return {
                "status": "timeout",
                "execution_time": execution_time,
                "error": f"Timed out after {timeout_minutes} minutes",
                "start_time": start_time,
                "end_time": end_time
            }
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"  Status: ERROR")
            print(f"  Execution Time: {execution_time:.2f}s")
            print(f"  Error: {str(e)}")
            
            return {
                "status": "error", 
                "execution_time": execution_time,
                "error": str(e),
                "start_time": start_time,
                "end_time": end_time
            }

    def cleanup_resources(self, delay: int = 3):
        """Clean up resources between benchmark runs."""
        print(f"  Cleaning up resources (waiting {delay}s)...")
        
        # Force garbage collection
        gc.collect()
        
        # Clear PyTorch cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # Wait to allow system to settle
        time.sleep(delay)

    def execute_benchmarks(self):
        """Execute benchmarks with real data collection."""
        print("SIMPLIFIED BENCHMARK EXECUTION DEMONSTRATION")
        print("="*80)
        print("Running a subset of benchmarks to demonstrate the implementation")
        print("="*80)
        
        # Check system resources
        if not self.check_system_resources(min_memory_gb=2):
            print("Insufficient resources. Exiting.")
            return None
        
        print(f"\nConfiguration:")
        print(f"  Models: {len(self.models)} ({', '.join(self.models)})")
        print(f"  Categories: {len(self.categories)} ({', '.join(self.categories)})")
        print(f"  Total Benchmark Sets: {len(self.models) * len(self.categories)}")
        print(f"  Estimated Duration: Up to {len(self.categories) * 15} minutes")
        
        print("Automatically proceeding with benchmark execution...")
        
        # Initialize results structure
        results = {
            "execution_metadata": {
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "total_duration_seconds": None,
                "total_benchmarks": len(self.categories),
                "models": self.models,
                "categories": self.categories,
                "platform": sys.platform,
                "python_version": sys.version,
                "system_info": {
                    "cpu_count": os.cpu_count(),
                    "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                    "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                    "cuda_available": torch.cuda.is_available()
                }
            },
            "benchmark_results": {},
            "execution_summary": {
                "successful_benchmarks": 0,
                "failed_benchmarks": 0,
                "timed_out_benchmarks": 0,
                "total_execution_time": 0
            }
        }
        
        print(f"\nStarting execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute each benchmark category
        for i, category in enumerate(self.categories, 1):
            print(f"\n[{i}/{len(self.categories)}] Executing {category.upper()} benchmarks...")
            
            script_name = self.script_mapping[category]
            benchmark_result = self.run_benchmark_with_monitoring(script_name, category)
            
            # Store result
            results["benchmark_results"][category] = benchmark_result
            
            # Update summary
            if benchmark_result["status"] == "success":
                results["execution_summary"]["successful_benchmarks"] += 1
            elif benchmark_result["status"] == "failed":
                results["execution_summary"]["failed_benchmarks"] += 1
            elif benchmark_result["status"] == "timeout":
                results["execution_summary"]["timed_out_benchmarks"] += 1
                
            results["execution_summary"]["total_execution_time"] += benchmark_result.get("execution_time", 0)
            
            # Clean up resources between benchmarks (except after the last one)
            if i < len(self.categories):
                self.cleanup_resources(delay=5)
        
        # Finalize results
        results["execution_metadata"]["end_time"] = datetime.now().isoformat()
        total_duration = time.time() - datetime.fromisoformat(results["execution_metadata"]["start_time"]).timestamp()
        results["execution_metadata"]["total_duration_seconds"] = total_duration
        
        # Print summary
        self.print_summary(results)
        
        # Save results
        self.save_results(results)
        
        return results

    def print_summary(self, results: Dict[str, Any]):
        """Print an execution summary."""
        print("\n" + "="*80)
        print("BENCHMARK EXECUTION SUMMARY")
        print("="*80)
        
        metadata = results["execution_metadata"]
        summary = results["execution_summary"]
        
        print(f"Execution Metadata:")
        print(f"  Start Time: {metadata['start_time']}")
        print(f"  End Time:   {metadata['end_time']}")
        print(f"  Duration:   {metadata['total_duration_seconds']:.2f}s ({metadata['total_duration_seconds']/60:.2f} min)")
        print(f"  Platform:   {metadata['platform']}")
        print(f"  Python:     {metadata['python_version'][:30]}")
        
        print(f"\nSystem Information:")
        sys_info = metadata['system_info']
        print(f"  CPU Cores:  {sys_info['cpu_count']}")
        print(f"  Total RAM:  {sys_info['memory_total_gb']:.1f} GB")
        print(f"  CUDA:       {'Yes' if sys_info['cuda_available'] else 'No'}")
        
        print(f"\nTest Configuration:")
        print(f"  Models:     {len(metadata['models'])}")
        for model in metadata['models']:
            print(f"    - {model}")
        print(f"  Categories: {len(metadata['categories'])}")
        for category in metadata['categories']:
            print(f"    - {category}")
        
        print(f"\nExecution Results:")
        print(f"  Successful: {summary['successful_benchmarks']}")
        print(f"  Failed:     {summary['failed_benchmarks']}")
        print(f"  Timed Out:  {summary['timed_out_benchmarks']}")
        print(f"  Total:      {metadata['total_benchmarks']}")
        
        success_rate = (summary['successful_benchmarks'] / metadata['total_benchmarks'] * 100) if metadata['total_benchmarks'] > 0 else 0
        print(f"  Success Rate: {success_rate:.1f}%")
        
        print(f"\nDetailed Results by Category:")
        for category, result in results["benchmark_results"].items():
            status = result["status"].upper()
            exec_time = result.get("execution_time", 0)
            mem_change = result.get("memory_delta", 0)
            print(f"  {category:<20} {status:<10} {exec_time:>6.1f}s ({mem_change:+.1f}%)")
            
            if result["status"] != "success":
                error_preview = result.get("error", result.get("stderr", ""))[:150]
                print(f"    Error: {error_preview}...")

    def save_results(self, results: Dict[str, Any]):
        """Save results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = f"simplified_benchmark_results_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {detailed_file}")
        
        # Save summary results
        summary_file = f"simplified_benchmark_results_summary_{timestamp}.json"
        summary_data = {
            "execution_summary": results["execution_metadata"],
            "benchmark_summary": results["execution_summary"],
            "detailed_results": {
                cat: {
                    "status": res["status"],
                    "execution_time": res.get("execution_time", 0),
                    "memory_delta": res.get("memory_delta", 0),
                    "return_code": res.get("return_code")
                } 
                for cat, res in results["benchmark_results"].items()
            }
        }
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        print(f"Summary results saved to: {summary_file}")
        
        # Create report
        report_file = f"simplified_benchmark_report_{timestamp}.txt"
        self.create_report(results, report_file)
        print(f"Report saved to: {report_file}")

    def create_report(self, results: Dict[str, Any], filename: str):
        """Create a human-readable report."""
        with open(filename, 'w') as f:
            f.write("INFERENC-PIO SIMPLIFIED BENCHMARK EXECUTION REPORT\n")
            f.write("="*70 + "\n\n")
            
            metadata = results["execution_metadata"]
            summary = results["execution_summary"]
            
            f.write(f"EXECUTION OVERVIEW:\n")
            f.write(f"  Start Time: {metadata['start_time']}\n")
            f.write(f"  End Time:   {metadata['end_time']}\n")
            f.write(f"  Duration:   {metadata['total_duration_seconds']:.2f} seconds\n")
            f.write(f"  Platform:   {metadata['platform']}\n\n")
            
            f.write(f"SYSTEM INFORMATION:\n")
            sys_info = metadata['system_info']
            f.write(f"  CPU Cores:  {sys_info['cpu_count']}\n")
            f.write(f"  Total RAM:  {sys_info['memory_total_gb']:.1f} GB\n")
            f.write(f"  CUDA:       {'Yes' if sys_info['cuda_available'] else 'No'}\n\n")
            
            f.write(f"TEST CONFIGURATION:\n")
            f.write(f"  Models: {len(metadata['models'])}\n")
            for model in metadata['models']:
                f.write(f"    - {model}\n")
            f.write(f"  Categories: {len(metadata['categories'])}\n")
            for category in metadata['categories']:
                f.write(f"    - {category}\n\n")
            
            f.write(f"EXECUTION SUMMARY:\n")
            f.write(f"  Successful: {summary['successful_benchmarks']}\n")
            f.write(f"  Failed:     {summary['failed_benchmarks']}\n")
            f.write(f"  Timed Out:  {summary['timed_out_benchmarks']}\n")
            f.write(f"  Success Rate: {(summary['successful_benchmarks'] / metadata['total_benchmarks'] * 100):.1f}%\n\n")
            
            f.write(f"DETAILED RESULTS:\n")
            for category, result in results["benchmark_results"].items():
                f.write(f"  {category.upper()}:\n")
                f.write(f"    Status: {result['status'].upper()}\n")
                f.write(f"    Execution Time: {result.get('execution_time', 0):.2f}s\n")
                f.write(f"    Memory Change: {result.get('memory_delta', 0):+.2f}%\n")
                
                if result["status"] != "success":
                    error = result.get("error", result.get("stderr", ""))[:300]
                    f.write(f"    Error: {error}...\n")
                    
                f.write(f"    Details:\n")
                stdout = result.get("stdout", "")
                if stdout:
                    lines = stdout.split('\n')
                    for line in lines[-10:]:  # Last 10 lines of output
                        if line.strip():
                            f.write(f"      {line[:100]}\n")
                f.write(f"\n")


def main():
    """Main execution function."""
    executor = SimplifiedBenchmarkExecutor()
    
    print("INFERENC-PIO SIMPLIFIED BENCHMARK EXECUTION DEMONSTRATION")
    print("This demonstrates the execution framework with a subset of benchmarks.")
    
    try:
        results = executor.execute_benchmarks()
        if results:
            print(f"\nBENCHMARK EXECUTION COMPLETED!")
            print(f"Results saved with timestamp {datetime.now().strftime('%Y%m%d_%H%M%S')}")
    except KeyboardInterrupt:
        print(f"\n\nExecution interrupted by user.")
        print("Partial results may be available in output files.")
    except Exception as e:
        print(f"\n\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()