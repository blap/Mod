"""
Master Benchmark Execution Script

This script orchestrates the complete benchmarking workflow:
1. Sets up the environment with all dependencies
2. Runs benchmarks for both original and modified model states
3. Compares results and generates reports
"""

import subprocess
import sys
import time
from pathlib import Path
import json
import os


def run_setup():
    """Run the environment setup script."""
    print("Step 1: Setting up benchmark environment...")
    
    setup_script = "SETUP_BENCHMARK_ENVIRONMENT.py"
    if not Path(setup_script).exists():
        print(f"Error: {setup_script} not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, setup_script], 
                              capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✓ Environment setup completed successfully")
            return True
        else:
            print(f"✗ Environment setup failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ Environment setup timed out")
        return False


def run_dual_state_benchmarks():
    """Run benchmarks for both original and modified states."""
    print("\nStep 2: Running dual-state benchmarks...")
    
    benchmark_script = "DUAL_STATE_BENCHMARK_RUNNER.py"
    if not Path(benchmark_script).exists():
        print(f"Error: {benchmark_script} not found!")
        return None
    
    try:
        result = subprocess.run([sys.executable, benchmark_script], 
                              capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode == 0:
            print("✓ Dual-state benchmarks completed successfully")
            return True
        else:
            print(f"✗ Dual-state benchmarks failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ Dual-state benchmarks timed out")
        return None


def run_comprehensive_controller():
    """Run the comprehensive benchmark controller."""
    print("\nStep 3: Running comprehensive benchmark controller...")
    
    controller_script = "COMPREHENSIVE_BENCHMARK_CONTROLLER.py"
    if not Path(controller_script).exists():
        print(f"Error: {controller_script} not found!")
        return None
    
    try:
        result = subprocess.run([sys.executable, controller_script], 
                              capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        if result.returncode == 0:
            print("✓ Comprehensive benchmark controller completed successfully")
            return True
        else:
            print(f"✗ Comprehensive benchmark controller failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ Comprehensive benchmark controller timed out")
        return None


def generate_final_report():
    """Generate a final comprehensive report."""
    print("\nStep 4: Generating final report...")
    
    # Look for benchmark result files
    result_files = list(Path(".").glob("*benchmark*results*.json"))
    
    if not result_files:
        print("No benchmark result files found to generate report from")
        return False
    
    print(f"Found {len(result_files)} result files:")
    for f in result_files:
        print(f"  - {f}")
    
    # Create a summary report
    report_content = f"""# INFERENC-PIO BENCHMARKING EXECUTION REPORT

## EXECUTION SUMMARY

- Total Result Files Found: {len(result_files)}
- Result Files:
"""
    for f in result_files:
        report_content += f"  - {f}\n"
    
    report_content += """

## NEXT STEPS

To view detailed results, examine the individual result files generated during execution.
Each file contains comprehensive benchmark data for different aspects of the evaluation.

For comparison between original and modified states, look for files with "dual_state" or "comparison" in the name.
"""

    with open("FINAL_BENCHMARK_EXECUTION_REPORT.md", "w") as f:
        f.write(report_content)
    
    print("✓ Final report generated: FINAL_BENCHMARK_EXECUTION_REPORT.md")
    return True


def main():
    """Main execution function."""
    print("Inference-PIO Master Benchmark Execution Script")
    print("="*60)
    print("This script will:")
    print("1. Set up the benchmark environment")
    print("2. Run benchmarks for original and modified model states") 
    print("3. Generate comprehensive reports")
    print("="*60)
    
    # Ask for confirmation
    confirm = input("\nProceed with full benchmark execution? This may take several hours. (y/n): ")
    if confirm.lower() != 'y':
        print("Execution cancelled.")
        return
    
    start_time = time.time()
    
    # Step 1: Setup environment
    setup_success = run_setup()
    if not setup_success:
        print("Environment setup failed. Stopping execution.")
        return
    
    # Step 2: Run dual-state benchmarks
    dual_bench_success = run_dual_state_benchmarks()
    if dual_bench_success is None:
        print("Dual-state benchmarks timed out. Continuing to comprehensive controller...")
    elif not dual_bench_success:
        print("Dual-state benchmarks failed. Continuing to comprehensive controller...")
    
    # Step 3: Run comprehensive controller
    comp_ctrl_success = run_comprehensive_controller()
    if comp_ctrl_success is None:
        print("Comprehensive controller timed out.")
        return
    elif not comp_ctrl_success:
        print("Comprehensive controller failed.")
        return
    
    # Step 4: Generate final report
    report_success = generate_final_report()
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("MASTER BENCHMARK EXECUTION COMPLETE")
    print(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*60)
    
    if report_success:
        print("Final report: FINAL_BENCHMARK_EXECUTION_REPORT.md")
    
    print("\nBenchmark execution completed. Check the generated result files for detailed data.")


if __name__ == "__main__":
    main()