"""
Main Benchmark Runner for Qwen3-VL Model Optimizations
This module provides a unified interface to run all benchmarking components.
"""
import sys
import os
import argparse
from typing import Dict, Any
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.comprehensive_benchmark_suite import run_comprehensive_benchmarks
from benchmarks.accuracy_preservation_tests import run_accuracy_preservation_tests
from benchmarks.hardware_specific_benchmarks import run_hardware_specific_benchmarks
from benchmarks.comparative_benchmarks import run_comparative_benchmarks
from benchmarks.memory_efficiency_tests import run_memory_efficiency_tests
from benchmarks.scalability_tests import run_scalability_tests
from benchmarks.system_benchmarks import run_system_benchmarks
from benchmarks.automated_test_framework import run_automated_tests, run_specific_tests


def run_all_benchmarks():
    """Run all benchmarking components"""
    print("=" * 80)
    print("RUNNING ALL BENCHMARKING COMPONENTS FOR QWEN3-VL OPTIMIZATIONS")
    print("=" * 80)
    
    # Create results directory
    Path("benchmark_results").mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run comprehensive benchmarks
    print("\n1. Running Comprehensive Benchmarks...")
    results['comprehensive'] = run_comprehensive_benchmarks()
    
    # Run accuracy preservation tests
    print("\n2. Running Accuracy Preservation Tests...")
    results['accuracy'] = run_accuracy_preservation_tests()
    
    # Run hardware-specific benchmarks
    print("\n3. Running Hardware-Specific Benchmarks...")
    results['hardware'] = run_hardware_specific_benchmarks()
    
    # Run comparative benchmarks
    print("\n4. Running Comparative Benchmarks...")
    results['comparative'] = run_comparative_benchmarks()
    
    # Run memory efficiency tests
    print("\n5. Running Memory Efficiency Tests...")
    results['memory_efficiency'] = run_memory_efficiency_tests()
    
    # Run scalability tests
    print("\n6. Running Scalability Tests...")
    results['scalability'] = run_scalability_tests()
    
    # Run system-level benchmarks
    print("\n7. Running System-Level Benchmarks...")
    results['system'] = run_system_benchmarks()
    
    # Save all results
    with open("benchmark_results/all_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("ALL BENCHMARKING COMPONENTS COMPLETED")
    print("Results saved to benchmark_results/all_benchmark_results.json")
    print("=" * 80)
    
    return results


def main():
    """Main function to handle command-line arguments"""
    parser = argparse.ArgumentParser(description="Benchmark Runner for Qwen3-VL Model Optimizations")
    parser.add_argument(
        "--benchmark-type", 
        choices=[
            "all", "comprehensive", "accuracy", "hardware", "comparative", 
            "memory", "scalability", "system", "automated", "specific"
        ],
        default="all",
        help="Type of benchmark to run"
    )
    parser.add_argument(
        "--specific-tests",
        nargs="+",
        help="Specific tests to run when benchmark-type is 'specific'"
    )
    
    args = parser.parse_args()
    
    if args.benchmark_type == "all":
        run_all_benchmarks()
    elif args.benchmark_type == "comprehensive":
        run_comprehensive_benchmarks()
    elif args.benchmark_type == "accuracy":
        run_accuracy_preservation_tests()
    elif args.benchmark_type == "hardware":
        run_hardware_specific_benchmarks()
    elif args.benchmark_type == "comparative":
        run_comparative_benchmarks()
    elif args.benchmark_type == "memory":
        run_memory_efficiency_tests()
    elif args.benchmark_type == "scalability":
        run_scalability_tests()
    elif args.benchmark_type == "system":
        run_system_benchmarks()
    elif args.benchmark_type == "automated":
        run_automated_tests()
    elif args.benchmark_type == "specific":
        if not args.specific_tests:
            print("Error: --specific-tests must be provided when using 'specific' benchmark type")
            return
        run_specific_tests(args.specific_tests)


if __name__ == "__main__":
    main()