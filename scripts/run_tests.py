"""
Advanced Test Runner with Parallelization and Caching

This script provides a comprehensive test runner that leverages parallel execution 
and caching to optimize test execution speed while maintaining reliability.
"""

import argparse
import sys
import os
import time
from pathlib import Path
import json
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_optimization import (
    OptimizedTestRunner,
    run_tests_with_optimization,
    run_test_directory
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced Test Runner with Optimization")
    parser.add_argument(
        '--directory', '-d',
        type=str,
        default='tests',
        help='Directory containing test files (default: tests)'
    )
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        default='test_*.py',
        help='Pattern for test files (default: test_*.py)'
    )
    parser.add_argument(
        '--parallel', '-P',
        action='store_true',
        default=True,
        help='Enable parallel execution (default: True)'
    )
    parser.add_argument(
        '--no-parallel',
        dest='parallel',
        action='store_false',
        help='Disable parallel execution'
    )
    parser.add_argument(
        '--cache', '-c',
        action='store_true',
        default=True,
        help='Enable result caching (default: True)'
    )
    parser.add_argument(
        '--no-cache',
        dest='cache',
        action='store_false',
        help='Disable result caching'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=False,
        help='Enable verbose output'
    )
    parser.add_argument(
        '--report', '-r',
        type=str,
        default=None,
        help='Generate test report in JSON format'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        default=False,
        help='List discovered tests without running them'
    )
    
    return parser.parse_args()


def discover_test_files(directory: str, pattern: str) -> List[str]:
    """Discover test files in the specified directory."""
    test_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and (file.startswith('test_') or '_test' in file):
                test_files.append(os.path.join(root, file))
    
    return test_files


def print_test_summary(results: Dict[str, Any], verbose: bool = False):
    """Print a summary of test results."""
    print("\n" + "="*60)
    print("TEST EXECUTION SUMMARY")
    print("="*60)
    
    print(f"Total tests: {results['total_tests']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Cached: {results['cached']}")
    print(f"Executed: {results['executed']}")
    print(f"Execution time: {results['execution_time']:.2f}s")
    
    if results['cached'] > 0:
        print(f"Cache hits: {results['cache_hits']}")
        print(f"Cache misses: {results['cache_misses']}")
    
    success_rate = (results['passed'] / results['total_tests']) * 100 if results['total_tests'] > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    if verbose and results['failed'] > 0:
        print("\nFAILED TESTS:")
        for result in results['results']:
            if not result['success']:
                print(f"  - {result['test']}: {result.get('error', 'Unknown error')}")
    
    print("="*60)


def generate_report(results: Dict[str, Any], report_path: str):
    """Generate a JSON report of test results."""
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Report generated: {report_path}")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    print(f"Test Runner Configuration:")
    print(f"  Directory: {args.directory}")
    print(f"  Parallel: {args.parallel}")
    print(f"  Cache: {args.cache}")
    print(f"  Workers: {args.workers or 'auto'}")
    print(f"  Verbose: {args.verbose}")
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory {args.directory} does not exist")
        sys.exit(1)
    
    # Discover test files
    test_files = discover_test_files(args.directory, args.pattern)
    print(f"Discovered {len(test_files)} test files")
    
    if args.list:
        print("\nDiscovered test files:")
        for test_file in test_files:
            print(f"  - {test_file}")
        return
    
    if not test_files:
        print("No test files found!")
        return
    
    # Create optimized test runner
    runner = OptimizedTestRunner(
        cache_enabled=args.cache,
        parallel_enabled=args.parallel,
        max_workers=args.workers
    )
    
    print(f"\nStarting test execution...")
    start_time = time.time()
    
    # Run tests using the existing function
    from test_optimization import run_test_directory
    results = run_test_directory(
        directory_path=args.directory,
        cache_enabled=args.cache,
        parallel_enabled=args.parallel,
        max_workers=args.workers
    )
    
    total_time = time.time() - start_time
    
    # Print summary
    print_test_summary(results, args.verbose)
    
    # Generate report if requested
    if args.report:
        generate_report(results, args.report)
    
    # Exit with appropriate code
    sys.exit(0 if results['passed'] == results['total_tests'] else 1)


def run_specific_tests(test_functions, test_names, config=None):
    """
    Run specific test functions with given configuration.
    
    Args:
        test_functions: List of test functions to run
        test_names: Corresponding names for the tests
        config: Configuration dict with keys: cache_enabled, parallel_enabled, max_workers
    """
    if config is None:
        config = {
            'cache_enabled': True,
            'parallel_enabled': True,
            'max_workers': None
        }
    
    results = run_tests_with_optimization(
        test_functions=test_functions,
        test_paths=test_names,
        cache_enabled=config.get('cache_enabled', True),
        parallel_enabled=config.get('parallel_enabled', True),
        max_workers=config.get('max_workers', None)
    )
    
    return results


if __name__ == "__main__":
    main()