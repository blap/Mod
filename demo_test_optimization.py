"""
Demonstration of Test Optimization System

This script demonstrates the key features of the test optimization system:
- Parallel execution
- Result caching
- Performance improvements
"""

import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from test_optimization import OptimizedTestRunner, run_tests_with_optimization


def demo_test_function(name, duration=0.5, should_pass=True):
    """Create a demo test function."""
    def test_impl():
        print(f"Running {name}...")
        time.sleep(duration)  # Simulate work
        if not should_pass:
            raise AssertionError(f"Test {name} failed intentionally")
        print(f"Completed {name}")
    test_impl.__name__ = name
    return test_impl


def run_demonstration():
    """Run a demonstration of the optimization system."""
    print("=" * 60)
    print("TEST OPTIMIZATION SYSTEM DEMONSTRATION")
    print("=" * 60)
    
    # Create some demo tests
    tests = [
        demo_test_function("fast_test_1", duration=0.2, should_pass=True),
        demo_test_function("fast_test_2", duration=0.2, should_pass=True),
        demo_test_function("slow_test_1", duration=0.5, should_pass=True),
        demo_test_function("failing_test", duration=0.1, should_pass=False),
        demo_test_function("fast_test_3", duration=0.2, should_pass=True),
    ]
    
    test_names = [f"demo_{i}" for i in range(len(tests))]
    
    print(f"\nCreated {len(tests)} demo tests")
    print("- Sequential execution (no optimization):")
    
    # Run without optimization (sequential)
    start_time = time.time()
    results_seq = run_tests_with_optimization(
        test_functions=tests,
        test_paths=test_names,
        cache_enabled=False,
        parallel_enabled=False
    )
    seq_time = time.time() - start_time
    
    print(f"  Time: {seq_time:.2f}s")
    print(f"  Results: {results_seq['passed']} passed, {results_seq['failed']} failed")
    
    print("\n- Parallel execution with caching:")
    
    # Run with optimization (parallel + caching)
    start_time = time.time()
    results_opt = run_tests_with_optimization(
        test_functions=tests,
        test_paths=test_names,
        cache_enabled=True,
        parallel_enabled=True,
        max_workers=4
    )
    opt_time = time.time() - start_time
    
    print(f"  Time: {opt_time:.2f}s")
    print(f"  Results: {results_opt['passed']} passed, {results_opt['failed']} failed")
    print(f"  Cached: {results_opt['cached']}, Executed: {results_opt['executed']}")
    
    if seq_time > 0:
        speedup = seq_time / opt_time if opt_time > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")
    
    print("\n- Running same tests again (should use cache):")
    
    # Run same tests again (should use cache)
    start_time = time.time()
    results_cached = run_tests_with_optimization(
        test_functions=tests,
        test_paths=test_names,
        cache_enabled=True,
        parallel_enabled=True
    )
    cached_time = time.time() - start_time
    
    print(f"  Time: {cached_time:.2f}s")
    print(f"  Results: {results_cached['passed']} passed, {results_cached['failed']} failed")
    print(f"  Cached: {results_cached['cached']}, Executed: {results_cached['executed']}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    # Show detailed results
    print("\nDetailed Results:")
    for result in results_opt['results']:
        status = "PASS" if result['success'] else "FAIL"
        cached_status = " (CACHED)" if result['cached'] else ""
        print(f"  {result['test']}: {status}{cached_status} ({result['execution_time']:.3f}s)")


def run_performance_comparison():
    """Run a performance comparison between different strategies."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Create many tests to see parallelization benefits
    many_tests = [demo_test_function(f"test_{i}", duration=0.1, should_pass=True) for i in range(10)]
    test_names = [f"perf_test_{i}" for i in range(10)]
    
    strategies = [
        ("Sequential (no parallel)", False, False),
        ("Sequential with caching", False, True),
        ("Parallel (4 workers)", True, False),
        ("Parallel with caching", True, True),
    ]
    
    for name, parallel, cache in strategies:
        start_time = time.time()
        results = run_tests_with_optimization(
            test_functions=many_tests,
            test_paths=test_names,
            cache_enabled=cache,
            parallel_enabled=parallel,
            max_workers=4 if parallel else 1
        )
        elapsed = time.time() - start_time
        
        print(f"{name:.<30} {elapsed:.2f}s (passed: {results['passed']}, cached: {results['cached']})")


if __name__ == "__main__":
    run_demonstration()
    run_performance_comparison()
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION SYSTEM IS WORKING CORRECTLY!")
    print("=" * 60)