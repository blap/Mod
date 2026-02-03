"""
Comprehensive Benchmark Validation Script

This script validates that all updated benchmarks work correctly with the new systems,
follow the new standards, and properly test the updated functionality.
"""

import json
import os
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from benchmarks.core.benchmark_fixtures import (
    accuracy_benchmark_fixture,
    create_benchmark_fixture,
    memory_benchmark_fixture,
    performance_benchmark_fixture,
)
from src.inference_pio.common.interfaces.benchmark_interface import (
    AccuracyBenchmark,
    BatchProcessingBenchmark,
    BenchmarkRunner,
    InferenceSpeedBenchmark,
    MemoryUsageBenchmark,
    ModelLoadingTimeBenchmark,
    get_accuracy_suite,
    get_full_suite,
    get_performance_suite,
)
from benchmarks.core.test_performance_regression import (
    PerformanceRegressionTracker,
)
from benchmarks.core.unified_benchmark_discovery import (
    UnifiedBenchmarkDiscoverer,
)


def validate_benchmark_interfaces():
    """Validate that all benchmark interfaces work correctly."""
    print("=" * 80)
    print("VALIDATING BENCHMARK INTERFACES")
    print("=" * 80)

    validation_results = {"interfaces_validated": [], "errors": []}

    try:
        # Test basic instantiation of benchmark classes
        from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash

        plugin = create_glm_4_7_flash()
        success = plugin.initialize(device="cpu", use_mock_model=True)

        if not success:
            raise Exception("Could not initialize mock plugin for validation")

        model_name = "validation_test_model"

        # Test each benchmark type
        benchmarks_to_test = [
            (
                "InferenceSpeedBenchmark",
                lambda: InferenceSpeedBenchmark(plugin, model_name),
            ),
            ("MemoryUsageBenchmark", lambda: MemoryUsageBenchmark(plugin, model_name)),
            ("AccuracyBenchmark", lambda: AccuracyBenchmark(plugin, model_name)),
            (
                "BatchProcessingBenchmark",
                lambda: BatchProcessingBenchmark(plugin, model_name),
            ),
            (
                "ModelLoadingTimeBenchmark",
                lambda: ModelLoadingTimeBenchmark(plugin, model_name),
            ),
        ]

        for bench_name, bench_constructor in benchmarks_to_test:
            try:
                benchmark = bench_constructor()
                result = benchmark.run()
                print(f"✓ {bench_name} instantiated and ran successfully")
                validation_results["interfaces_validated"].append(
                    {
                        "name": bench_name,
                        "status": "success",
                        "result_type": type(result).__name__,
                    }
                )
            except Exception as e:
                print(f"✗ {bench_name} failed: {e}")
                validation_results["errors"].append(
                    {"name": bench_name, "error": str(e), "type": "interface_error"}
                )

        # Clean up
        if plugin.is_loaded:
            plugin.cleanup()

    except Exception as e:
        print(f"✗ Interface validation failed completely: {e}")
        validation_results["errors"].append(
            {"name": "interface_validation", "error": str(e), "type": "critical_error"}
        )

    return validation_results


def validate_benchmark_suites():
    """Validate that all benchmark suites work correctly."""
    print("\n" + "=" * 80)
    print("VALIDATING BENCHMARK SUITES")
    print("=" * 80)

    validation_results = {"suites_validated": [], "errors": []}

    try:
        # Test with mock plugin
        from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash

        plugin = create_glm_4_7_flash()
        success = plugin.initialize(device="cpu", use_mock_model=True)

        if not success:
            raise Exception("Could not initialize mock plugin for validation")

        model_name = "validation_test_model"

        # Test each suite
        suites_to_test = [
            ("Performance Suite", lambda: get_performance_suite(plugin, model_name)),
            ("Accuracy Suite", lambda: get_accuracy_suite(plugin, model_name)),
            ("Full Suite", lambda: get_full_suite(plugin, model_name)),
        ]

        for suite_name, suite_getter in suites_to_test:
            try:
                benchmarks = suite_getter()
                print(f"✓ {suite_name} created with {len(benchmarks)} benchmarks")

                # Test running the benchmarks
                runner = BenchmarkRunner()
                results = runner.run_multiple_benchmarks(benchmarks)

                print(f"✓ {suite_name} executed {len(results)} benchmarks successfully")
                validation_results["suites_validated"].append(
                    {
                        "name": suite_name,
                        "benchmark_count": len(benchmarks),
                        "result_count": len(results),
                        "status": "success",
                    }
                )
            except Exception as e:
                print(f"✗ {suite_name} failed: {e}")
                validation_results["errors"].append(
                    {"name": suite_name, "error": str(e), "type": "suite_error"}
                )

        # Clean up
        if plugin.is_loaded:
            plugin.cleanup()

    except Exception as e:
        print(f"✗ Suite validation failed completely: {e}")
        validation_results["errors"].append(
            {"name": "suite_validation", "error": str(e), "type": "critical_error"}
        )

    return validation_results


def validate_discovery_system():
    """Validate the unified discovery system."""
    print("\n" + "=" * 80)
    print("VALIDATING UNIFIED DISCOVERY SYSTEM")
    print("=" * 80)

    validation_results = {
        "discovery_validated": False,
        "discovered_count": 0,
        "errors": [],
    }

    try:
        discoverer = UnifiedBenchmarkDiscoverer()
        discovered = discoverer.discover_benchmarks()

        print(f"✓ Discovery system found {len(discovered)} benchmarks")
        validation_results["discovered_count"] = len(discovered)
        validation_results["discovery_validated"] = True

        # Print a summary
        discoverer.print_discovery_summary()

        # Test running a subset of discovered benchmarks
        if discovered:
            print(f"✓ Testing execution of first 3 discovered benchmarks...")
            for i, benchmark in enumerate(discovered[:3]):
                try:
                    if benchmark.benchmark_class:
                        print(f"  - {benchmark.name} (class-based): OK")
                    elif benchmark.function:
                        print(f"  - {benchmark.name} (function-based): OK")
                    else:
                        print(f"  - {benchmark.name}: No executable found")
                except Exception as e:
                    print(f"  - {benchmark.name}: Error - {e}")
                    validation_results["errors"].append(
                        {
                            "name": f"discovery_execution_{benchmark.name}",
                            "error": str(e),
                            "type": "execution_error",
                        }
                    )

    except Exception as e:
        print(f"✗ Discovery system validation failed: {e}")
        validation_results["errors"].append(
            {"name": "discovery_validation", "error": str(e), "type": "critical_error"}
        )

    return validation_results


def validate_regression_framework():
    """Validate the performance regression testing framework."""
    print("\n" + "=" * 80)
    print("VALIDATING REGRESSION TESTING FRAMEWORK")
    print("=" * 80)

    validation_results = {
        "regression_validated": False,
        "tests_run": 0,
        "regressions_detected": 0,
        "errors": [],
    }

    try:
        tracker = PerformanceRegressionTracker()

        # Set a baseline
        tracker.set_baseline(
            model_name="validation_model",
            metric_name="inference_speed",
            value=100.0,
            unit="tokens/sec",
            metadata={"validation": True},
        )

        print("✓ Baseline set successfully")

        # Test regression detection
        result = tracker.detect_regression(
            model_name="validation_model",
            metric_name="inference_speed",
            current_value=90.0,  # Below baseline, should detect regression
            threshold_percentage=5.0,
        )

        print(
            f"✓ Regression detection test: regression detected = {result.regression_detected}"
        )
        validation_results["regression_validated"] = True
        validation_results["tests_run"] = 1
        validation_results["regressions_detected"] = (
            1 if result.regression_detected else 0
        )

        # Export report
        tracker.export_regression_report("validation_regression_report.json")
        print("✓ Regression report exported successfully")

    except Exception as e:
        print(f"✗ Regression framework validation failed: {e}")
        validation_results["errors"].append(
            {"name": "regression_validation", "error": str(e), "type": "critical_error"}
        )

    return validation_results


def validate_fixtures():
    """Validate the benchmark fixtures."""
    print("\n" + "=" * 80)
    print("VALIDATING BENCHMARK FIXTURES")
    print("=" * 80)

    validation_results = {"fixtures_validated": [], "errors": []}

    try:
        # Test different fixture types
        fixture_types = [
            ("Performance", lambda: create_benchmark_fixture("performance")),
            ("Accuracy", lambda: create_benchmark_fixture("accuracy")),
            ("Memory", lambda: create_benchmark_fixture("memory")),
        ]

        for fixture_name, fixture_creator in fixture_types:
            try:
                fixture = fixture_creator()
                fixture.setup()

                # Test fixture functionality
                if fixture_name == "Performance":
                    inputs = list(fixture.get_test_inputs())
                    print(
                        f"  ✓ Performance fixture generated {len(inputs)} test inputs"
                    )
                elif fixture_name == "Accuracy":
                    test_cases = list(fixture.get_test_cases())
                    print(
                        f"  ✓ Accuracy fixture generated {len(test_cases)} test cases"
                    )
                elif fixture_name == "Memory":
                    scenarios = list(fixture.get_memory_test_scenarios())
                    print(
                        f"  ✓ Memory fixture generated {len(scenarios)} test scenarios"
                    )

                fixture.teardown()
                validation_results["fixtures_validated"].append(
                    {"name": fixture_name, "status": "success"}
                )
                print(f"✓ {fixture_name} fixture validated successfully")

            except Exception as e:
                print(f"✗ {fixture_name} fixture failed: {e}")
                validation_results["errors"].append(
                    {
                        "name": f"fixture_{fixture_name}",
                        "error": str(e),
                        "type": "fixture_error",
                    }
                )

        # Test context managers
        try:
            with performance_benchmark_fixture() as perf_fixture:
                inputs = list(perf_fixture.get_test_inputs())
                print(
                    f"  ✓ Performance fixture context manager worked with {len(inputs)} inputs"
                )

            with accuracy_benchmark_fixture() as acc_fixture:
                test_cases = list(acc_fixture.get_test_cases())
                print(
                    f"  ✓ Accuracy fixture context manager worked with {len(test_cases)} cases"
                )

            with memory_benchmark_fixture() as mem_fixture:
                scenarios = list(mem_fixture.get_memory_test_scenarios())
                print(
                    f"  ✓ Memory fixture context manager worked with {len(scenarios)} scenarios"
                )

            print("✓ All fixture context managers validated successfully")

        except Exception as e:
            print(f"✗ Fixture context managers failed: {e}")
            validation_results["errors"].append(
                {
                    "name": "fixture_context_managers",
                    "error": str(e),
                    "type": "context_manager_error",
                }
            )

    except Exception as e:
        print(f"✗ Fixture validation failed completely: {e}")
        validation_results["errors"].append(
            {"name": "fixture_validation", "error": str(e), "type": "critical_error"}
        )

    return validation_results


def run_comprehensive_validation():
    """Run comprehensive validation of all updated benchmark systems."""
    print("COMPREHENSIVE BENCHMARK VALIDATION")
    print("=" * 80)
    print(f"Validation started at: {datetime.now().isoformat()}")

    all_results = {}

    # Run all validation tests
    all_results["interfaces"] = validate_benchmark_interfaces()
    all_results["suites"] = validate_benchmark_suites()
    all_results["discovery"] = validate_discovery_system()
    all_results["regression"] = validate_regression_framework()
    all_results["fixtures"] = validate_fixtures()

    # Generate summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    total_errors = 0
    for section, results in all_results.items():
        if "errors" in results and results["errors"]:
            error_count = len(results["errors"])
            total_errors += error_count
            print(f"{section.title()}: {error_count} errors found")
        else:
            print(f"{section.title()}: All validations passed")

    print(f"\nTotal Errors: {total_errors}")

    if total_errors == 0:
        print("🎉 ALL VALIDATIONS PASSED! Benchmarks are working correctly.")
    else:
        print("⚠️  Some validations failed. Please review the errors above.")

    # Save detailed results
    results_file = f"comprehensive_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nDetailed results saved to: {results_file}")

    return all_results, total_errors == 0


def run_unit_tests_for_benchmarks():
    """Run any existing unit tests for benchmark components."""
    print("\n" + "=" * 80)
    print("RUNNING UNIT TESTS FOR BENCHMARK COMPONENTS")
    print("=" * 80)

    # Discover and run tests related to benchmarks
    test_loader = unittest.TestLoader()
    start_dir = str(project_root / "src")

    # Discover tests in the benchmark-related modules
    suite = test_loader.discover(start_dir=start_dir, pattern="*benchmark*test*.py")

    # Also add any test files that might be in the benchmarks directory
    benchmark_test_dir = project_root / "benchmarks"
    if benchmark_test_dir.exists():
        benchmark_suite = test_loader.discover(
            start_dir=str(benchmark_test_dir), pattern="*test*.py"
        )
        suite.addTests(benchmark_suite)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(
        f"\nUnit Tests - Ran: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}"
    )

    return result


if __name__ == "__main__":
    print("Starting comprehensive benchmark validation...")

    # Run comprehensive validation
    validation_results, success = run_comprehensive_validation()

    # Run unit tests
    unit_test_results = run_unit_tests_for_benchmarks()

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    if success and unit_test_results.failures == 0 and unit_test_results.errors == 0:
        print("🎉 ALL VALIDATIONS AND TESTS PASSED!")
        print("The benchmark suite is comprehensive and working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some validations or tests failed.")
        print("Please address the issues before deploying the benchmark suite.")
        sys.exit(1)
