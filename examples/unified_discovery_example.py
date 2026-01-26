"""
Example usage of the unified test discovery system.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inference_pio.unified_test_discovery import (
    UnifiedTestDiscovery,
    discover_and_run_all_items,
    discover_and_run_tests_only,
    discover_and_run_benchmarks_only,
    discover_tests_for_model,
    discover_benchmarks_for_model,
    run_tests_for_model,
    run_benchmarks_for_model,
    get_discovery_summary
)


def example_basic_usage():
    """Example of basic usage of the unified discovery system."""
    print("=== Basic Usage Example ===")
    
    # Create a discovery instance
    discovery = UnifiedTestDiscovery()
    
    # Discover all items in the project
    items = discovery.discover_all()
    
    print(f"Discovered {len(items)} total items")
    print(f"  Tests: {len(discovery.test_functions)}")
    print(f"  Benchmarks: {len(discovery.benchmark_functions)}")
    
    # Show some details about discovered items
    if items:
        print("\nFirst few discovered items:")
        for i, item in enumerate(items[:5]):  # Show first 5 items
            print(f"  {i+1}. {item['full_name']} ({item['type'].value}) - {item['category']}")
    
    print()


def example_model_specific_discovery():
    """Example of discovering items for a specific model."""
    print("=== Model-Specific Discovery Example ===")
    
    # Discover tests for a specific model
    model_tests = discover_tests_for_model('qwen3_vl_2b')
    print(f"Found {len(model_tests)} tests for qwen3_vl_2b model")
    
    # Discover benchmarks for a specific model
    model_benchmarks = discover_benchmarks_for_model('qwen3_vl_2b')
    print(f"Found {len(model_benchmarks)} benchmarks for qwen3_vl_2b model")
    
    print()


def example_running_tests_and_benchmarks():
    """Example of running tests and benchmarks."""
    print("=== Running Tests and Benchmarks Example ===")
    
    # Get discovery summary
    summary = get_discovery_summary()
    print(f"Discovery Summary:")
    print(f"  Total Items: {summary['total_items']}")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Total Benchmarks: {summary['total_benchmarks']}")
    print(f"  By Type: {summary['by_type']}")
    print(f"  By Category: {summary['by_category']}")
    print(f"  By Model: {summary['by_model']}")
    
    print()


def example_convenience_functions():
    """Example of using convenience functions."""
    print("=== Convenience Functions Example ===")
    
    # Discover and run all items (tests and benchmarks)
    print("Using discover_and_run_all_items()...")
    # Note: Commenting out actual execution to avoid running all tests during example
    # results = discover_and_run_all_items()
    
    # Discover and run only tests
    print("Using discover_and_run_tests_only()...")
    # Note: Commenting out actual execution to avoid running all tests during example
    # results = discover_and_run_tests_only()
    
    # Discover and run only benchmarks
    print("Using discover_and_run_benchmarks_only()...")
    # Note: Commenting out actual execution to avoid running all benchmarks during example
    # results = discover_and_run_benchmarks_only()
    
    print("Convenience functions demonstrated (execution commented out for example purposes)")
    print()


def example_advanced_filtering():
    """Example of advanced filtering capabilities."""
    print("=== Advanced Filtering Example ===")
    
    discovery = UnifiedTestDiscovery()
    discovery.discover_all()
    
    # Get items by type
    unit_tests = discovery.get_items_by_type(discovery.TestType.UNIT_TEST)
    print(f"Found {len(unit_tests)} unit tests")
    
    # Get items by category
    performance_items = discovery.get_items_by_category('performance')
    print(f"Found {len(performance_items)} performance items")
    
    # Get items by model
    model_items = discovery.get_items_by_model('general')
    print(f"Found {len(model_items)} items for 'general' model")
    
    print()


if __name__ == "__main__":
    print("Unified Test Discovery System - Example Usage")
    print("=" * 50)
    
    example_basic_usage()
    example_model_specific_discovery()
    example_running_tests_and_benchmarks()
    example_convenience_functions()
    example_advanced_filtering()
    
    print("Example usage completed!")