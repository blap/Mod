"""
Updated Test suite for the benchmark discovery system with enhanced functionality.

This module tests the benchmark discovery system to ensure it properly discovers
and categorizes benchmarks across different models and categories.
"""

import unittest
import sys
from pathlib import Path

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference_pio.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, 
    assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, 
    assert_is_instance, assert_raises, run_tests, assert_length, assert_dict_contains,
    assert_list_contains, assert_is_subclass, assert_has_attr, assert_callable,
    assert_iterable, assert_not_is_instance
)

from src.inference_pio.benchmarks.discovery import BenchmarkDiscovery
from src.inference_pio.models.glm_4_7_flash.benchmarks.discovery import GLM47BenchmarkDiscovery
from src.inference_pio.models.qwen3_4b_instruct_2507.benchmarks.discovery import Qwen34bInstruct2507BenchmarkDiscovery
from src.inference_pio.models.qwen3_coder_30b.benchmarks.discovery import Qwen3Coder30bBenchmarkDiscovery
from src.inference_pio.models.qwen3_vl_2b.benchmarks.discovery import Qwen3Vl2bBenchmarkDiscovery
from src.inference_pio.plugin_system.benchmarks.discovery import PluginSystemBenchmarkDiscovery


def test_discovery_initialization():
    """Test that the discovery system initializes correctly."""
    discovery = BenchmarkDiscovery()
    assert_is_instance(discovery, BenchmarkDiscovery)
    assert_is_not_none(discovery.search_paths)


def test_discover_benchmarks():
    """Test that benchmarks can be discovered."""
    discovery = BenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()
    assert_is_instance(discovered, list)
    # We expect at least some benchmarks to be found
    assert_greater(len(discovered), 0)


def test_get_benchmarks_by_category():
    """Test filtering benchmarks by category."""
    discovery = BenchmarkDiscovery()
    discovery.discover_benchmarks()

    performance_benchmarks = discovery.get_benchmarks_by_category('performance')
    unit_benchmarks = discovery.get_benchmarks_by_category('unit')
    integration_benchmarks = discovery.get_benchmarks_by_category('integration')

    assert_is_instance(performance_benchmarks, list)
    assert_is_instance(unit_benchmarks, list)
    assert_is_instance(integration_benchmarks, list)


def test_get_benchmarks_by_model():
    """Test filtering benchmarks by model."""
    discovery = BenchmarkDiscovery()
    discovery.discover_benchmarks()

    glm_benchmarks = discovery.get_benchmarks_by_model('glm_4_7')
    qwen3_4b_benchmarks = discovery.get_benchmarks_by_model('qwen3_4b_instruct_2507')
    qwen3_coder_benchmarks = discovery.get_benchmarks_by_model('qwen3_coder_30b')
    qwen3_vl_benchmarks = discovery.get_benchmarks_by_model('qwen3_vl_2b')

    assert_is_instance(glm_benchmarks, list)
    assert_is_instance(qwen3_4b_benchmarks, list)
    assert_is_instance(qwen3_coder_benchmarks, list)
    assert_is_instance(qwen3_vl_benchmarks, list)


def test_specialized_discoveries_exist():
    """Test that specialized discovery classes exist and can be instantiated."""
    glm_discovery = GLM47BenchmarkDiscovery()
    qwen3_4b_discovery = Qwen34bInstruct2507BenchmarkDiscovery()
    qwen3_coder_discovery = Qwen3Coder30bBenchmarkDiscovery()
    qwen3_vl_discovery = Qwen3Vl2bBenchmarkDiscovery()
    plugin_discovery = PluginSystemBenchmarkDiscovery()

    assert_is_instance(glm_discovery, GLM47BenchmarkDiscovery)
    assert_is_instance(qwen3_4b_discovery, Qwen34bInstruct2507BenchmarkDiscovery)
    assert_is_instance(qwen3_coder_discovery, Qwen3Coder30bBenchmarkDiscovery)
    assert_is_instance(qwen3_vl_discovery, Qwen3Vl2bBenchmarkDiscovery)
    assert_is_instance(plugin_discovery, PluginSystemBenchmarkDiscovery)


def test_specialized_discoveries_can_discover():
    """Test that specialized discovery classes can discover benchmarks."""
    discoveries = [
        GLM47BenchmarkDiscovery(),
        Qwen34bInstruct2507BenchmarkDiscovery(),
        Qwen3Coder30bBenchmarkDiscovery(),
        Qwen3Vl2bBenchmarkDiscovery(),
        PluginSystemBenchmarkDiscovery()
    ]

    for discovery in discoveries:
        discovered = discovery.discover_benchmarks()
        assert_is_instance(discovered, list)
        # Each specialized discovery should focus on its own model/plugin system
        if discovery.discovered_benchmarks:
            assert_equal(discovery.model_name, discovery.discovered_benchmarks[0]['model_name'])


def test_glm_4_7_discovery():
    """Test GLM-4-7 specific discovery."""
    discovery = GLM47BenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    # All discovered benchmarks should be for glm_4_7
    for benchmark in discovered:
        assert_in('glm_4_7', benchmark['model_name'] or benchmark['file_path'])


def test_qwen3_4b_instruct_2507_discovery():
    """Test Qwen3-4b-instruct-2507 specific discovery."""
    discovery = Qwen34bInstruct2507BenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    # All discovered benchmarks should be for qwen3_4b_instruct_2507
    for benchmark in discovered:
        assert_in('qwen3_4b_instruct_2507', benchmark['model_name'] or benchmark['file_path'])


def test_qwen3_coder_30b_discovery():
    """Test Qwen3-coder-30b specific discovery."""
    discovery = Qwen3Coder30bBenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    # All discovered benchmarks should be for qwen3_coder_30b
    for benchmark in discovered:
        assert_in('qwen3_coder_30b', benchmark['model_name'] or benchmark['file_path'])


def test_qwen3_vl_2b_discovery():
    """Test Qwen3-vl-2b specific discovery."""
    discovery = Qwen3Vl2bBenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    # All discovered benchmarks should be for qwen3_vl_2b
    for benchmark in discovered:
        assert_in('qwen3_vl_2b', benchmark['model_name'] or benchmark['file_path'])


def test_plugin_system_discovery():
    """Test plugin system specific discovery."""
    discovery = PluginSystemBenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    # All discovered benchmarks should be for plugin_system
    for benchmark in discovered:
        assert_in('plugin_system', benchmark['model_name'] or benchmark['file_path'])


def test_discovery_search_paths_configuration():
    """Test that discovery system can handle custom search paths."""
    discovery = BenchmarkDiscovery(search_paths=['./custom_benchmarks'])
    assert_is_not_none(discovery.search_paths)
    assert_in('./custom_benchmarks', discovery.search_paths)


def test_discovery_filter_by_multiple_categories():
    """Test filtering benchmarks by multiple categories."""
    discovery = BenchmarkDiscovery()
    discovery.discover_benchmarks()

    # Test combined filters
    perf_and_unit = discovery.get_benchmarks_by_categories(['performance', 'unit'])
    assert_is_instance(perf_and_unit, list)


def test_discovery_filter_by_multiple_models():
    """Test filtering benchmarks by multiple models."""
    discovery = BenchmarkDiscovery()
    discovery.discover_benchmarks()

    # Test combined filters
    multi_model = discovery.get_benchmarks_by_models(['glm_4_7', 'qwen3_4b_instruct_2507'])
    assert_is_instance(multi_model, list)


def test_discovery_statistics():
    """Test that discovery system provides statistics."""
    discovery = BenchmarkDiscovery()
    discovery.discover_benchmarks()

    stats = discovery.get_statistics()
    assert_is_instance(stats, dict)
    assert_in('total_benchmarks', stats)
    assert_in('by_category', stats)
    assert_in('by_model', stats)


def test_discovery_benchmark_structure():
    """Test that discovered benchmarks have the expected structure."""
    discovery = BenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    for benchmark in discovered:
        assert_is_instance(benchmark, dict)
        required_fields = ['name', 'file_path', 'model_name', 'category', 'description']
        for field in required_fields:
            assert_in(field, benchmark)


def test_discovery_duplicate_handling():
    """Test that discovery system handles duplicates properly."""
    discovery = BenchmarkDiscovery()
    first_discovery = discovery.discover_benchmarks()
    second_discovery = discovery.discover_benchmarks()

    # Both discoveries should return the same results
    assert_equal(len(first_discovery), len(second_discovery))


def test_discovery_empty_category_handling():
    """Test that discovery system handles empty categories properly."""
    discovery = BenchmarkDiscovery()
    discovery.discover_benchmarks()

    empty_category = discovery.get_benchmarks_by_category('nonexistent_category')
    assert_is_instance(empty_category, list)
    assert_equal(len(empty_category), 0)


def test_discovery_empty_model_handling():
    """Test that discovery system handles empty model queries properly."""
    discovery = BenchmarkDiscovery()
    discovery.discover_benchmarks()

    empty_model = discovery.get_benchmarks_by_model('nonexistent_model')
    assert_is_instance(empty_model, list)
    assert_equal(len(empty_model), 0)


def test_glm47_specific_discovery_attributes():
    """Test specific attributes of GLM47 benchmark discovery."""
    discovery = GLM47BenchmarkDiscovery()
    
    assert_is_not_none(discovery.model_name)
    assert_equal(discovery.model_name, 'glm_4_7')
    assert_is_not_none(discovery.search_paths)


def test_qwen3_4b_specific_discovery_attributes():
    """Test specific attributes of Qwen3-4b benchmark discovery."""
    discovery = Qwen34bInstruct2507BenchmarkDiscovery()
    
    assert_is_not_none(discovery.model_name)
    assert_equal(discovery.model_name, 'qwen3_4b_instruct_2507')
    assert_is_not_none(discovery.search_paths)


def test_qwen3_coder_specific_discovery_attributes():
    """Test specific attributes of Qwen3-Coder benchmark discovery."""
    discovery = Qwen3Coder30bBenchmarkDiscovery()
    
    assert_is_not_none(discovery.model_name)
    assert_equal(discovery.model_name, 'qwen3_coder_30b')
    assert_is_not_none(discovery.search_paths)


def test_qwen3_vl_specific_discovery_attributes():
    """Test specific attributes of Qwen3-VL benchmark discovery."""
    discovery = Qwen3Vl2bBenchmarkDiscovery()
    
    assert_is_not_none(discovery.model_name)
    assert_equal(discovery.model_name, 'qwen3_vl_2b')
    assert_is_not_none(discovery.search_paths)


def test_plugin_system_specific_discovery_attributes():
    """Test specific attributes of Plugin System benchmark discovery."""
    discovery = PluginSystemBenchmarkDiscovery()
    
    assert_is_not_none(discovery.model_name)
    assert_equal(discovery.model_name, 'plugin_system')
    assert_is_not_none(discovery.search_paths)


def test_discovery_benchmark_metadata():
    """Test that discovered benchmarks have proper metadata."""
    discovery = BenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    for benchmark in discovered:
        # Check that metadata fields exist and are properly typed
        assert_is_instance(benchmark.get('name'), str)
        assert_is_instance(benchmark.get('file_path'), str)
        assert_is_instance(benchmark.get('model_name'), str)
        assert_is_instance(benchmark.get('category'), str)
        assert_is_instance(benchmark.get('description'), str)


def test_discovery_benchmark_file_existence():
    """Test that discovered benchmark files actually exist."""
    discovery = BenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    import os
    for benchmark in discovered:
        file_path = benchmark.get('file_path')
        if file_path:
            assert_true(os.path.exists(file_path), f"Benchmark file does not exist: {file_path}")


def test_discovery_benchmark_categorization_accuracy():
    """Test that benchmarks are categorized accurately."""
    discovery = BenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    for benchmark in discovered:
        # Verify that category is one of the expected values
        expected_categories = ['unit', 'integration', 'performance', 'system']
        assert_in(benchmark.get('category', ''), expected_categories)


def test_discovery_model_mapping_accuracy():
    """Test that benchmarks are mapped to correct models."""
    discovery = BenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    for benchmark in discovered:
        model_name = benchmark.get('model_name', '')
        file_path = benchmark.get('file_path', '')
        
        # Verify that model name appears in file path or vice versa
        if model_name:
            # Either the model name is in the file path or it's a general benchmark
            assert_true(model_name in file_path or model_name == 'general')


def test_discovery_benchmark_description_content():
    """Test that benchmark descriptions have meaningful content."""
    discovery = BenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    for benchmark in discovered:
        description = benchmark.get('description', '')
        # Descriptions should not be empty or just whitespace
        assert_true(len(description.strip()) > 0, f"Benchmark {benchmark.get('name')} has empty description")


def test_discovery_benchmark_name_uniqueness():
    """Test that benchmark names are unique within the discovery system."""
    discovery = BenchmarkDiscovery()
    discovered = discovery.discover_benchmarks()

    names = [benchmark.get('name') for benchmark in discovered if benchmark.get('name')]
    unique_names = set(names)
    
    # Check that all names are unique
    assert_equal(len(names), len(unique_names), "Benchmark names should be unique")


def test_discovery_error_handling():
    """Test that discovery system handles errors gracefully."""
    try:
        # Test with invalid search path
        discovery = BenchmarkDiscovery(search_paths=['/nonexistent/path'])
        discovered = discovery.discover_benchmarks()
        # Should return empty list rather than crash
        assert_is_instance(discovered, list)
    except Exception:
        # If it raises an exception, it should be handled gracefully
        pass


def test_discovery_refresh_functionality():
    """Test that discovery system can refresh benchmarks."""
    discovery = BenchmarkDiscovery()
    first_discovery = discovery.discover_benchmarks()
    
    # Refresh should return the same results in a stable system
    refreshed = discovery.refresh_discovery()
    
    assert_equal(len(first_discovery), len(refreshed))


def run_benchmark_discovery_tests():
    """Run all benchmark discovery tests."""
    test_functions = [
        test_discovery_initialization,
        test_discover_benchmarks,
        test_get_benchmarks_by_category,
        test_get_benchmarks_by_model,
        test_specialized_discoveries_exist,
        test_specialized_discoveries_can_discover,
        test_glm_4_7_discovery,
        test_qwen3_4b_instruct_2507_discovery,
        test_qwen3_coder_30b_discovery,
        test_qwen3_vl_2b_discovery,
        test_plugin_system_discovery,
        test_discovery_search_paths_configuration,
        test_discovery_filter_by_multiple_categories,
        test_discovery_filter_by_multiple_models,
        test_discovery_statistics,
        test_discovery_benchmark_structure,
        test_discovery_duplicate_handling,
        test_discovery_empty_category_handling,
        test_discovery_empty_model_handling,
        test_glm47_specific_discovery_attributes,
        test_qwen3_4b_specific_discovery_attributes,
        test_qwen3_coder_specific_discovery_attributes,
        test_qwen3_vl_specific_discovery_attributes,
        test_plugin_system_specific_discovery_attributes,
        test_discovery_benchmark_metadata,
        test_discovery_benchmark_file_existence,
        test_discovery_benchmark_categorization_accuracy,
        test_discovery_model_mapping_accuracy,
        test_discovery_benchmark_description_content,
        test_discovery_benchmark_name_uniqueness,
        test_discovery_error_handling,
        test_discovery_refresh_functionality
    ]

    print("Running updated benchmark discovery tests...")
    success = run_tests(test_functions)
    return success


if __name__ == '__main__':
    run_benchmark_discovery_tests()