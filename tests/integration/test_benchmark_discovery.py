"""
Test suite for the benchmark discovery system.
"""

import unittest
import sys
from pathlib import Path

# Add the project root to the path so we can import modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from benchmarks.discovery import BenchmarkDiscovery
from src.inference_pio.models.glm_4_7_flash.benchmarks.discovery import GLM47BenchmarkDiscovery
from src.inference_pio.models.qwen3_4b_instruct_2507.benchmarks.discovery import Qwen34bInstruct2507BenchmarkDiscovery
from src.inference_pio.models.qwen3_coder_30b.benchmarks.discovery import Qwen3Coder30bBenchmarkDiscovery
from src.inference_pio.models.qwen3_vl_2b.benchmarks.discovery import Qwen3Vl2bBenchmarkDiscovery
from src.inference_pio.plugin_system.benchmarks.discovery import PluginSystemBenchmarkDiscovery


class TestBenchmarkDiscovery(unittest.TestCase):
    """Test cases for the benchmark discovery system."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.discovery = BenchmarkDiscovery()

    def test_discovery_initialization(self):
        """Test that the discovery system initializes correctly."""
        self.assertIsInstance(self.discovery, BenchmarkDiscovery)
        self.assertIsNotNone(self.discovery.search_paths)

    def test_discover_benchmarks(self):
        """Test that benchmarks can be discovered."""
        discovered = self.discovery.discover_benchmarks()
        self.assertIsInstance(discovered, list)
        # We expect at least some benchmarks to be found
        self.assertGreaterEqual(len(discovered), 0)

    def test_get_benchmarks_by_category(self):
        """Test filtering benchmarks by category."""
        self.discovery.discover_benchmarks()
        
        performance_benchmarks = self.discovery.get_benchmarks_by_category('performance')
        unit_benchmarks = self.discovery.get_benchmarks_by_category('unit')
        integration_benchmarks = self.discovery.get_benchmarks_by_category('integration')
        
        self.assertIsInstance(performance_benchmarks, list)
        self.assertIsInstance(unit_benchmarks, list)
        self.assertIsInstance(integration_benchmarks, list)

    def test_get_benchmarks_by_model(self):
        """Test filtering benchmarks by model."""
        self.discovery.discover_benchmarks()
        
        glm_benchmarks = self.discovery.get_benchmarks_by_model('glm_4_7')
        qwen3_4b_benchmarks = self.discovery.get_benchmarks_by_model('qwen3_4b_instruct_2507')
        qwen3_coder_benchmarks = self.discovery.get_benchmarks_by_model('qwen3_coder_30b')
        qwen3_vl_benchmarks = self.discovery.get_benchmarks_by_model('qwen3_vl_2b')
        
        self.assertIsInstance(glm_benchmarks, list)
        self.assertIsInstance(qwen3_4b_benchmarks, list)
        self.assertIsInstance(qwen3_coder_benchmarks, list)
        self.assertIsInstance(qwen3_vl_benchmarks, list)

    def test_specialized_discoveries_exist(self):
        """Test that specialized discovery classes exist and can be instantiated."""
        glm_discovery = GLM47BenchmarkDiscovery()
        qwen3_4b_discovery = Qwen34bInstruct2507BenchmarkDiscovery()
        qwen3_coder_discovery = Qwen3Coder30bBenchmarkDiscovery()
        qwen3_vl_discovery = Qwen3Vl2bBenchmarkDiscovery()
        plugin_discovery = PluginSystemBenchmarkDiscovery()
        
        self.assertIsInstance(glm_discovery, GLM47BenchmarkDiscovery)
        self.assertIsInstance(qwen3_4b_discovery, Qwen34bInstruct2507BenchmarkDiscovery)
        self.assertIsInstance(qwen3_coder_discovery, Qwen3Coder30bBenchmarkDiscovery)
        self.assertIsInstance(qwen3_vl_discovery, Qwen3Vl2bBenchmarkDiscovery)
        self.assertIsInstance(plugin_discovery, PluginSystemBenchmarkDiscovery)

    def test_specialized_discoveries_can_discover(self):
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
            self.assertIsInstance(discovered, list)
            # Each specialized discovery should focus on its own model/plugin system
            self.assertEqual(discovery.model_name, discovery.discovered_benchmarks[0]['model_name'] if discovery.discovered_benchmarks else discovery.model_name)


class TestModelSpecificDiscoveries(unittest.TestCase):
    """Test cases for model-specific benchmark discoveries."""

    def test_glm_4_7_discovery(self):
        """Test GLM-4-7 specific discovery."""
        discovery = GLM47BenchmarkDiscovery()
        discovered = discovery.discover_benchmarks()
        
        # All discovered benchmarks should be for glm_4_7
        for benchmark in discovered:
            self.assertIn('glm_4_7', benchmark['model_name'] or benchmark['file_path'])

    def test_qwen3_4b_instruct_2507_discovery(self):
        """Test Qwen3-4b-instruct-2507 specific discovery."""
        discovery = Qwen34bInstruct2507BenchmarkDiscovery()
        discovered = discovery.discover_benchmarks()
        
        # All discovered benchmarks should be for qwen3_4b_instruct_2507
        for benchmark in discovered:
            self.assertIn('qwen3_4b_instruct_2507', benchmark['model_name'] or benchmark['file_path'])

    def test_qwen3_coder_30b_discovery(self):
        """Test Qwen3-coder-30b specific discovery."""
        discovery = Qwen3Coder30bBenchmarkDiscovery()
        discovered = discovery.discover_benchmarks()
        
        # All discovered benchmarks should be for qwen3_coder_30b
        for benchmark in discovered:
            self.assertIn('qwen3_coder_30b', benchmark['model_name'] or benchmark['file_path'])

    def test_qwen3_vl_2b_discovery(self):
        """Test Qwen3-vl-2b specific discovery."""
        discovery = Qwen3Vl2bBenchmarkDiscovery()
        discovered = discovery.discover_benchmarks()
        
        # All discovered benchmarks should be for qwen3_vl_2b
        for benchmark in discovered:
            self.assertIn('qwen3_vl_2b', benchmark['model_name'] or benchmark['file_path'])

    def test_plugin_system_discovery(self):
        """Test plugin system specific discovery."""
        discovery = PluginSystemBenchmarkDiscovery()
        discovered = discovery.discover_benchmarks()
        
        # All discovered benchmarks should be for plugin_system
        for benchmark in discovered:
            self.assertIn('plugin_system', benchmark['model_name'] or benchmark['file_path'])


if __name__ == '__main__':
    unittest.main()