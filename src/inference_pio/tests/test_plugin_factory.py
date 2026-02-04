
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock torch and psutil before importing anything else
mock_torch = MagicMock()
mock_torch.Tensor = MagicMock
mock_torch.nn.functional = MagicMock()
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = mock_torch.nn.functional

mock_psutil = MagicMock()
sys.modules['psutil'] = mock_psutil

sys.modules['intel_extension_for_pytorch'] = MagicMock()

# Now we can safely import our modules
# We might need to mock other things that are imported at top level
# src.inference_pio.__init__ imports things?
# Let's bypass src.inference_pio.__init__ if it causes trouble by mocking it or its deps.
# The previous error showed __init__.py importing base_attention.py importing torch.
# Since torch is mocked, it should be fine.

from src.inference_pio.common.hardware.hardware_analyzer import SystemProfile
# We need to import the factory.
# But factory imports plugins, plugins import plugin_interface, interface imports torch.
# Should be fine.

from src.inference_pio.plugins.factory import get_processor_plugin
from src.inference_pio.plugins.intel.intel_kaby_lake_plugin import IntelKabyLakePlugin
from src.inference_pio.plugins.amd.amd_ryzen_plugin import AmdRyzenPlugin
from src.inference_pio.plugins.intel.intel_comet_lake_plugin import IntelCometLakePlugin
from src.inference_pio.plugins.cpu.cpu_plugin import GenericCPUPlugin

class TestPluginFactory(unittest.TestCase):
    def create_mock_profile(self, cpu_brand):
        return SystemProfile(
            cpu_cores_physical=4,
            cpu_cores_logical=8,
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            total_vram_gb=4.0,
            available_vram_gb=3.0,
            gpu_name="TestGPU",
            disk_free_gb=100.0,
            is_weak_hardware=False,
            recommended_offload_strategy="none",
            safe_vram_limit_gb=3.0,
            processor_architecture="x86_64",
            cpu_brand=cpu_brand,
            instruction_sets=["AVX2"],
            secondary_gpu_detected="None",
            hybrid_capability=False,
            ipex_available=False
        )

    def test_kaby_lake_detection(self):
        profile = self.create_mock_profile("Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz")
        plugin = get_processor_plugin(profile)
        self.assertIsInstance(plugin, IntelKabyLakePlugin)

    def test_ryzen_detection(self):
        profile = self.create_mock_profile("AMD Ryzen 7 5700")
        plugin = get_processor_plugin(profile)
        self.assertIsInstance(plugin, AmdRyzenPlugin)

    def test_generic_ryzen_detection(self):
        profile = self.create_mock_profile("AMD Ryzen 5 5600X")
        plugin = get_processor_plugin(profile)
        self.assertIsInstance(plugin, AmdRyzenPlugin)

    def test_comet_lake_detection(self):
        profile = self.create_mock_profile("Intel(R) Core(TM) i5-10210U CPU")
        plugin = get_processor_plugin(profile)
        self.assertIsInstance(plugin, IntelCometLakePlugin)

    def test_generic_fallback(self):
        profile = self.create_mock_profile("Generic Intel CPU")
        plugin = get_processor_plugin(profile)
        self.assertIsInstance(plugin, GenericCPUPlugin)

if __name__ == "__main__":
    unittest.main()
