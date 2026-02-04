import sys
import unittest
from unittest.mock import MagicMock, patch
import os

# Mock dependencies
sys.modules["torch"] = MagicMock()
sys.modules["torch"].__version__ = "2.0.0" # Fix for __version__ access
sys.modules["torch"].cuda = MagicMock()
sys.modules["torch"].cuda.is_available.return_value = False
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["accelerate"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["psutil"] = MagicMock()
sys.modules["cpuinfo"] = MagicMock()
sys.modules["GPUtil"] = MagicMock()
sys.modules["pynvml"] = MagicMock()
sys.modules["huggingface_hub"] = MagicMock()

# Mock inference_pio.benchmarks.scripts.standardized_runner
mock_runner = MagicMock()
sys.modules["inference_pio.benchmarks.scripts.standardized_runner"] = mock_runner

sys.path.insert(0, os.path.abspath("src"))

from inference_pio.__main__ import main
from inference_pio.core.model_loader import ModelLoader

class TestCLIPhase2(unittest.TestCase):

    @patch('inference_pio.core.tools.rich_utils.console.print')
    @patch('inference_pio.__main__.perform_system_check')
    def test_check(self, mock_perform_check, mock_print):
        mock_perform_check.return_value = {
            "os": "Test OS",
            "python_version": "3.10",
            "torch_version": "2.0",
            "cuda_available": False,
            "cpu": {"brand": "Test CPU", "cores_physical": 4, "cores_logical": 8, "usage_percent": 10},
            "memory": {"total_gb": 16, "available_gb": 8, "percent": 50},
            "gpus": [],
            "disks": {"h_drive_detected": True}
        }

        sys.argv = ["inference-pio", "check"]
        try:
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    @patch('inference_pio.core.tools.cleaner.console.print')
    def test_clean(self, mock_print):
        # Just verify it runs without crashing
        sys.argv = ["inference-pio", "clean"]
        try:
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)

    @patch('inference_pio.core.model_loader.ModelLoader.get_h_drive_base')
    def test_model_loader_h_drive(self, mock_get_h):
        mock_get_h.return_value = None # Simulate no H drive for test safety
        path = ModelLoader.resolve_model_path("test-model")
        # Should return model name if not found and no repo id
        self.assertEqual(path, "test-model")

if __name__ == '__main__':
    unittest.main()
