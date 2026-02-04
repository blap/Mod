import sys
import unittest
from unittest.mock import MagicMock, patch
import os

# Mock dependencies
sys.modules["torch"] = MagicMock()
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

# Mock inference_pio.benchmarks.scripts.standardized_runner
mock_runner = MagicMock()
sys.modules["inference_pio.benchmarks.scripts.standardized_runner"] = mock_runner

sys.path.insert(0, os.path.abspath("src"))

from inference_pio.__main__ import main

class TestCLI(unittest.TestCase):

    @patch('inference_pio.core.model_factory.ModelFactory.list_supported_models')
    def test_list(self, mock_list):
        mock_list.return_value = ["model1", "model2"]
        sys.argv = ["inference-pio", "list"]
        try:
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        # We can't easily capture stdout here without more mocking, but assuming it runs without error is good.

    @patch('inference_pio.__main__.create_model')
    def test_run(self, mock_create):
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        mock_model.generate_text.return_value = "Result"

        sys.argv = ["inference-pio", "run", "--model", "test-model", "--prompt", "test"]
        try:
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)

        mock_create.assert_called_with("test-model")
        mock_model.initialize.assert_called_once()
        mock_model.generate_text.assert_called_with("test")

    @patch('inference_pio.__main__.create_model')
    def test_info(self, mock_create):
        mock_model = MagicMock()
        mock_create.return_value = mock_model
        mock_model.metadata.name = "Test Model"
        mock_model.get_model_info.return_value = {"key": "value"}

        sys.argv = ["inference-pio", "info", "--model", "test-model"]
        try:
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)

        mock_create.assert_called_with("test-model")

    def test_benchmark(self):
        sys.argv = ["inference-pio", "benchmark", "--suite", "performance"]
        try:
            main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)

        mock_runner.run_standardized_benchmarks.assert_called_with(benchmark_suite="performance")

if __name__ == '__main__':
    unittest.main()
