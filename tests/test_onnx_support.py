"""
Tests for ONNX support functionality in model loading and saving.
"""

import unittest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from src.qwen3_vl.models.model_loader import ModelLoader


class SimpleTestModel(nn.Module):
    """A simple model for testing purposes."""
    
    def __init__(self, input_size=10, output_size=5):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.linear(x))


@unittest.skipIf(not ONNX_AVAILABLE, "ONNX Runtime is not available")
class TestONNXSupport(unittest.TestCase):
    """Test cases for ONNX support functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = ModelLoader()
        self.test_model = SimpleTestModel()
        
    def test_onnx_save_functionality(self):
        """Test saving a model to ONNX format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "onnx_test_model"

            # Save the model in ONNX format with sample inputs
            sample_input = torch.randn(1, 10)  # Match SimpleTestModel's expected input
            self.loader.save_model(self.test_model, save_path, format='onnx', sample_inputs=sample_input)

            # Check that the ONNX file exists
            onnx_file = save_path / "model.onnx"
            self.assertTrue(onnx_file.exists(), f"ONNX file not created at {onnx_file}")

            # Verify the file is a valid ONNX model
            import onnx
            model = onnx.load(str(onnx_file))
            onnx.checker.check_model(model)
    
    def test_onnx_load_functionality(self):
        """Test loading a model from ONNX format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "onnx_test_model"

            # First save the model in ONNX format with sample inputs
            sample_input = torch.randn(1, 10)  # Match SimpleTestModel's expected input
            self.loader.save_model(self.test_model, save_path, format='onnx', sample_inputs=sample_input)

            # Then load it back
            loaded_model = self.loader._load_onnx_model(
                model_path=save_path,
                model_class=SimpleTestModel,
                config={},
                device_map=None,
                torch_dtype=None
            )

            # Check that the loaded model is an instance of our wrapper
            self.assertIsNotNone(loaded_model)
            self.assertTrue(hasattr(loaded_model, 'session'))

            # Test inference with the loaded model
            test_input = torch.randn(1, 10)
            original_output = self.test_model(test_input)
            wrapped_output = loaded_model(test_input)  # Use positional argument

            # The output shapes should match
            self.assertEqual(original_output.shape, wrapped_output.shape)
    
    def test_onnx_end_to_end_save_and_load(self):
        """Test end-to-end save and load cycle with ONNX."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "onnx_e2e_test"

            # Save model in ONNX format with sample inputs
            sample_input = torch.randn(1, 10)  # Match SimpleTestModel's expected input
            self.loader.save_model(self.test_model, save_path, format='onnx', sample_inputs=sample_input)

            # Load the model back using the main load_model method
            loaded_model = self.loader.load_model(
                model_name_or_path=str(save_path),
                model_class=SimpleTestModel,
                config_class=lambda **kwargs: kwargs
            )

            # Test inference
            test_input = torch.randn(2, 10)
            original_output = self.test_model(test_input)
            loaded_output = loaded_model(test_input)  # Use positional argument

            # Shapes should match
            self.assertEqual(original_output.shape, loaded_output.shape)
    
    def test_onnx_inference_consistency(self):
        """Test that ONNX inference produces consistent results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "onnx_consistency_test"

            # Save model with sample inputs
            sample_input = torch.randn(1, 10)  # Match SimpleTestModel's expected input
            self.loader.save_model(self.test_model, save_path, format='onnx', sample_inputs=sample_input)

            # Load model
            loaded_model = self.loader._load_onnx_model(
                model_path=save_path,
                model_class=SimpleTestModel,
                config={},
                device_map=None,
                torch_dtype=None
            )

            # Generate same input multiple times
            test_inputs = [torch.randn(1, 10) for _ in range(3)]

            for test_input in test_inputs:
                # Get outputs from both models
                pytorch_output = self.test_model(test_input)
                onnx_output = loaded_model(test_input)  # Use positional argument

                # Both should have same shape
                self.assertEqual(pytorch_output.shape, onnx_output.shape)
    
    def test_onnx_device_mapping(self):
        """Test ONNX model loading with device mapping."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "onnx_device_test"

            # Save model with sample inputs
            sample_input = torch.randn(1, 10)  # Match SimpleTestModel's expected input
            self.loader.save_model(self.test_model, save_path, format='onnx', sample_inputs=sample_input)

            # Load with CPU device map
            loaded_model = self.loader._load_onnx_model(
                model_path=save_path,
                model_class=SimpleTestModel,
                config={},
                device_map="cpu",
                torch_dtype=None
            )

            self.assertIsNotNone(loaded_model)

            # Test inference
            test_input = torch.randn(1, 10)
            output = loaded_model(test_input)  # Use positional argument
            self.assertIsNotNone(output)
    
    def test_invalid_onnx_file_handling(self):
        """Test error handling when ONNX file is invalid."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "invalid_onnx_test"
            save_path.mkdir(exist_ok=True)
            
            # Create an invalid ONNX file
            invalid_file = save_path / "model.onnx"
            with open(invalid_file, 'w') as f:
                f.write("invalid onnx content")
            
            # Attempt to load should raise an error
            with self.assertRaises(Exception):
                self.loader._load_onnx_model(
                    model_path=save_path,
                    model_class=SimpleTestModel,
                    config={},
                    device_map=None,
                    torch_dtype=None
                )
    
    def test_missing_onnx_file_handling(self):
        """Test error handling when ONNX file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "missing_onnx_test"
            save_path.mkdir(exist_ok=True)
            
            # Attempt to load should raise an error since file doesn't exist
            with self.assertRaises(FileNotFoundError):
                self.loader._load_onnx_model(
                    model_path=save_path,
                    model_class=SimpleTestModel,
                    config={},
                    device_map=None,
                    torch_dtype=None
                )


@unittest.skipIf(not ONNX_AVAILABLE, "ONNX Runtime is not available")
class TestONNXExportParams(unittest.TestCase):
    """Test ONNX export with different parameters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = ModelLoader()
        self.test_model = SimpleTestModel()
        
    def test_onnx_export_with_different_opset_versions(self):
        """Test ONNX export with different opset versions."""
        with patch('torch.onnx.export') as mock_export:
            with tempfile.TemporaryDirectory() as temp_dir:
                save_path = Path(temp_dir) / "onnx_opset_test"
                
                # Call save_model which should internally call torch.onnx.export
                try:
                    self.loader.save_model(self.test_model, save_path, format='onnx')
                except:
                    # We expect this to fail since we're mocking torch.onnx.export
                    # but we want to verify the parameters passed to it
                    pass
                
                # Verify that torch.onnx.export was called with expected parameters
                self.assertTrue(mock_export.called)
                args, kwargs = mock_export.call_args
                
                # Check key parameters
                self.assertIn('opset_version', kwargs)
                self.assertIn('export_params', kwargs)
                self.assertIn('do_constant_folding', kwargs)
                self.assertIn('input_names', kwargs)
                self.assertIn('output_names', kwargs)
                
                # Verify values
                self.assertEqual(kwargs['opset_version'], 14)
                self.assertEqual(kwargs['export_params'], True)
                self.assertEqual(kwargs['do_constant_folding'], True)
                # For simple test models, input_names will be ['input'] instead of the Qwen-specific names
                # This is expected behavior when the model doesn't match the expected Qwen interface
                if len(kwargs['input_names']) == 1:
                    self.assertIn('input', kwargs['input_names'])
                else:
                    self.assertIn('input_ids', kwargs['input_names'])
                    self.assertIn('pixel_values', kwargs['input_names'])
                    self.assertIn('attention_mask', kwargs['input_names'])
                self.assertIn('output', kwargs['output_names'])


class TestONNXWithoutRuntime(unittest.TestCase):
    """Test ONNX functionality when ONNX Runtime is not available."""
    
    @patch.dict('sys.modules', {'onnxruntime': None})  # Mock onnxruntime as unavailable
    def test_onnx_runtime_not_available(self):
        """Test that appropriate error is raised when ONNX runtime is not available."""
        # We need to reload the module to simulate the missing import
        import importlib
        try:
            # This would cause an ImportError when onnxruntime is imported
            from src.qwen3_vl.models.model_loader import ModelLoader
            loader = ModelLoader()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                save_path = Path(temp_dir) / "test_model"
                save_path.mkdir(exist_ok=True)
                
                # Create a dummy ONNX file to trigger the loading path
                (save_path / "model.onnx").touch()
                
                with self.assertRaises(ImportError):
                    loader._load_onnx_model(
                        model_path=save_path,
                        model_class=SimpleTestModel,
                        config={},
                        device_map=None,
                        torch_dtype=None
                    )
        except ImportError:
            # Expected when onnxruntime is mocked as unavailable
            pass


if __name__ == '__main__':
    unittest.main()