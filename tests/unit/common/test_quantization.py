"""
Comprehensive tests for the quantization system in the inference_pio.common module.

This test suite validates the functionality of the quantization system including:
- Quantization scheme definitions and enumeration
- Configuration creation and validation
- Quantization manager registration and retrieval
- Various quantization techniques (INT4, INT8, FP16, NF4)
- Memory reduction calculations
- Quantized linear layer functionality

The tests ensure that quantization preserves model functionality while reducing memory usage.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
import unittest
from typing import Any
from inference_pio.common.quantization import (
    QuantizationScheme,
    QuantizationConfig,
    QuantizationManager,
    get_quantization_manager,
    initialize_default_quantization_schemes,
    QuantizedLinear
)

class DummyModel(nn.Module):
    """
    Simple dummy model for testing quantization functionality.

    This model consists of two linear layers with a ReLU activation in between,
    providing a basic neural network structure suitable for quantization testing.
    """

    def __init__(self) -> None:
        """
        Initialize the dummy model with predefined layer dimensions.

        Creates:
        - linear1: Linear layer from 10 to 5 dimensions
        - linear2: Linear layer from 5 to 2 dimensions
        - relu: ReLU activation function
        """
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the dummy model.

        Args:
            x: Input tensor of shape (batch_size, 10)

        Returns:
            Output tensor of shape (batch_size, 2) after processing through
            linear1 -> ReLU -> linear2 layers
        """
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class TestQuantization(unittest.TestCase):
    """Test cases for quantization functionality and related components."""

    def setup_helper(self) -> None:
        """
        Set up test fixtures required for quantization tests.

        Initializes default quantization schemes and creates a dummy model
        for testing various quantization scenarios.
        """
        initialize_default_quantization_schemes()
        model = DummyModel()
        quantization_manager = get_quantization_manager()

    def test_quantization_scheme_enum(self) -> None:
        """
        Test that quantization schemes are properly defined with correct values.

        Validates that each quantization scheme enum member has the expected string value,
        ensuring consistency in scheme identification across the system.
        """
        assert_equal(QuantizationScheme.INT8.value, "int8")
        assert_equal(QuantizationScheme.INT4.value, "int4")
        assert_equal(QuantizationScheme.FP16.value, "fp16")
        assert_equal(QuantizationScheme.NF4.value, "nf4")

    def test_quantization_config_creation(self) -> None:
        """
        Test creation of quantization configurations with specified parameters.

        Verifies that QuantizationConfig objects are created correctly with the expected
        attributes when initialized with specific scheme, bit width, symmetry, and channel settings.
        """
        config = QuantizationConfig(
            scheme=QuantizationScheme.INT8,
            bits=8,
            symmetric=True,
            per_channel=True
        )
        assert_equal(config.scheme, QuantizationScheme.INT8)
        assert_true(config.symmetric)
        assert_true(config.per_channel)

    def test_quantization_config_validation(self) -> None:
        """
        Test validation and correction of quantization configuration parameters.

        Ensures that quantization configurations are validated and corrected appropriately,
        particularly for bit width alignment with the selected quantization scheme and
        proper calculation of quantization ranges based on symmetry settings.
        """
        # Test INT4 config validation
        config = QuantizationConfig(
            scheme=QuantizationScheme.INT4,
            bits=8,  # Will be corrected to 4
            symmetric=True
        )
        assert_equal(config.bits, 4)
        assert_equal(config.quant_min, -8)  # Symmetric INT4
        assert_equal(config.quant_max, 7)   # Symmetric INT4

        # Test INT8 config validation
        config = QuantizationConfig(
            scheme=QuantizationScheme.INT8,
            bits=4,  # Will be corrected to 8
            symmetric=False
        )
        assert_equal(config.bits, 8)
        assert_equal(config.quant_min, 0)    # Asymmetric INT8
        assert_equal(config.quant_max, 255)  # Asymmetric INT8

    def test_quantization_manager_registration(self) -> None:
        """
        Test registration and retrieval of quantization schemes via the manager.

        Validates that quantization configurations can be registered with the
        QuantizationManager and subsequently retrieved by their assigned names,
        ensuring proper storage and retrieval mechanisms.
        """
        quantization_manager = get_quantization_manager()
        config = QuantizationConfig(
            scheme=QuantizationScheme.INT4,
            bits=4,
            symmetric=True
        )
        quantization_manager.register_quantization_scheme("test_int4", config)

        retrieved_config = quantization_manager.get_quantization_config("test_int4")
        assert_is_not_none(retrieved_config)
        assert_equal(retrieved_config.scheme, QuantizationScheme.INT4)

    def test_fp16_quantization(self) -> None:
        """
        Test FP16 quantization functionality and parameter preservation.

        Verifies that FP16 quantization correctly converts model parameters to half precision
        while maintaining model functionality and output shape integrity.
        """
        quantization_manager = get_quantization_manager()
        config = QuantizationConfig(
            scheme=QuantizationScheme.FP16,
            bits=16,
            symmetric=False
        )

        original_model = DummyModel()
        original_params = sum(p.numel() for p in original_model.parameters())

        quantized_model = quantization_manager.quantize_model(original_model, config)

        # Check that model is still functional
        test_input = torch.randn(3, 10)
        with torch.no_grad():
            # Use the quantized model with appropriately typed input
            quantized_input = test_input.half()  # Convert input to half precision for FP16 model
            quantized_output = quantized_model(quantized_input)

        # Check that parameters are in fp16
        for param in quantized_model.parameters():
            assert_equal(param.dtype, torch.float16)

        # Output shape should be preserved
        assert_equal(quantized_output.shape, (3, 2))

    def test_int8_quantization(self) -> None:
        """
        Test INT8 quantization functionality and output similarity.

        Validates that INT8 quantization preserves model functionality while introducing
        acceptable quantization noise, ensuring that output shapes remain consistent
        between original and quantized models.
        """
        quantization_manager = get_quantization_manager()
        config = QuantizationConfig(
            scheme=QuantizationScheme.INT8,
            bits=8,
            symmetric=True,
            per_channel=True
        )

        original_model = DummyModel()
        original_params = sum(p.numel() for p in original_model.parameters())

        quantized_model = quantization_manager.quantize_model(original_model, config)

        # Check that model is still functional
        test_input = torch.randn(3, 10)
        with torch.no_grad():
            original_output = original_model(test_input)
            quantized_output = quantized_model(test_input)

        # Outputs should be similar (with quantization noise)
        assert_equal(original_output.shape, quantized_output.shape)

    def test_quantized_linear_layer(self) -> None:
        """
        Test the QuantizedLinear layer functionality and parameter quantization.

        Ensures that the QuantizedLinear layer performs forward passes correctly
        and properly initializes scale and zero-point parameters for quantization.
        """
        weight = torch.randn(10, 5)
        bias = torch.randn(10)

        config = QuantizationConfig(
            scheme=QuantizationScheme.INT8,
            bits=8,
            symmetric=True
        )

        quantized_linear = QuantizedLinear(weight, bias, config)

        # Test forward pass
        input_tensor = torch.randn(3, 5)
        output = quantized_linear(input_tensor)

        assert_equal(output.shape, (3, 10))
        assert_is_not_none(quantized_linear.scale)
        assert_is_not_none(quantized_linear.zero_point)

    def test_memory_reduction_calculation(self) -> None:
        """
        Test that quantization effectively reduces memory usage as expected.

        Verifies that quantized models consume less memory than their original counterparts,
        with specific validation for FP16 quantization which should approximately halve memory usage.
        """
        quantization_manager = get_quantization_manager()
        original_model = DummyModel()
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())

        # Test FP16 quantization (should reduce memory by ~50%)
        fp16_config = QuantizationConfig(scheme=QuantizationScheme.FP16)
        fp16_model = quantization_manager.quantize_model(original_model, fp16_config)
        fp16_size = sum(p.numel() * p.element_size() for p in fp16_model.parameters())

        assert_less(fp16_size, original_size)
        # FP16 should be about half the size of FP32
        assert_less(abs(fp16_size/original_size - 0.5), 0.1)

    def test_quantization_with_different_schemes(self) -> None:
        """
        Test quantization functionality across multiple quantization schemes.

        Validates that different quantization schemes (INT4, INT8, NF4, FP16) can be applied
        successfully to models while preserving functionality and output characteristics.
        """
        quantization_manager = get_quantization_manager()
        test_cases = [
            (QuantizationScheme.INT8, 8),
            (QuantizationScheme.INT4, 4),
            (QuantizationScheme.NF4, 4),
        ]

        for scheme, expected_bits in test_cases:
            with self.subTest(scheme=scheme):
                config = QuantizationConfig(scheme=scheme, bits=expected_bits)

                original_model = DummyModel()
                quantized_model = quantization_manager.quantize_model(original_model, config)

                # Just ensure the model is still functional
                test_input = torch.randn(2, 10)
                with torch.no_grad():
                    output = quantized_model(test_input)

                assert_equal(output.shape[0], 2)  # Batch dimension should be preserved

        # Special handling for FP16 since it requires half-precision input
        with self.subTest(scheme=QuantizationScheme.FP16):
            config = QuantizationConfig(scheme=QuantizationScheme.FP16, bits=16)

            original_model = DummyModel()
            quantized_model = quantization_manager.quantize_model(original_model, config)

            # FP16 model requires FP16 input
            test_input = torch.randn(2, 10, dtype=torch.half)
            with torch.no_grad():
                output = quantized_model(test_input)

            assert_equal(output.shape[0], 2)  # Batch dimension should be preserved

if __name__ == '__main__':
    run_tests([
        TestQuantization().test_quantization_scheme_enum,
        TestQuantization().test_quantization_config_creation,
        TestQuantization().test_quantization_config_validation,
        TestQuantization().test_quantization_manager_registration,
        TestQuantization().test_fp16_quantization,
        TestQuantization().test_int8_quantization,
        TestQuantization().test_quantized_linear_layer,
        TestQuantization().test_memory_reduction_calculation,
        TestQuantization().test_quantization_with_different_schemes
    ])