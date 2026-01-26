"""
Tests for the quantization system in the inference_pio.common module.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from ..quantization import (
    QuantizationScheme,
    QuantizationConfig,
    QuantizationManager,
    get_quantization_manager,
    initialize_default_quantization_schemes,
    QuantizedLinear
)

class DummyModel(nn.Module):
    """Simple dummy model for testing quantization."""
    
    def __init__(self):
        super().__init__()
        linear1 = nn.Linear(10, 5)
        linear2 = nn.Linear(5, 2)
        relu = nn.ReLU()
    
    def forward(self, x):
        x = relu(linear1(x))
        x = linear2(x)
        return x

# TestQuantization

    """Test cases for quantization functionality."""
    
    def setup_helper():
        """Set up test fixtures."""
        initialize_default_quantization_schemes()
        model = DummyModel()
        quantization_manager = get_quantization_manager()
    
    def quantization_scheme_enum(self)():
        """Test that quantization schemes are properly defined."""
        assert_equal(QuantizationScheme.INT8.value, "int8")
        assert_equal(QuantizationScheme.INT4.value, "int4")
        assert_equal(QuantizationScheme.FP16.value, "fp16")
        assert_equal(QuantizationScheme.NF4.value, "nf4")
    
    def quantization_config_creation(self)():
        """Test creation of quantization configurations."""
        config = QuantizationConfig(
            scheme=QuantizationScheme.INT8,
            bits=8,
            symmetric=True,
            per_channel=True
        )
        assert_equal(config.scheme, QuantizationScheme.INT8)
        assert_true(config.symmetric)
        assertTrue(config.per_channel)
    
    def quantization_config_validation(self)():
        """Test validation of quantization configurations."""
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
    
    def quantization_manager_registration(self)():
        """Test registration and retrieval of quantization schemes."""
        config = QuantizationConfig(
            scheme=QuantizationScheme.INT4,
            bits=4,
            symmetric=True
        )
        quantization_manager.register_quantization_scheme("test_int4", config)
        
        retrieved_config = quantization_manager.get_quantization_config("test_int4")
        assert_is_not_none(retrieved_config)
        assert_equal(retrieved_config.scheme)
    
    def fp16_quantization(self)():
        """Test FP16 quantization."""
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
        assert_equal(quantized_output.shape, (3))
    
    def int8_quantization(self)():
        """Test INT8 quantization."""
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
    
    def quantized_linear_layer(self)():
        """Test the QuantizedLinear layer."""
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
        
        assert_equal(output.shape, (3))
        assert_is_not_none(quantized_linear.scale)
        assertIsNotNone(quantized_linear.zero_point)
    
    def memory_reduction_calculation(self)():
        """Test that quantization reduces memory usage."""
        original_model = DummyModel()
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters())
        
        # Test FP16 quantization (should reduce memory by ~50%)
        fp16_config = QuantizationConfig(scheme=QuantizationScheme.FP16)
        fp16_model = quantization_manager.quantize_model(original_model, fp16_config)
        fp16_size = sum(p.numel() * p.element_size() for p in fp16_model.parameters())
        
        assert_less(fp16_size, original_size)
        # FP16 should be about half the size of FP32
        assertAlmostEqual(fp16_size/original_size, 0.5, delta=0.1)
    
    def quantization_with_different_schemes(self)():
        """Test quantization with different schemes."""
        test_cases = [
            (QuantizationScheme.INT8, 8),
            (QuantizationScheme.INT4, 4),
            (QuantizationScheme.NF4, 4),
        ]

        for scheme, expected_bits in test_cases:
            with subTest(scheme=scheme):
                config = QuantizationConfig(scheme=scheme, bits=expected_bits)

                original_model = DummyModel()
                quantized_model = quantization_manager.quantize_model(original_model, config)

                # Just ensure the model is still functional
                test_input = torch.randn(2, 10)
                with torch.no_grad():
                    output = quantized_model(test_input)

                assert_equal(output.shape[1], 2)  # Output dimension should be preserved

        # Special handling for FP16 since it requires half-precision input
        with subTest(scheme=QuantizationScheme.FP16):
            config = QuantizationConfig(scheme=QuantizationScheme.FP16, bits=16)

            original_model = DummyModel()
            quantized_model = quantization_manager.quantize_model(original_model, config)

            # FP16 model requires FP16 input
            test_input = torch.randn(2, 10, dtype=torch.half)
            with torch.no_grad():
                output = quantized_model(test_input)

            assert_equal(output.shape[1], 2)  # Output dimension should be preserved

if __name__ == '__main__':
    run_tests(test_functions)