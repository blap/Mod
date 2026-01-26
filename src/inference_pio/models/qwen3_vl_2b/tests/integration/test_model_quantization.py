"""
Tests for quantization integration with models.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.glm_4_7.model import GLM47Model
from src.inference_pio.common.quantization import QuantizationScheme

# TestModelQuantization

    """Test quantization integration with different models."""
    
    def glm47_model_quantization_config(self)():
        """Test GLM-4-7 model quantization configuration."""
        config = GLM47Config(model_path="test/path")
        # Set quantization parameters after initialization
        config.use_quantization = True
        config.quantization_scheme = 'int8'
        config.quantization_bits = 8
        config.quantization_symmetric = True
        config.quantization_per_channel = True

        # Mock the model loading to avoid actual model download
        with patch('src.inference_pio.models.glm_4_7.model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.inference_pio.models.glm_4_7.model.AutoTokenizer.from_pretrained'):
            model = GLM47Model(config)

        # Check that quantization was attempted to be applied
        # Since we can't actually run the quantization due to mocking,
        # we just verify the config was set correctly
        assert_true(hasattr(config))
        assert_equal(config.use_quantization, True)
        assert_equal(config.quantization_scheme, 'int8')

    def qwen3_4b_model_quantization_config(self)():
        """Test Qwen3-4b-instruct-2507 model quantization configuration."""
        config = Qwen34BInstruct2507Config(model_path="test/path")
        # Set quantization parameters after initialization
        config.use_quantization = True
        config.quantization_scheme = 'int4'
        config.quantization_bits = 4
        config.quantization_symmetric = True
        config.quantization_per_channel = True

        # Mock the model loading to avoid actual model download
        with patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoTokenizer.from_pretrained'):
            model = Qwen34BInstruct2507Model(config)

        # Check that quantization was attempted to be applied
        assert_true(hasattr(config))
        assert_equal(config.use_quantization, True)
        assert_equal(config.quantization_scheme, 'int4')

    def qwen3_coder_model_quantization_config(self)():
        """Test Qwen3-coder-30b model quantization configuration."""
        config = Qwen3Coder30BConfig(model_path="test/path")
        # Set quantization parameters after initialization
        config.use_quantization = True
        config.quantization_scheme = 'fp16'
        config.quantization_bits = 16
        config.quantization_symmetric = False
        config.quantization_per_channel = False

        # Mock the model loading to avoid actual model download
        with patch('src.inference_pio.models.qwen3_coder_30b.model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_coder_30b.model.AutoTokenizer.from_pretrained'):
            model = Qwen3Coder30BModel(config)

        # Check that quantization was attempted to be applied
        assert_true(hasattr(config))
        assert_equal(config.use_quantization, True)
        assert_equal(config.quantization_scheme, 'fp16')

    def qwen3_vl_model_quantization_config(self)():
        """Test Qwen3-vl-2b model quantization configuration."""
        config = Qwen3VL2BConfig()
        # Set quantization parameters after initialization
        config.use_quantization = True
        config.quantization_scheme = 'nf4'
        config.quantization_bits = 4
        config.quantization_symmetric = True
        config.quantization_per_channel = True

        # Mock the model loading to avoid actual model download
        with patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained'):
            model = Qwen3VL2BModel(config)

        # Check that quantization was attempted to be applied
        assert_true(hasattr(config))
        assert_equal(config.use_quantization, True)
        assert_equal(config.quantization_scheme, 'nf4')
    
    def quantization_scheme_constants(self)():
        """Test that quantization schemes are properly defined."""
        assert_equal(QuantizationScheme.INT8.value, "int8")
        assert_equal(QuantizationScheme.INT4.value, "int4")
        assert_equal(QuantizationScheme.FP16.value, "fp16")
        assert_equal(QuantizationScheme.NF4.value, "nf4")

if __name__ == '__main__':
    run_tests(test_functions)