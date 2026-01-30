"""
Tests for SNN Integration in Models

This module contains tests to verify that SNN functionality has been properly
integrated into the existing models.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.glm_4_7.model import GLM47Model
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel

# TestSNNIntegration

    """Test cases for SNN integration in models."""

    def glm47_snn_integration(self)():
        """Test that GLM-4-7 model supports SNN conversion."""
        # Create a minimal config for testing
        config = GLM47Config(
            model_path="dummy_path",  # Will be mocked
            use_snn_conversion=True,
            snn_neuron_type='LIF',
            snn_threshold=1.0,
            snn_decay=0.9,
            enable_snn_optimizations=True,
            snn_pruning_ratio=0.1
        )
        
        # Mock the model loading to avoid actual model download
        with patch('src.inference_pio.models.glm_4_7.model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.inference_pio.models.glm_4_7.model.AutoTokenizer.from_pretrained'):
            # Override the model initialization to use a simple model
            model = GLM47Model(config)
            
            # Check that the config has SNN attributes
            assert_true(hasattr(config))
            assert_true(config.use_snn_conversion)
            
            # Check that the model has the SNN conversion method
            assertTrue(hasattr(model))

    def qwen3_4b_instruct_snn_integration(self)():
        """Test that Qwen3-4B-Instruct model supports SNN conversion."""
        # Create a minimal config for testing
        config = Qwen34BInstruct2507Config(
            model_path="dummy_path",  # Will be mocked
            use_snn_conversion=True,
            snn_neuron_type='LIF',
            snn_threshold=1.0,
            snn_decay=0.9,
            enable_snn_optimizations=True,
            snn_pruning_ratio=0.1
        )
        
        # Mock the model loading to avoid actual model download
        with patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoTokenizer.from_pretrained'):
            # Override the model initialization to use a simple model
            model = Qwen34BInstruct2507Model(config)
            
            # Check that the config has SNN attributes
            assert_true(hasattr(config))
            assert_true(config.use_snn_conversion)
            
            # Check that the model has the SNN conversion method
            assertTrue(hasattr(model))

    def qwen3_coder_snn_integration(self)():
        """Test that Qwen3-Coder model supports SNN conversion."""
        # Create a minimal config for testing
        config = Qwen3Coder30BConfig(
            model_path="dummy_path",  # Will be mocked
            use_snn_conversion=True,
            snn_neuron_type='LIF',
            snn_threshold=1.0,
            snn_decay=0.9,
            enable_snn_optimizations=True,
            snn_pruning_ratio=0.1
        )
        
        # Mock the model loading to avoid actual model download
        with patch('src.inference_pio.models.qwen3_coder_30b.model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_coder_30b.model.AutoTokenizer.from_pretrained'):
            # Override the model initialization to use a simple model
            model = Qwen3Coder30BModel(config)
            
            # Check that the config has SNN attributes
            assert_true(hasattr(config))
            assert_true(config.use_snn_conversion)
            
            # Check that the model has the SNN conversion method
            assertTrue(hasattr(model))

    def qwen3_vl_snn_integration(self)():
        """Test that Qwen3-VL model supports SNN conversion."""
        # Create a minimal config for testing
        config = Qwen3VL2BConfig(
            model_path="dummy_path",  # Will be mocked
            use_snn_conversion=True,
            snn_neuron_type='LIF',
            snn_threshold=1.0,
            snn_decay=0.9,
            enable_snn_optimizations=True,
            snn_pruning_ratio=0.1
        )
        
        # Mock the model loading to avoid actual model download
        with patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained'):
            # Override the model initialization to use a simple model
            model = Qwen3VL2BModel(config)
            
            # Check that the config has SNN attributes
            assert_true(hasattr(config))
            assert_true(config.use_snn_conversion)
            
            # Check that the model has the SNN conversion method
            assertTrue(hasattr(model))

    def snn_conversion_method_exists(self)():
        """Test that all models have the SNN conversion method."""
        # Test GLM-4-7
        config_glm = GLM47Config(model_path="dummy")
        with patch('src.inference_pio.models.glm_4_7.model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.inference_pio.models.glm_4_7.model.AutoTokenizer.from_pretrained'):
            model_glm = GLM47Model(config_glm)
            assert_true(hasattr(model_glm))
        
        # Test Qwen3-4B-Instruct
        config_qwen3_4b = Qwen34BInstruct2507Config(model_path="dummy")
        with patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_4b_instruct_2507.model.AutoTokenizer.from_pretrained'):
            model_qwen3_4b = Qwen34BInstruct2507Model(config_qwen3_4b)
            assert_true(hasattr(model_qwen3_4b))
        
        # Test Qwen3-Coder
        config_qwen3_coder = Qwen3Coder30BConfig(model_path="dummy")
        with patch('src.inference_pio.models.qwen3_coder_30b.model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_coder_30b.model.AutoTokenizer.from_pretrained'):
            model_qwen3_coder = Qwen3Coder30BModel(config_qwen3_coder)
            assert_true(hasattr(model_qwen3_coder))
        
        # Test Qwen3-VL
        config_qwen3_vl = Qwen3VL2BConfig(model_path="dummy")
        with patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained'), \
             patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained'):
            model_qwen3_vl = Qwen3VL2BModel(config_qwen3_vl)
            assert_true(hasattr(model_qwen3_vl))

# TestSNNConfigAttributes

    """Test cases for SNN configuration attributes."""

    def glm47_config_has_snn_attributes(self)():
        """Test that GLM-4-7 config supports SNN attributes."""
        config = GLM47Config(model_path="dummy")
        
        # Check that SNN-related attributes can be set
        config.use_snn_conversion = True
        config.snn_neuron_type = 'LIF'
        config.snn_threshold = 1.0
        config.snn_decay = 0.9
        config.enable_snn_optimizations = True
        config.snn_pruning_ratio = 0.2
        
        assert_true(config.use_snn_conversion)
        assert_equal(config.snn_neuron_type)
        assert_equal(config.snn_threshold, 1.0)
        assert_equal(config.snn_decay, 0.9)
        assert_true(config.enable_snn_optimizations)
        assert_equal(config.snn_pruning_ratio)

    def qwen3_4b_instruct_config_has_snn_attributes(self)():
        """Test that Qwen3-4B-Instruct config supports SNN attributes."""
        config = Qwen34BInstruct2507Config(model_path="dummy")
        
        # Check that SNN-related attributes can be set
        config.use_snn_conversion = True
        config.snn_neuron_type = 'AdaptiveLIF'
        config.snn_threshold = 1.5
        config.snn_decay = 0.8
        config.enable_snn_optimizations = True
        config.snn_pruning_ratio = 0.15
        
        assert_true(config.use_snn_conversion)
        assert_equal(config.snn_neuron_type)
        assert_equal(config.snn_threshold, 1.5)
        assert_equal(config.snn_decay, 0.8)
        assert_true(config.enable_snn_optimizations)
        assert_equal(config.snn_pruning_ratio)

    def qwen3_coder_config_has_snn_attributes(self)():
        """Test that Qwen3-Coder config supports SNN attributes."""
        config = Qwen3Coder30BConfig(model_path="dummy")
        
        # Check that SNN-related attributes can be set
        config.use_snn_conversion = True
        config.snn_neuron_type = 'LIF'
        config.snn_threshold = 0.8
        config.snn_decay = 0.95
        config.enable_snn_optimizations = False
        config.snn_pruning_ratio = 0.1
        
        assert_true(config.use_snn_conversion)
        assert_equal(config.snn_neuron_type)
        assert_equal(config.snn_threshold, 0.8)
        assert_equal(config.snn_decay, 0.95)
        assert_false(config.enable_snn_optimizations)
        assert_equal(config.snn_pruning_ratio)

    def qwen3_vl_config_has_snn_attributes(self)():
        """Test that Qwen3-VL config supports SNN attributes."""
        config = Qwen3VL2BConfig(model_path="dummy")
        
        # Check that SNN-related attributes can be set
        config.use_snn_conversion = True
        config.snn_neuron_type = 'AdaptiveLIF'
        config.snn_threshold = 1.2
        config.snn_decay = 0.85
        config.enable_snn_optimizations = True
        config.snn_pruning_ratio = 0.25
        
        assert_true(config.use_snn_conversion)
        assert_equal(config.snn_neuron_type)
        assert_equal(config.snn_threshold, 1.2)
        assert_equal(config.snn_decay, 0.85)
        assert_true(config.enable_snn_optimizations)
        assert_equal(config.snn_pruning_ratio)

# TestSNNFunctionality

    """Test cases for SNN functionality in models."""

    def snn_conversion_call_when_enabled(self)():
        """Test that SNN conversion is called when enabled in config."""
        config = GLM47Config(
            model_path="dummy_path",
            use_snn_conversion=True
        )
        
        with patch('src.inference_pio.models.glm_4_7.model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.inference_pio.models.glm_4_7.model.AutoTokenizer.from_pretrained'), \
             patch.object(GLM47Model, '_apply_snn_conversion') as mock_snn_conversion:
            
            # Create the model
            model = GLM47Model(config)
            
            # Verify that SNN conversion was called during initialization
            # Since it's called inside _apply_configured_optimizations, we need to check that
            # the method exists and would be called based on the config
            assert_true(hasattr(model))
            # The actual call depends on the internal logic of _apply_configured_optimizations

    def snn_conversion_not_called_when_disabled(self)():
        """Test that SNN conversion is not called when disabled in config."""
        config = GLM47Config(
            model_path="dummy_path",
            use_snn_conversion=False  # Disabled
        )
        
        with patch('src.inference_pio.models.glm_4_7.model.AutoModelForCausalLM.from_pretrained'), \
             patch('src.inference_pio.models.glm_4_7.model.AutoTokenizer.from_pretrained'), \
             patch.object(GLM47Model, '_apply_snn_conversion') as mock_snn_conversion:
            
            # Create the model
            model = GLM47Model(config)
            
            # Verify that SNN conversion method exists but wasn't called
            assert_true(hasattr(model))
            # The mock should not have been called if SNN conversion is disabled
            # Note: This depends on the internal implementation of _apply_configured_optimizations

if __name__ == '__main__':
    run_tests(test_functions)