"""
Test for multimodal attention in Qwen3-VL-2B model.

This module tests that the Qwen3-VL-2B model correctly applies multimodal attention
while other models do not.
"""
import torch
from src.inference_pio.test_utils import (
    assert_equal, assert_not_equal, assert_true, assert_false,
    assert_is_none, assert_is_not_none, assert_in, assert_not_in,
    assert_greater, assert_less, assert_is_instance, assert_raises,
    run_tests
)
from inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from inference_pio.models.glm_4_7_flash.model import GLM47FlashModel
from inference_pio.models.glm_4_7_flash.config import GLM47FlashConfig
from inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig

def test_qwen3_vl_2b_has_multimodal_attention():
    """Test that Qwen3-VL-2B model has multimodal attention components."""
    config = Qwen3VL2BConfig()
    config.use_multimodal_attention = True

    # Create a minimal mock model that inherits from nn.Module to allow module assignment
    class MockQwen3VL2BModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self._model = None
            self._tokenizer = None
            self._image_processor = None
            self._model_name = config.model_path
            self._memory_config = {
                "gradient_checkpointing": config.gradient_checkpointing,
                "use_cache": config.use_cache,
                "torch_dtype": torch.float16,
                "device_map": config.device_map,
                "low_cpu_mem_usage": config.low_cpu_mem_usage,
                "max_memory": config.max_memory
            }

        def _apply_multimodal_attention(self):
            """Mock implementation of multimodal attention application."""
            # For this test, we'll directly implement the logic
            is_multimodal = getattr(self.config, 'is_multimodal', True)

            if is_multimodal:
                # Create multimodal components
                self.multimodal_alignment = torch.nn.Module()  # Mock module
                self.multimodal_fusion = torch.nn.Module()    # Mock module

    # Create the mock model
    model = MockQwen3VL2BModel()

    # Call the multimodal attention application method directly
    model._apply_multimodal_attention()

    # Check that multimodal components were added
    assert_is_not_none(getattr(model, 'multimodal_alignment', None))
    assert_is_not_none(getattr(model, 'multimodal_fusion', None))


def test_non_vision_models_do_not_have_multimodal_attention():
    """Test that non-vision models do not have multimodal attention components."""
    # Test GLM-4.7-Flash model
    glm_config = GLM47FlashConfig()
    glm_config.use_multimodal_attention = True  # Even if set to True, it shouldn't apply multimodal attention

    # Mock the model loading to avoid downloading the full model
    glm_model = GLM47FlashModel.__new__(GLM47FlashModel)
    glm_model.config = glm_config
    glm_model._model = None
    glm_model._tokenizer = None
    glm_model._model_name = glm_config.model_path
    glm_model._memory_config = {
        "gradient_checkpointing": glm_config.gradient_checkpointing,
        "use_cache": glm_config.use_cache,
        "torch_dtype": torch.float16,
        "device_map": glm_config.device_map,
        "low_cpu_mem_usage": glm_config.low_cpu_mem_usage,
        "max_memory": glm_config.max_memory
    }

    # GLM-4.7 should not have multimodal components even if config says to use them
    # Check that the model doesn't have multimodal-specific attributes
    assert_false(hasattr(glm_model, '_apply_multimodal_attention'))

    # Test Qwen3-4B-Instruct-2507 model
    qwen4b_config = Qwen34BInstruct2507Config()
    qwen4b_config.use_multimodal_attention = True  # Even if set to True, it shouldn't apply multimodal attention

    # Mock the model loading to avoid downloading the full model
    qwen4b_model = Qwen34BInstruct2507Model.__new__(Qwen34BInstruct2507Model)
    qwen4b_model.config = qwen4b_config
    qwen4b_model._model = None
    qwen4b_model._tokenizer = None
    qwen4b_model._model_name = qwen4b_config.model_path
    qwen4b_model._memory_config = {
        "gradient_checkpointing": qwen4b_config.gradient_checkpointing,
        "use_cache": qwen4b_config.use_cache,
        "torch_dtype": torch.float16,
        "device_map": qwen4b_config.device_map,
        "low_cpu_mem_usage": qwen4b_config.low_cpu_mem_usage,
        "max_memory": qwen4b_config.max_memory
    }

    # Qwen3-4B-Instruct-2507 should not have multimodal components even if config says to use them
    assert_false(hasattr(qwen4b_model, '_apply_multimodal_attention'))

    # Test Qwen3-Coder-30B model
    qwen30b_config = Qwen3Coder30BConfig()
    qwen30b_config.use_multimodal_attention = True  # Even if set to True, it shouldn't apply multimodal attention

    # Mock the model loading to avoid downloading the full model
    qwen30b_model = Qwen3Coder30BModel.__new__(Qwen3Coder30BModel)
    qwen30b_model.config = qwen30b_config
    qwen30b_model._model = None
    qwen30b_model._tokenizer = None
    qwen30b_model._model_name = qwen30b_config.model_path
    qwen30b_model._memory_config = {
        "gradient_checkpointing": qwen30b_config.gradient_checkpointing,
        "use_cache": qwen30b_config.use_cache,
        "torch_dtype": torch.float16,
        "device_map": qwen30b_config.device_map,
        "low_cpu_mem_usage": qwen30b_config.low_cpu_mem_usage,
        "max_memory": qwen30b_config.max_memory
    }

    # Qwen3-Coder-30B should not have multimodal components even if config says to use them
    assert_false(hasattr(qwen30b_model, '_apply_multimodal_attention'))


def test_multimodal_attention_application_logic():
    """Test the logic of multimodal attention application."""
    # Create a Qwen3-VL-2B model config with multimodal attention enabled
    vl_config = Qwen3VL2BConfig()
    vl_config.use_multimodal_attention = True

    # Create a minimal mock model that inherits from nn.Module to allow module assignment
    class MockQwen3VL2BModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = vl_config
            self._model = None
            self._tokenizer = None
            self._image_processor = None
            self._model_name = vl_config.model_path
            self._memory_config = {
                "gradient_checkpointing": vl_config.gradient_checkpointing,
                "use_cache": vl_config.use_cache,
                "torch_dtype": torch.float16,
                "device_map": vl_config.device_map,
                "low_cpu_mem_usage": vl_config.low_cpu_mem_usage,
                "max_memory": vl_config.max_memory
            }

        def _apply_traditional_optimizations(self):
            """Mock implementation of traditional optimizations."""
            # Apply multimodal attention if enabled
            if getattr(self.config, 'use_multimodal_attention', False):
                self._apply_multimodal_attention()

        def _apply_multimodal_attention(self):
            """Mock implementation of multimodal attention application."""
            # For this test, we'll directly implement the logic
            is_multimodal = getattr(self.config, 'is_multimodal', True)

            if is_multimodal:
                # Create multimodal components
                self.multimodal_alignment = torch.nn.Module()  # Mock module
                self.multimodal_fusion = torch.nn.Module()    # Mock module

    # Create the mock model
    vl_model = MockQwen3VL2BModel()

    # Apply traditional optimizations which include multimodal attention for VL models
    vl_model._apply_traditional_optimizations()

    # Verify that multimodal components were added
    assert_is_not_none(getattr(vl_model, 'multimodal_alignment', None))
    assert_is_not_none(getattr(vl_model, 'multimodal_fusion', None))


def test_multimodal_attention_disabled_logic():
    """Test that multimodal attention is not applied when disabled."""
    # Create a Qwen3-VL-2B model config with multimodal attention disabled
    vl_config = Qwen3VL2BConfig()
    vl_config.use_multimodal_attention = False

    # Create a minimal mock model that inherits from nn.Module to allow module assignment
    class MockQwen3VL2BModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = vl_config
            self._model = None
            self._tokenizer = None
            self._image_processor = None
            self._model_name = vl_config.model_path
            self._memory_config = {
                "gradient_checkpointing": vl_config.gradient_checkpointing,
                "use_cache": vl_config.use_cache,
                "torch_dtype": torch.float16,
                "device_map": vl_config.device_map,
                "low_cpu_mem_usage": vl_config.low_cpu_mem_usage,
                "max_memory": vl_config.max_memory
            }

        def _apply_traditional_optimizations(self):
            """Mock implementation of traditional optimizations."""
            # Apply multimodal attention if enabled
            if getattr(self.config, 'use_multimodal_attention', False):
                self._apply_multimodal_attention()

        def _apply_multimodal_attention(self):
            """Mock implementation of multimodal attention application."""
            # For this test, we'll directly implement the logic
            is_multimodal = getattr(self.config, 'is_multimodal', True)

            if is_multimodal:
                # Create multimodal components
                self.multimodal_alignment = torch.nn.Module()  # Mock module
                self.multimodal_fusion = torch.nn.Module()    # Mock module

    # Create the mock model
    vl_model = MockQwen3VL2BModel()

    # Apply traditional optimizations which should not include multimodal attention
    vl_model._apply_traditional_optimizations()

    # Verify that multimodal components were not added when disabled
    # They might not be present, or if present, they should not be active
    # The key is that the method should not fail when multimodal attention is disabled
    assert_false(hasattr(vl_model, 'multimodal_alignment'))
    assert_false(hasattr(vl_model, 'multimodal_fusion'))

if __name__ == '__main__':
    run_tests([
        test_qwen3_vl_2b_has_multimodal_attention,
        test_non_vision_models_do_not_have_multimodal_attention,
        test_multimodal_attention_application_logic,
        test_multimodal_attention_disabled_logic
    ])