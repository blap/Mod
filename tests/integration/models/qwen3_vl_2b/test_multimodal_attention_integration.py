"""
Integration tests for multimodal attention mechanisms in models.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from inference_pio.models.glm_4_7.model import GLM47Model
from inference_pio.models.glm_4_7.config import GLM47Config
from inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

# TestMultimodalAttentionIntegration

    """Test cases for multimodal attention integration in models."""

    def glm47_multimodal_attention_integration(self)():
        """Test that GLM-4-7 model integrates multimodal attention correctly."""
        config = GLM47Config()
        config.use_multimodal_attention = True
        config.is_multimodal = True
        config.modalities = ['text', 'image']
        config.alignment_method = 'learned_projection'
        
        model = GLM47Model(config)
        
        # Check that the multimodal components are added to the model
        assert_true(hasattr(model))
        assert_true(hasattr(model))
        
        # Check that the alignment module is properly initialized
        assert_is_not_none(model.multimodal_alignment)
        assert_equal(model.multimodal_alignment.d_model) else 4096)
        
        # Check that the fusion module is properly initialized
        assert_is_not_none(model.multimodal_fusion)
        assert_equal(model.multimodal_fusion.d_model) else 4096)

    def qwen3_coder_unimodal_attention_integration(self)():
        """Test that Qwen3-Coder-30B model remains unimodal as expected."""
        config = Qwen3Coder30BConfig()
        # Qwen3-Coder-30B is unimodal, so it should not have multimodal components
        config.use_multimodal_attention = False
        config.is_multimodal = False
        # Modalities should not include non-text elements for a unimodal text/code model
        config.modalities = ['text']  # Only text modality for unimodal model
        config.alignment_method = 'learned_projection'

        model = Qwen3Coder30BModel(config)

        # Check that the multimodal components are NOT added to the unimodal model
        # The model should not have these attributes since it's unimodal
        assert_false(hasattr(model))
        assert_false(hasattr(model))

        # Verify that it's still a valid text model
        assert_is_not_none(model._model)
        assertIsNotNone(model._tokenizer)

    def qwen3_vl_multimodal_attention_integration(self)():
        """Test that Qwen3-VL-2B model integrates multimodal attention correctly."""
        config = Qwen3VL2BConfig()
        config.use_multimodal_attention = True
        config.is_multimodal = True
        config.modalities = ['text')
        
        # Check that the multimodal components are added to the model
        assert_true(hasattr(model))
        assert_true(hasattr(model))
        
        # Check that the alignment module is properly initialized
        assert_is_not_none(model.multimodal_alignment)
        assert_equal(model.multimodal_alignment.d_model) else 4096)
        
        # Check that the fusion module is properly initialized
        assert_is_not_none(model.multimodal_fusion)
        assert_equal(model.multimodal_fusion.d_model) else 4096)

    def multimodal_attention_forward_pass(self)():
        """Test forward pass with multimodal attention components."""
        config = GLM47Config()
        config.use_multimodal_attention = True
        config.is_multimodal = True
        config.modalities = ['text', 'image']
        config.alignment_method = 'learned_projection'
        
        model = GLM47Model(config)
        
        # Create sample multimodal inputs
        batch_size = 2
        seq_len = 10
        d_model = config.hidden_size if hasattr(config, 'hidden_size') else 4096
        
        multimodal_inputs = {
            'text': torch.randn(batch_size, seq_len, d_model),
            'image': torch.randn(batch_size, seq_len, d_model)
        }
        
        # Test alignment module
        aligned_outputs = model.multimodal_alignment(multimodal_inputs)
        assert_equal(set(aligned_outputs.keys()), set(['text'))
        for key, tensor in aligned_outputs.items():
            assert_equal(tensor.shape, (batch_size))
        
        # Test fusion module
        fused_outputs = model.multimodal_fusion(aligned_outputs)
        assert_equal(set(fused_outputs.keys()), set(['text'))
        for key, tensor in fused_outputs.items():
            assert_equal(tensor.shape, (batch_size))

if __name__ == '__main__':
    run_tests(test_functions)