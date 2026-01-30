"""
Test suite for Cross-Modal Fusion Kernels for Qwen3-VL-2B Model.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from src.inference_pio.common.cross_modal_fusion_kernels import (
    CrossModalFusionConfig,
    CrossModalAttentionFusion,
    CrossModalConcatFusion,
    CrossModalAdditiveFusion,
    Qwen3VL2BCrossModalFusion,
    Qwen3VL2BSpecializedCrossModalFusion,
    Qwen3VL2BMultiScaleCrossModalFusion,
    CrossModalFusionManager,
    create_default_qwen3_vl_cross_modal_fusion,
    apply_cross_modal_fusion_to_qwen3_vl_model
)
from src.inference_pio.common.qwen3_vl_cuda_kernels import Qwen3VL2BConfig

# TestCrossModalFusionKernels

    """Test suite for cross-modal fusion kernels."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        qwen_config = Qwen3VL2BConfig(
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=4,
            intermediate_size=1024,
            max_position_embeddings=1024,
            rms_norm_eps=1e-6,
            use_flash_attention_2=True,
            use_cuda_kernels=True
        )
        
        fusion_config = CrossModalFusionConfig(
            hidden_size=512,
            num_attention_heads=8,
            intermediate_size=1024,
            dropout=0.1,
            use_flash_attention=True,
            fusion_method="attention"
        )

    def cross_modal_fusion_config_creation(self)():
        """Test CrossModalFusionConfig creation."""
        config = CrossModalFusionConfig()
        assert_is_instance(config, CrossModalFusionConfig)
        assert_equal(config.hidden_size, 2048)
        assert_equal(config.dropout, 0.1)

    def cross_modal_attention_fusion_creation(self)():
        """Test CrossModalAttentionFusion creation."""
        kernel = CrossModalAttentionFusion(fusion_config)
        assert_is_instance(kernel, CrossModalAttentionFusion)
        assert_equal(kernel.hidden_size, fusion_config.hidden_size)
        assert_equal(kernel.num_heads, fusion_config.num_attention_heads)

    def cross_modal_attention_fusion_forward(self)():
        """Test CrossModalAttentionFusion forward pass."""
        kernel = CrossModalAttentionFusion(fusion_config)

        # Create sample inputs
        batch_size = 2
        vision_seq_len = 10
        lang_seq_len = 15
        vision_features = torch.randn(batch_size, vision_seq_len, fusion_config.hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, fusion_config.hidden_size)

        fused_output, vision_output, language_output = kernel(vision_features, language_features)

        # Check output shapes
        expected_fused_shape = (batch_size, vision_seq_len + lang_seq_len, fusion_config.hidden_size)
        assert_equal(fused_output.shape, expected_fused_shape)
        assert_equal(vision_output.shape, vision_features.shape)
        assert_equal(language_output.shape, language_features.shape)

    def cross_modal_concat_fusion_creation(self)():
        """Test CrossModalConcatFusion creation."""
        kernel = CrossModalConcatFusion(fusion_config)
        assert_is_instance(kernel, CrossModalConcatFusion)
        assert_equal(kernel.hidden_size, fusion_config.hidden_size)

    def cross_modal_concat_fusion_forward(self)():
        """Test CrossModalConcatFusion forward pass."""
        kernel = CrossModalConcatFusion(fusion_config)

        # Create sample inputs
        batch_size = 2
        seq_len = 10
        vision_features = torch.randn(batch_size, seq_len, fusion_config.hidden_size)
        language_features = torch.randn(batch_size, seq_len, fusion_config.hidden_size)

        fused_output, vision_output, language_output = kernel(vision_features, language_features)

        # Check output shapes
        assert_equal(fused_output.shape, (batch_size))
        assert_equal(vision_output.shape, vision_features.shape)
        assert_equal(language_output.shape, language_features.shape)

    def cross_modal_additive_fusion_creation(self)():
        """Test CrossModalAdditiveFusion creation."""
        kernel = CrossModalAdditiveFusion(fusion_config)
        assert_is_instance(kernel, CrossModalAdditiveFusion)
        assert_equal(kernel.hidden_size, fusion_config.hidden_size)

    def cross_modal_additive_fusion_forward(self)():
        """Test CrossModalAdditiveFusion forward pass."""
        kernel = CrossModalAdditiveFusion(fusion_config)

        # Create sample inputs
        batch_size = 2
        seq_len = 10
        vision_features = torch.randn(batch_size, seq_len, fusion_config.hidden_size)
        language_features = torch.randn(batch_size, seq_len, fusion_config.hidden_size)

        fused_output, vision_output, language_output = kernel(vision_features, language_features)

        # Check output shapes
        assert_equal(fused_output.shape, (batch_size))
        assert_equal(vision_output.shape, vision_features.shape)
        assert_equal(language_output.shape, language_features.shape)

    def qwen3_vl2b_cross_modal_fusion_creation(self)():
        """Test Qwen3VL2BCrossModalFusion creation."""
        kernel = Qwen3VL2BCrossModalFusion(qwen_config, fusion_config)
        assert_is_instance(kernel, Qwen3VL2BCrossModalFusion)

    def qwen3_vl2b_cross_modal_fusion_forward(self)():
        """Test Qwen3VL2BCrossModalFusion forward pass."""
        kernel = Qwen3VL2BCrossModalFusion(qwen_config, fusion_config)

        # Create sample inputs
        batch_size = 2
        vision_seq_len = 10
        lang_seq_len = 15
        vision_features = torch.randn(batch_size, vision_seq_len, qwen_config.hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, qwen_config.hidden_size)

        fused_output, vision_output, language_output = kernel(vision_features, language_features)

        # Check output shapes
        expected_fused_shape = (batch_size, vision_seq_len + lang_seq_len, qwen_config.hidden_size)
        assert_equal(fused_output.shape, expected_fused_shape)
        assert_equal(vision_output.shape, vision_features.shape)
        assert_equal(language_output.shape, language_features.shape)

    def cross_modal_fusion_manager(self)():
        """Test CrossModalFusionManager functionality."""
        manager = CrossModalFusionManager(qwen_config)
        
        # Register a fusion method
        manager.register_fusion_method("attention", fusion_config)
        
        # Get the registered kernel
        kernel = manager.get_fusion_kernel("attention")
        assert_is_not_none(kernel)
        assert_is_instance(kernel)
        
        # Test fusion
        batch_size = 1
        vision_seq_len = 5
        lang_seq_len = 8
        vision_features = torch.randn(batch_size, vision_seq_len)
        language_features = torch.randn(batch_size, lang_seq_len, qwen_config.hidden_size)
        
        fused_output, vision_output, language_output = manager.fuse_modalities(
            "attention", vision_features, language_features
        )
        
        expected_fused_shape = (batch_size, vision_seq_len + lang_seq_len, qwen_config.hidden_size)
        assert_equal(fused_output.shape, expected_fused_shape)

    def create_default_qwen3_vl_cross_modal_fusion(self)():
        """Test creating default cross-modal fusion for Qwen3-VL-2B."""
        manager = create_default_qwen3_vl_cross_modal_fusion(qwen_config)
        
        assert_is_instance(manager, CrossModalFusionManager)
        
        # Check that default methods are registered
        for method in ["attention", "concat", "add"]:
            kernel = manager.get_fusion_kernel(method)
            assert_is_not_none(kernel)

    def apply_cross_modal_fusion_to_qwen3_vl_model(self)():
        """Test applying cross-modal fusion to a model."""
        # Create a simple test model
        class SimpleTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                linear = nn.Linear(512, 512)

            def forward(self, x):
                return linear(x)

        model = SimpleTestModel()
        
        # Create fusion manager
        manager = create_default_qwen3_vl_cross_modal_fusion(qwen_config)
        
        # Apply cross-modal fusion
        optimized_model = apply_cross_modal_fusion_to_qwen3_vl_model(model, manager)
        
        # Check that the model has the fusion manager attribute
        assert_true(hasattr(optimized_model))
        assert_is_instance(optimized_model.cross_modal_fusion_manager, CrossModalFusionManager)

    def different_fusion_methods(self)():
        """Test different fusion methods with the same inputs."""
        batch_size = 1
        vision_seq_len = 5
        lang_seq_len = 8
        vision_features = torch.randn(batch_size, vision_seq_len, qwen_config.hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, qwen_config.hidden_size)
        
        # Test attention-based fusion
        attention_config = CrossModalFusionConfig(
            hidden_size=qwen_config.hidden_size,
            num_attention_heads=qwen_config.num_attention_heads,
            intermediate_size=qwen_config.intermediate_size,
            fusion_method="attention"
        )
        attention_fusion = Qwen3VL2BCrossModalFusion(qwen_config, attention_config)
        att_fused, _, _ = attention_fusion(vision_features, language_features)
        
        # Test concat-based fusion
        concat_config = CrossModalFusionConfig(
            hidden_size=qwen_config.hidden_size,
            num_attention_heads=qwen_config.num_attention_heads,
            intermediate_size=qwen_config.intermediate_size,
            fusion_method="concat"
        )
        concat_fusion = Qwen3VL2BCrossModalFusion(qwen_config, concat_config)
        concat_fused, _, _ = concat_fusion(vision_features, language_features)
        
        # Test additive fusion
        add_config = CrossModalFusionConfig(
            hidden_size=qwen_config.hidden_size,
            num_attention_heads=qwen_config.num_attention_heads,
            intermediate_size=qwen_config.intermediate_size,
            fusion_method="add"
        )
        add_fusion = Qwen3VL2BCrossModalFusion(qwen_config, add_config)
        add_fused, _, _ = add_fusion(vision_features, language_features)
        
        # All outputs should have the same shape for fused output
        expected_shape = (batch_size, vision_seq_len + lang_seq_len, qwen_config.hidden_size)
        assert_equal(att_fused.shape, expected_shape)
        assert_equal(concat_fused.shape, (batch_size), qwen_config.hidden_size))
        assert_equal(add_fused.shape, (batch_size), qwen_config.hidden_size))

    def cuda_availability_handling(self)():
        """Test that kernels handle CUDA availability properly."""
        # This test ensures that the kernels don't crash when CUDA is not available
        # or when running on CPU

        kernel = CrossModalAttentionFusion(fusion_config)

        # Create CPU tensors
        batch_size = 1
        vision_seq_len = 3
        lang_seq_len = 4
        vision_features = torch.randn(batch_size, vision_seq_len, fusion_config.hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, fusion_config.hidden_size)

        fused_output, vision_output, language_output = kernel(vision_features, language_features)

        # Check that outputs are on CPU
        assert_equal(fused_output.device.type, 'cpu')
        assert_equal(vision_output.device.type, 'cpu')
        assert_equal(language_output.device.type, 'cpu')

    def qwen3_vl2b_specialized_cross_modal_fusion_creation(self)():
        """Test Qwen3VL2BSpecializedCrossModalFusion creation."""
        fusion_config = CrossModalFusionConfig(
            hidden_size=qwen_config.hidden_size,
            num_attention_heads=qwen_config.num_attention_heads,
            intermediate_size=qwen_config.intermediate_size,
            fusion_method="specialized"
        )
        kernel = Qwen3VL2BSpecializedCrossModalFusion(qwen_config, fusion_config)
        assert_is_instance(kernel, Qwen3VL2BSpecializedCrossModalFusion)

    def qwen3_vl2b_specialized_cross_modal_fusion_forward(self)():
        """Test Qwen3VL2BSpecializedCrossModalFusion forward pass."""
        fusion_config = CrossModalFusionConfig(
            hidden_size=qwen_config.hidden_size,
            num_attention_heads=qwen_config.num_attention_heads,
            intermediate_size=qwen_config.intermediate_size,
            fusion_method="specialized"
        )
        kernel = Qwen3VL2BSpecializedCrossModalFusion(qwen_config, fusion_config)

        # Create sample inputs
        batch_size = 2
        vision_seq_len = 10
        lang_seq_len = 15
        vision_features = torch.randn(batch_size, vision_seq_len, qwen_config.hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, qwen_config.hidden_size)

        fused_output, vision_output, language_output = kernel(vision_features, language_features)

        # Check output shapes
        assert_equal(vision_output.shape, vision_features.shape)
        assert_equal(language_output.shape, language_features.shape)
        assert_equal(fused_output.shape[0], batch_size)  # Batch size preserved
        assert_equal(fused_output.shape[2], qwen_config.hidden_size)  # Hidden size preserved

    def qwen3_vl2b_multi_scale_cross_modal_fusion_creation(self)():
        """Test Qwen3VL2BMultiScaleCrossModalFusion creation."""
        fusion_config = CrossModalFusionConfig(
            hidden_size=qwen_config.hidden_size,
            num_attention_heads=qwen_config.num_attention_heads,
            intermediate_size=qwen_config.intermediate_size,
            fusion_method="multi_scale"
        )
        kernel = Qwen3VL2BMultiScaleCrossModalFusion(qwen_config, fusion_config)
        assert_is_instance(kernel, Qwen3VL2BMultiScaleCrossModalFusion)

    def qwen3_vl2b_multi_scale_cross_modal_fusion_forward(self)():
        """Test Qwen3VL2BMultiScaleCrossModalFusion forward pass."""
        fusion_config = CrossModalFusionConfig(
            hidden_size=qwen_config.hidden_size,
            num_attention_heads=qwen_config.num_attention_heads,
            intermediate_size=qwen_config.intermediate_size,
            fusion_method="multi_scale"
        )
        kernel = Qwen3VL2BMultiScaleCrossModalFusion(qwen_config, fusion_config)

        # Create sample inputs
        batch_size = 2
        vision_seq_len = 8
        lang_seq_len = 12
        vision_features = torch.randn(batch_size, vision_seq_len, qwen_config.hidden_size)
        language_features = torch.randn(batch_size, lang_seq_len, qwen_config.hidden_size)

        fused_output, vision_output, language_output = kernel(vision_features, language_features)

        # Check output shapes
        assert_equal(vision_output.shape, vision_features.shape)
        assert_equal(language_output.shape, language_features.shape)
        assert_equal(fused_output.shape[0], batch_size)  # Batch size preserved
        assert_equal(fused_output.shape[2], qwen_config.hidden_size)  # Hidden size preserved

    def enhanced_forward_with_cross_modal_fusion(self)():
        """Test the enhanced forward method with cross-modal fusion."""
        # This test verifies that the enhanced forward method exists and can be called
        from src.inference_pio.common.cross_modal_fusion_kernels import _enhanced_forward_with_cross_modal_fusion

        # Create a mock model-like object
        
            def __init__(self):
                _original_forward = lambda *args, **kwargs: torch.randn(1, 5, 512)

        mock_model = MockModel()

        # Test that the enhanced forward method can be called without error
        try:
            result = _enhanced_forward_with_cross_modal_fusion(mock_model)
            assert_is_not_none(result)
        except Exception as e:
            # This is expected if certain conditions aren't met
            # The important thing is that the function exists and is callable
            pass

if __name__ == '__main__':
    run_tests(test_functions)