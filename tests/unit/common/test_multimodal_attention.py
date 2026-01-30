"""
Unit tests for multimodal attention mechanisms.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
import torch.nn as nn
from inference_pio.common.multimodal_attention import (
    EfficientMultimodalCrossAttention,
    MultimodalAlignmentModule,
    MultimodalFusionLayer,
    AdaptiveMultimodalAttention
)

# TestEfficientMultimodalCrossAttention

    """Test cases for EfficientMultimodalCrossAttention."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        d_model = 512
        nhead = 8
        modalities = ["text", "image", "audio"]
        batch_size = 2
        seq_len = 10

    def initialization(self)():
        """Test initialization of EfficientMultimodalCrossAttention."""
        attention = EfficientMultimodalCrossAttention(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities
        )
        
        assert_equal(attention.d_model, d_model)
        assert_equal(attention.nhead, nhead)
        assert_equal(attention.modalities, modalities)
        assert_equal(len(attention.modality_projections), len(modalities))

    def forward_pass(self)():
        """Test forward pass of EfficientMultimodalCrossAttention."""
        attention = EfficientMultimodalCrossAttention(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities
        )

        # Create sample inputs for each modality
        queries = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }
        keys = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }
        values = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }

        outputs, attention_weights = attention(
            queries=queries,
            keys=keys,
            values=values,
            need_weights=True
        )

        # Check output shapes
        for modality in modalities:
            assert_equal(outputs[modality].shape, (batch_size))
            if attention_weights:
                assert_equal(attention_weights[modality].shape, 
                               (batch_size) * seq_len))

    def forward_pass_without_weights(self)():
        """Test forward pass without returning attention weights."""
        attention = EfficientMultimodalCrossAttention(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities
        )

        # Create sample inputs for each modality
        queries = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }
        keys = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }
        values = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }

        outputs, attention_weights = attention(
            queries=queries,
            keys=keys,
            values=values,
            need_weights=False
        )

        # Check output shapes
        for modality in modalities:
            assert_equal(outputs[modality].shape, (batch_size))
        
        # Check that attention weights are None
        assert_is_none(attention_weights)

    def different_sequence_lengths(self)():
        """Test attention with different sequence lengths across modalities."""
        attention = EfficientMultimodalCrossAttention(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities
        )

        # Create sample inputs with different sequence lengths
        seq_lens = [5, 10, 15]
        queries = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality, seq_len in zip(modalities, seq_lens)
        }
        keys = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality, seq_len in zip(modalities, seq_lens)
        }
        values = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality, seq_len in zip(modalities, seq_lens)
        }

        outputs, _ = attention(
            queries=queries,
            keys=keys,
            values=values,
            need_weights=False
        )

        # Check output shapes match input shapes
        for modality, seq_len in zip(modalities, seq_lens):
            assert_equal(outputs[modality].shape, (batch_size))

# TestMultimodalAlignmentModule

    """Test cases for MultimodalAlignmentModule."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        d_model = 512
        modalities = ["text", "image", "audio"]
        batch_size = 2
        seq_len = 10

    def initialization(self)():
        """Test initialization of MultimodalAlignmentModule."""
        alignment = MultimodalAlignmentModule(
            d_model=d_model,
            modalities=modalities
        )
        
        assert_equal(alignment.d_model, d_model)
        assert_equal(alignment.modalities, modalities)
        assert_equal(len(alignment.projection_layers), len(modalities))

    def forward_pass_learned_projection(self)():
        """Test forward pass with learned projection alignment method."""
        alignment = MultimodalAlignmentModule(
            d_model=d_model,
            modalities=modalities,
            alignment_method="learned_projection"
        )

        # Create sample inputs for each modality
        inputs = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }

        aligned_outputs = alignment(inputs)

        # Check output shapes
        for modality in modalities:
            assert_equal(aligned_outputs[modality].shape, (batch_size))

    def forward_pass_cross_attention(self)():
        """Test forward pass with cross-attention alignment method."""
        alignment = MultimodalAlignmentModule(
            d_model=d_model,
            modalities=modalities,
            alignment_method="cross_attention"
        )

        # Create sample inputs for each modality
        inputs = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }

        aligned_outputs = alignment(inputs)

        # Check output shapes
        for modality in modalities:
            assert_equal(aligned_outputs[modality].shape, (batch_size))

    def forward_pass_contrastive(self)():
        """Test forward pass with contrastive alignment method."""
        alignment = MultimodalAlignmentModule(
            d_model=d_model,
            modalities=modalities,
            alignment_method="contrastive"
        )

        # Create sample inputs for each modality
        inputs = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }

        aligned_outputs = alignment(inputs)

        # Check output shapes (should be same as input for contrastive method)
        for modality in modalities:
            assert_equal(aligned_outputs[modality].shape, (batch_size))

# TestMultimodalFusionLayer

    """Test cases for MultimodalFusionLayer."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        d_model = 512
        nhead = 8
        modalities = ["text", "image", "audio"]
        batch_size = 2
        seq_len = 10

    def initialization(self)():
        """Test initialization of MultimodalFusionLayer."""
        fusion_layer = MultimodalFusionLayer(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities
        )
        
        assert_equal(fusion_layer.d_model, d_model)
        assert_equal(fusion_layer.nhead, nhead)
        assert_equal(fusion_layer.modalities, modalities)

    def forward_pass_with_alignment(self)():
        """Test forward pass with alignment enabled."""
        fusion_layer = MultimodalFusionLayer(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities,
            use_alignment=True
        )

        # Create sample inputs for each modality
        inputs = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }

        outputs = fusion_layer(inputs)

        # Check output shapes
        for modality in modalities:
            assert_equal(outputs[modality].shape, (batch_size))

    def forward_pass_without_alignment(self)():
        """Test forward pass with alignment disabled."""
        fusion_layer = MultimodalFusionLayer(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities,
            use_alignment=False
        )

        # Create sample inputs for each modality
        inputs = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }

        outputs = fusion_layer(inputs)

        # Check output shapes
        for modality in modalities:
            assert_equal(outputs[modality].shape, (batch_size))

    def forward_pass_different_modalities(self)():
        """Test forward pass with different sets of modalities."""
        modalities_subset = ["text", "image"]
        fusion_layer = MultimodalFusionLayer(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities_subset,
            use_alignment=True
        )

        # Create sample inputs for subset of modalities
        inputs = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities_subset
        }

        outputs = fusion_layer(inputs)

        # Check output shapes
        for modality in modalities_subset:
            assert_equal(outputs[modality].shape, (batch_size))

# TestAdaptiveMultimodalAttention

    """Test cases for AdaptiveMultimodalAttention."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        d_model = 512
        nhead = 8
        modalities = ["text", "image", "audio"]
        batch_size = 2
        seq_len = 10

    def initialization(self)():
        """Test initialization of AdaptiveMultimodalAttention."""
        attention = AdaptiveMultimodalAttention(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities
        )
        
        assert_equal(attention.d_model, d_model)
        assert_equal(attention.nhead, nhead)
        assert_equal(attention.modalities, modalities)

    def forward_pass(self)():
        """Test forward pass of AdaptiveMultimodalAttention."""
        attention = AdaptiveMultimodalAttention(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities
        )

        # Create sample inputs for each modality
        queries = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }
        keys = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }
        values = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }

        outputs, attention_weights = attention(
            queries=queries,
            keys=keys,
            values=values,
            need_weights=True
        )

        # Check output shapes
        for modality in modalities:
            assert_equal(outputs[modality].shape, (batch_size))
            if attention_weights:
                assert_equal(attention_weights[modality].shape, 
                               (batch_size) * seq_len))

    def forward_pass_without_adaptive_features(self)():
        """Test forward pass with adaptive features disabled."""
        attention = AdaptiveMultimodalAttention(
            d_model=d_model,
            nhead=nhead,
            modalities=modalities,
            adaptive_temperature=False,
            adaptive_sparsity=False
        )

        # Create sample inputs for each modality
        queries = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }
        keys = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }
        values = {
            modality: torch.randn(batch_size, seq_len, d_model)
            for modality in modalities
        }

        outputs, attention_weights = attention(
            queries=queries,
            keys=keys,
            values=values,
            need_weights=True
        )

        # Check output shapes
        for modality in modalities:
            assert_equal(outputs[modality].shape, (batch_size))
            if attention_weights:
                assert_equal(attention_weights[modality].shape, 
                               (batch_size) * seq_len))

if __name__ == '__main__':
    run_tests(test_functions)