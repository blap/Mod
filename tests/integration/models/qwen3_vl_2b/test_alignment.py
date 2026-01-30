#!/usr/bin/env python
"""
Test script for cross-modal alignment optimization implementation.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import torch
from src.inference_pio.common.cross_modal_alignment_optimization import Qwen3VL2BCrossModalAlignmentOptimizer, CrossModalAlignmentConfig
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig

print('Creating test configuration...')
# Create a test configuration
qwen_config = Qwen3VL2BConfig()
alignment_config = CrossModalAlignmentConfig(
    alignment_temperature=0.5,
    alignment_lambda=0.1,
    use_contrastive_alignment=True,
    contrastive_margin=0.2,
    enable_dynamic_alignment=True,
    alignment_frequency=10,
    alignment_threshold=0.8,
    use_attention_alignment=True,
    use_learned_alignment=True,
    alignment_projection_dim=qwen_config.hidden_size,
    enable_similarity_alignment=True,
    similarity_method='cosine'
)

print('Creating optimizer...')
# Create the optimizer
optimizer = Qwen3VL2BCrossModalAlignmentOptimizer(qwen_config, alignment_config)

print('Creating sample features...')
# Create sample vision and language features
batch_size = 2
vision_seq_len = 10
lang_seq_len = 15
hidden_size = qwen_config.hidden_size

vision_features = torch.randn(batch_size, vision_seq_len, hidden_size)
language_features = torch.randn(batch_size, lang_seq_len, hidden_size)

print('Testing alignment...')
# Test alignment
try:
    aligned_features, alignment_loss = optimizer.align_modalities(vision_features, language_features)
    print('Cross-modal alignment test passed!')
    print(f'Input vision shape: {vision_features.shape}')
    print(f'Input language shape: {language_features.shape}')
    print(f'Aligned vision shape: {aligned_features[0].shape}')
    print(f'Aligned language shape: {aligned_features[1].shape}')
    print(f'Alignment loss: {alignment_loss.item():.6f}')
except Exception as e:
    print(f'Error during alignment: {e}')
    import traceback
    traceback.print_exc()