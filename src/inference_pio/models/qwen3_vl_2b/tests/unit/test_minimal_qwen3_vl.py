#!/usr/bin/env python3
"""
Minimal test for Qwen3-VL-2B model to verify implementation without downloading the full model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from unittest.mock import patch, MagicMock

def test_qwen3_vl_model_creation():
    """Test Qwen3-VL-2B model creation with mocked components."""

    # Create a minimal config
    config = Qwen3VL2BConfig()
    config.model_path = "dummy_path"  # Use a dummy path to avoid download

    # Mock the AutoModelForVision2Seq.from_pretrained to avoid downloading
    with patch('transformers.AutoModelForVision2Seq.from_pretrained') as mock_model_fn, \
         patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_fn, \
         patch('transformers.AutoImageProcessor.from_pretrained') as mock_image_processor_fn:

        # Create mock model, tokenizer, and image processor
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_image_processor = MagicMock()

        # Configure the mocks to return the mock objects
        mock_model_fn.return_value = mock_model
        mock_tokenizer_fn.return_value = mock_tokenizer
        mock_image_processor_fn.return_value = mock_image_processor

        # Set necessary attributes on the mock model
        mock_model.gradient_checkpointing_enable = MagicMock()
        mock_model.device = 'cpu'

        # Import and create the model
        from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel

        try:
            model = Qwen3VL2BModel(config)
            print("✓ Qwen3-VL-2B model created successfully with mocked components")

            # Verify that the model has the expected attributes
            assert hasattr(model, '_model'), "Model should have _model attribute"
            assert hasattr(model, '_tokenizer'), "Model should have _tokenizer attribute"
            assert hasattr(model, '_image_processor'), "Model should have _image_processor attribute"

            print("✓ Model has expected attributes")

            return True

        except Exception as e:
            print(f"✗ Error creating Qwen3-VL-2B model: {e}")
            import traceback
            traceback.print_exc()
            return False

def test_config_attributes():
    """Test that the config has all required attributes."""
    config = Qwen3VL2BConfig()
    
    required_attrs = [
        'model_path',
        'hidden_size',
        'num_attention_heads',
        'num_hidden_layers',
        'intermediate_size',
        'vocab_size',
        'layer_norm_eps',
        'max_position_embeddings',
        'rope_theta',
        'use_flash_attention_2',
        'use_sparse_attention',
        'use_sliding_window_attention',
        'use_multi_query_attention',
        'use_grouped_query_attention',
        'use_paged_attention',
        'use_fused_layer_norm',
        'use_bias_removal_optimization',
        'use_kv_cache_compression',
        'use_prefix_caching',
        'use_cuda_kernels',
        'enable_disk_offloading',
        'enable_intelligent_pagination',
        'enable_continuous_nas',
        'enable_sequence_parallelism',
        'enable_vision_language_parallelism',
        'use_quantization',
        'use_multimodal_attention',
        'use_snn_conversion'
    ]
    
    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(config, attr):
            missing_attrs.append(attr)
    
    if missing_attrs:
        print(f"✗ Missing config attributes: {missing_attrs}")
        return False
    else:
        print("✓ All required config attributes are present")
        return True

if __name__ == "__main__":
    print("Testing Qwen3-VL-2B model implementation...")
    
    config_ok = test_config_attributes()
    model_ok = test_qwen3_vl_model_creation()
    
    if config_ok and model_ok:
        print("\n✓ All tests passed! Qwen3-VL-2B implementation is ready.")
    else:
        print("\n✗ Some tests failed.")
        sys.exit(1)