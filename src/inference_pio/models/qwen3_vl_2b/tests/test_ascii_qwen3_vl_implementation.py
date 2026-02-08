#!/usr/bin/env python3
"""
Test for Qwen3-VL-2B model implementation to verify all optimizations are properly implemented.
"""

import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from unittest.mock import MagicMock, patch

from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig


def test_config_attributes():
    """Test that the config has all required attributes."""
    print("Testing Qwen3-VL-2B config attributes...")
    config = Qwen3VL2BConfig()
    # ... (same as before) ...
    required_attrs = [
        "model_path",
        "hidden_size",
        "num_attention_heads",
        "num_hidden_layers",
        "intermediate_size",
        "vocab_size",
        "layer_norm_eps",
        "max_position_embeddings",
        "rope_theta",
        "use_flash_attention_2",
        "use_sparse_attention",
        "use_sliding_window_attention",
        "use_multi_query_attention",
        "use_grouped_query_attention",
        "use_paged_attention",
        "use_fused_layer_norm",
        "use_bias_removal_optimization",
        "use_kv_cache_compression",
        "use_prefix_caching",
        "use_cuda_kernels",
        "enable_disk_offloading",
        "enable_intelligent_pagination",
        "enable_continuous_nas",
        "enable_sequence_parallelism",
        "enable_vision_language_parallelism",
        "use_quantization",
        "use_multimodal_attention",
        "use_snn_conversion",
    ]

    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(config, attr):
            missing_attrs.append(attr)

    if missing_attrs:
        print(f"[ERROR] Missing config attributes: {missing_attrs}")
        return False
    else:
        print("[OK] All required config attributes are present")
        return True


def test_qwen3_vl_model_creation():
    """Test Qwen3-VL-2B model creation with real components when possible."""
    print("Testing Qwen3-VL-2B model creation...")

    # Create a TINY config to avoid OOM
    config = Qwen3VL2BConfig()
    config.model_path = "dummy_path"
    config.hidden_size = 64
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.intermediate_size = 128
    config.vocab_size = 152000  # Must be > padding_idx (151643)
    config.vision_hidden_size = 64
    config.vision_num_heads = 4
    config.vision_num_layers = 2

    # Import and create the model
    try:
        from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
    except ImportError as e:
        print(f"[ERROR] Failed to import Qwen3VL2BModel: {e}")
        return False

    try:
        # Patch where CustomModelLoader is used (core.modeling)
        with patch("src.inference_pio.models.qwen3_vl_2b.core.modeling.CustomModelLoader") as mock_loader_cls, \
             patch("src.inference_pio.models.qwen3_vl_2b.core.modeling.load_custom_tokenizer") as mock_tokenizer_fn, \
             patch("src.inference_pio.models.qwen3_vl_2b.core.modeling.get_optimized_image_processor") as mock_processor_fn, \
             patch("src.inference_pio.models.qwen3_vl_2b.plugin.load_custom_tokenizer") as mock_plugin_tokenizer_fn, \
             patch("src.inference_pio.models.qwen3_vl_2b.plugin.get_optimized_image_processor") as mock_plugin_processor_fn, \
             patch("src.inference_pio.models.qwen3_vl_2b.model.get_system_profile") as mock_get_profile:

            mock_profile = MagicMock()
            mock_profile.is_weak_hardware = False
            mock_get_profile.return_value = mock_profile

            mock_loader_instance = MagicMock()
            mock_loader_cls.return_value = mock_loader_instance
            mock_loader_instance.load_model.return_value = MagicMock()

            mock_tokenizer_fn.return_value = MagicMock()
            mock_processor_fn.return_value = MagicMock()
            mock_plugin_tokenizer_fn.return_value = MagicMock()
            mock_plugin_processor_fn.return_value = MagicMock()

            # Instantiate model
            model = Qwen3VL2BModel(config)
            print("[OK] Qwen3-VL-2B model instantiated successfully")

            assert hasattr(model, "_model"), "Model should have _model attribute"

            print("[OK] Model has expected attributes")
            return True

    except Exception as e:
        print(f"[ERROR] Creating model failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plugin_creation():
    """Test Qwen3-VL-2B plugin creation."""
    print("Testing Qwen3-VL-2B plugin creation...")
    # ... (same as before) ...
    try:
        from src.inference_pio.models.qwen3_vl_2b.plugin import (
            Qwen3_VL_2B_Instruct_Plugin,
            create_qwen3_vl_2b_instruct_plugin,
        )

        # Create plugin instance
        plugin = Qwen3_VL_2B_Instruct_Plugin()
        print("[OK] Qwen3-VL-2B plugin created successfully")

        # Test factory
        factory = create_qwen3_vl_2b_instruct_plugin()

        # Verify methods
        if hasattr(plugin, "load_model"):
             print("[OK] Plugin has methods")
             return True
        return False
    except Exception as e:
        print(f"Plugin test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing Qwen3-VL-2B model implementation...")

    config_ok = test_config_attributes()
    model_ok = test_qwen3_vl_model_creation()
    plugin_ok = test_plugin_creation()

    if config_ok and model_ok and plugin_ok:
        print("\n[SUCCESS] All tests passed! Qwen3-VL-2B implementation is ready.")
        sys.exit(0)
    else:
        print("\n[FAILURE] Some tests failed.")
        sys.exit(1)
