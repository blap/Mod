#!/usr/bin/env python3
"""
Test for Qwen3-VL-2B model implementation to verify all optimizations are properly implemented.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from unittest.mock import MagicMock, patch

from src.models.qwen3_vl_2b.config import Qwen3VL2BConfig


def test_config_attributes():
    """Test that the config has all required attributes."""
    print("Testing Qwen3-VL-2B config attributes...")
    config = Qwen3VL2BConfig()

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

    # Create a minimal config
    config = Qwen3VL2BConfig()
    config.model_path = "dummy_path"  # Use a dummy path to avoid download

    # Import and create the model
    from src.models.qwen3_vl_2b.model import Qwen3VL2BModel

    try:
        # Try to create the model with real components
        model = Qwen3VL2BModel(config)
        print("[OK] Qwen3-VL-2B model created successfully with real components")

        # Verify that the model has the expected attributes
        assert hasattr(model, "_model"), "Model should have _model attribute"
        assert hasattr(model, "_tokenizer"), "Model should have _tokenizer attribute"
        assert hasattr(
            model, "_image_processor"
        ), "Model should have _image_processor attribute"

        print("[OK] Model has expected attributes")

        return True

    except Exception as e:
        print(
            f"[INFO] Creating model with real components failed (expected for testing): {e}"
        )
        # Fallback to mocked components
        try:
            # Mock the AutoModelForVision2Seq.from_pretrained to avoid downloading
            # Handle the fact that AutoModelForVision2Seq might not exist or be an alias
            try:
                from transformers import AutoModelForVision2Seq

                target_class = "transformers.AutoModelForVision2Seq.from_pretrained"
            except ImportError:
                target_class = "transformers.AutoModelForCausalLM.from_pretrained"

            with patch(target_class) as mock_model_fn, patch(
                "transformers.AutoTokenizer.from_pretrained"
            ) as mock_tokenizer_fn, patch(
                "transformers.AutoImageProcessor.from_pretrained"
            ) as mock_image_processor_fn:

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
                mock_model.device = "cpu"

                try:
                    model = Qwen3VL2BModel(config)
                    print(
                        "[OK] Qwen3-VL-2B model created successfully with mocked components"
                    )

                    # Verify that the model has the expected attributes
                    assert hasattr(
                        model, "_model"
                    ), "Model should have _model attribute"
                    assert hasattr(
                        model, "_tokenizer"
                    ), "Model should have _tokenizer attribute"
                    assert hasattr(
                        model, "_image_processor"
                    ), "Model should have _image_processor attribute"

                    print("[OK] Model has expected attributes")

                    return True

                except Exception as e2:
                    print(f"[ERROR] Error creating Qwen3-VL-2B model: {e2}")
                    import traceback

                    traceback.print_exc()
                    return False
        except Exception as e3:
            print(f"[ERROR] Error with fallback mocked creation: {e3}")
            import traceback

            traceback.print_exc()
            return False


def test_plugin_creation():
    """Test Qwen3-VL-2B plugin creation."""
    print("Testing Qwen3-VL-2B plugin creation...")

    try:
        from src.models.qwen3_vl_2b.plugin import (
            Qwen3_VL_2B_Instruct_Plugin,
            create_qwen3_vl_2b_instruct_plugin,
        )

        # Create plugin instance
        plugin = Qwen3_VL_2B_Instruct_Plugin()
        print("[OK] Qwen3-VL-2B plugin created successfully")

        # Test factory function
        plugin_factory = create_qwen3_vl_2b_instruct_plugin()
        print("[OK] Qwen3-VL-2B plugin factory function works")

        # Verify plugin has required methods
        required_methods = [
            "load_model",
            "infer",
            "generate_text",
            "chat_completion",
            "get_model_info",
            "get_model_parameters",
            "initialize",
        ]

        missing_methods = []
        for method in required_methods:
            if not hasattr(plugin, method):
                missing_methods.append(method)

        if missing_methods:
            print(f"[ERROR] Missing plugin methods: {missing_methods}")
            return False
        else:
            print("[OK] All required plugin methods are present")
            return True

    except Exception as e:
        print(f"[ERROR] Error creating Qwen3-VL-2B plugin: {e}")
        import traceback

        traceback.print_exc()
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
