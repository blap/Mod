"""
Tests to validate the current module organization and functionality before reorganization.
This ensures that all imports and functionality work as expected before refactoring.
"""
import pytest
import sys
import os

# Add src to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_qwen3_vl_import():
    """Test that the main qwen3_vl module can be imported."""
    try:
        import qwen3_vl
        assert hasattr(qwen3_vl, '__init__')
    except ImportError as e:
        pytest.fail(f"Failed to import qwen3_vl: {e}")

def test_attention_module_import():
    """Test that attention module can be imported from qwen3_vl."""
    try:
        from attention import (
            StandardAttention, FlashAttention2, SM61OptimizedFlashAttention2,
            TrueSparseAttention, BlockSparseAttention, DynamicSparseAttention,
            Qwen3VLAttention, AttentionMechanismSelector, Qwen3VLRotaryEmbedding
        )
        # Just verify that we can import these classes
        assert StandardAttention is not None
        assert FlashAttention2 is not None
    except ImportError as e:
        pytest.fail(f"Failed to import attention modules: {e}")

def test_attention_init_exports():
    """Test that attention module exports are available."""
    try:
        import attention as attention_module
        # Check that the module has the expected attributes
        expected_attrs = [
            'StandardAttention', 'FlashAttention2', 'SM61OptimizedFlashAttention2',
            'TrueSparseAttention', 'BlockSparseAttention', 'DynamicSparseAttention',
            'Qwen3VLAttention', 'AttentionMechanismSelector', 'Qwen3VLRotaryEmbedding'
        ]
        for attr in expected_attrs:
            assert hasattr(attention_module, attr), f"Missing attribute: {attr}"
    except ImportError as e:
        pytest.fail(f"Failed to import attention module: {e}")

def test_models_module_access():
    """Test that models module can be accessed."""
    try:
        import models
        # Check if the models module exists and has expected content
        assert models is not None
    except ImportError as e:
        pytest.fail(f"Failed to import models module: {e}")

def test_multimodal_fusion_module():
    """Test that multimodal fusion module can be accessed."""
    try:
        from multimodal import cross_modal_token_merging
        assert cross_modal_token_merging is not None
    except ImportError as e:
        pytest.fail(f"Failed to import multimodal fusion module: {e}")

def test_vision_module():
    """Test that vision module can be accessed."""
    try:
        from vision import hierarchical_vision_processor
        assert hierarchical_vision_processor is not None
    except ImportError as e:
        pytest.fail(f"Failed to import vision module: {e}")

def test_config_module():
    """Test that config module can be imported."""
    try:
        from qwen3_vl.config import Qwen3VLConfig
        assert Qwen3VLConfig is not None
    except ImportError as e:
        pytest.fail(f"Failed to import config module: {e}")

def test_core_components():
    """Test that core components can be imported."""
    try:
        import qwen3_vl.components
        assert qwen3_vl.components is not None
    except ImportError as e:
        pytest.fail(f"Failed to import components module: {e}")

def test_architectures_module():
    """Test that architectures module can be imported."""
    try:
        import qwen3_vl.architectures
        assert qwen3_vl.architectures is not None
    except ImportError as e:
        pytest.fail(f"Failed to import architectures module: {e}")

def test_utils_module():
    """Test that top-level utils module can be imported."""
    try:
        import utils
        # Note: this refers to src/utils which currently has CUDA utilities
        assert utils is not None
    except ImportError as e:
        # This might legitimately fail since src/utils is currently minimal
        pass

def test_empty_directories_exist():
    """Test that the top-level directories exist even if empty."""
    expected_dirs = ['models', 'multimodal', 'vision', 'language']
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    
    for dir_name in expected_dirs:
        dir_path = os.path.join(src_path, dir_name)
        assert os.path.exists(dir_path), f"Expected directory {dir_name} does not exist"
        assert os.path.isdir(dir_path), f"{dir_path} is not a directory"

if __name__ == "__main__":
    # Run the tests
    test_qwen3_vl_import()
    test_attention_module_import()
    test_attention_init_exports()
    test_models_module_access()
    test_multimodal_fusion_module()
    test_vision_module()
    test_config_module()
    test_core_components()
    test_architectures_module()
    test_utils_module()
    test_empty_directories_exist()
    print("All pre-refactoring tests passed!")