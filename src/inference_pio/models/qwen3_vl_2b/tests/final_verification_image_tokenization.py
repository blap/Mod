"""
Final Verification Test for Image Tokenization System in Qwen3-VL-2B Model

This test verifies that the efficient image tokenization system is properly
integrated with the Qwen3-VL-2B model and functions as expected.
"""

import sys
sys.path.insert(0, r'C:\Users\Admin\Documents\GitHub\Mod')

from src.inference_pio.common.image_tokenization import ImageTokenizationConfig, ImageTokenizer
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from PIL import Image
import torch
from unittest.mock import patch

def test_image_tokenization_creation():
    """Test creating the image tokenization system."""
    print("Testing image tokenization system creation...")
    
    config = ImageTokenizationConfig(
        image_size=448,
        patch_size=14,
        max_image_tokens=1024,
        token_dim=1024
    )
    
    tokenizer = ImageTokenizer(config)
    
    assert tokenizer is not None, "Image tokenizer should be created"
    assert tokenizer.config == config, "Config should match"
    
    print("PASS: Image tokenization system created successfully")


def test_image_tokenization_functionality():
    """Test the image tokenization functionality."""
    print("Testing image tokenization functionality...")
    
    config = ImageTokenizationConfig(
        image_size=224,
        patch_size=16,
        max_image_tokens=256,
        token_dim=512
    )
    
    tokenizer = ImageTokenizer(config)
    
    # Create a test image
    image = Image.new('RGB', (224, 224), color='red')
    
    # Tokenize the image
    result = tokenizer.tokenize(image)
    
    assert 'pixel_values' in result, "Result should contain pixel_values"
    pixel_values = result['pixel_values']
    assert pixel_values.dim() == 2, f"Expected 2D tensor for Qwen format, got {pixel_values.dim()}D"
    assert pixel_values.shape[0] >= 1, "Should have at least one patch"
    assert pixel_values.shape[1] >= 1, "Each patch should have at least one dimension"
    
    print(f"PASS: Image tokenized successfully, shape: {pixel_values.shape}")


def test_qwen3_vl_2b_integration():
    """Test integration with Qwen3-VL-2B model."""
    print("Testing Qwen3-VL-2B model integration...")
    
    # Create config
    model_config = Qwen3VL2BConfig()
    model_config.model_path = "dummy_path"
    model_config.use_flash_attention_2 = False
    model_config.use_cuda_kernels = False
    model_config.enable_disk_offloading = False
    model_config.enable_intelligent_pagination = False
    model_config.enable_tensor_parallelism = False
    model_config.use_multimodal_attention = False
    model_config.enable_image_tokenization = True  # Enable image tokenization
    
    # Mock the model loading
    with patch('src.inference_pio.models.qwen3_vl_2b.model.AutoModelForVision2Seq.from_pretrained') as mock_model, \
         patch('src.inference_pio.models.qwen3_vl_2b.model.AutoTokenizer.from_pretrained') as mock_tokenizer, \
         patch('src.inference_pio.models.qwen3_vl_2b.model.AutoImageProcessor.from_pretrained') as mock_image_proc:
        
        # Set up mocks
        mock_model_instance = type('MockModel', (), {})()
        mock_model_instance.gradient_checkpointing_enable = lambda: None
        mock_model_instance.config = type('MockConfig', (), {})()
        mock_model_instance.config.hidden_size = 2048
        mock_model_instance.config.num_attention_heads = 16
        mock_model_instance.config.num_hidden_layers = 24
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = type('MockTokenizer', (), {})()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = '</s>'
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_image_proc_instance = type('MockImageProc', (), {})()
        mock_image_proc.return_value = mock_image_proc_instance
        
        # Create the model
        model = Qwen3VL2BModel(model_config)
        
        # Verify integration
        assert model._image_tokenizer is not None, "Image tokenizer should be initialized"
        assert hasattr(model._model, 'image_tokenizer'), "Model should have image_tokenizer attribute"
        assert model._model.image_tokenizer is not None, "Model's image_tokenizer should not be None"
        
        print("PASS: Qwen3-VL-2B model integration successful")


def test_plugin_integration():
    """Test integration with the Qwen3-VL-2B plugin."""
    print("Testing plugin integration...")

    # Just verify that the plugin has the required methods after our modifications
    from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin

    # Check that the class has the required methods
    plugin_cls = Qwen3_VL_2B_Instruct_Plugin

    # Test tokenize_image method exists
    assert hasattr(plugin_cls, 'tokenize_image'), "Plugin should have tokenize_image method"
    assert callable(getattr(plugin_cls, 'tokenize_image')), "tokenize_image should be callable"

    # Test batch_tokenize_images method exists
    assert hasattr(plugin_cls, 'batch_tokenize_images'), "Plugin should have batch_tokenize_images method"
    assert callable(getattr(plugin_cls, 'batch_tokenize_images')), "batch_tokenize_images should be callable"

    print("PASS: Plugin integration successful")


def test_performance_metrics():
    """Test performance metrics functionality."""
    print("Testing performance metrics...")
    
    config = ImageTokenizationConfig()
    tokenizer = ImageTokenizer(config)
    
    # Check initial metrics
    initial_stats = tokenizer.get_performance_stats()
    assert 'total_tokenization_time' in initial_stats
    assert 'num_tokenization_calls' in initial_stats
    assert 'average_tokenization_time' in initial_stats
    assert initial_stats['num_tokenization_calls'] == 0
    
    # Reset metrics
    tokenizer.reset_performance_stats()
    reset_stats = tokenizer.get_performance_stats()
    assert reset_stats['num_tokenization_calls'] == 0
    assert reset_stats['total_tokenization_time'] == 0.0
    
    print("PASS: Performance metrics functionality verified")


def run_all_tests():
    """Run all verification tests."""
    print("Running final verification tests for Image Tokenization System...\n")
    
    test_image_tokenization_creation()
    print()
    
    test_image_tokenization_functionality()
    print()
    
    test_qwen3_vl_2b_integration()
    print()
    
    test_plugin_integration()
    print()
    
    test_performance_metrics()
    print()
    
    print("SUCCESS: All verification tests passed! The Image Tokenization System is successfully integrated with Qwen3-VL-2B.")


if __name__ == "__main__":
    run_all_tests()