"""
Test for Qwen3-VL-2B model with intelligent pagination system.
"""
from tests.utils.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import shutil
import torch
from pathlib import Path

from .config import Qwen3VL2BConfig
from .model import Qwen3VL2BModel

# TestQwen3VL2BModelWithPagination

    """Test cases for Qwen3-VL-2B model with pagination."""
    
    def setup_helper():
        """Set up test fixtures."""
        test_dir = tempfile.mkdtemp()
        
    def cleanup_helper():
        """Clean up test fixtures."""
        shutil.rmtree(test_dir)
    
    def model_initialization_with_pagination(self)():
        """Test initializing the model with pagination enabled."""
        config = Qwen3VL2BConfig()
        
        # Override model path to avoid downloading
        config.model_path = "dummy_path"  # This will cause fallback to HuggingFace name
        
        # Enable pagination
        config.enable_intelligent_pagination = True
        config.pagination_swap_directory = str(Path(test_dir) / "tensor_swap")
        
        # Disable other heavy optimizations for faster testing
        config.use_flash_attention_2 = False
        config.use_sparse_attention = False
        config.use_sliding_window_attention = False
        config.use_paged_attention = False
        config.enable_disk_offloading = False
        config.enable_activation_offloading = False
        config.use_tensor_decomposition = False
        config.use_structured_pruning = False
        config.use_ml_optimizations = False
        config.use_modular_optimizations = False
        
        # Mock the model loading to avoid actual download
        .mock as mock
        with mock.patch('transformers.AutoModelForVision2Seq.from_pretrained') as mock_model, \
             mock.patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             mock.patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            
            # Create mock model with minimal required attributes
            mock_model.return_value = mock.Mock()
            mock_model.return_value.gradient_checkpointing_enable = mock.Mock()
            mock_model.return_value.generate = mock.Mock(return_value=torch.tensor([[1, 2, 3]]))
            mock_model.return_value.config = mock.Mock()
            mock_model.return_value.config.hidden_size = 2048
            mock_model.return_value.config.num_attention_heads = 16
            
            # Create mock tokenizer
            mock_tokenizer.return_value = mock.Mock()
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.pad_token = 151643
            mock_tokenizer.return_value.eos_token = 151643
            mock_tokenizer.return_value.encode = mock.Mock(return_value=[1, 2, 3])
            
            # Create mock processor
            mock_processor.return_value = mock.Mock()
            
            # Initialize the model
            model = Qwen3VL2BModel(config)
            
            # Check that pagination system was initialized
            assert_is_not_none(model._pagination_system)
            assertIsNotNone(model._multimodal_pager)
            
            # Test pagination functionality
            test_tensor = torch.randn(10)
            success = model._multimodal_pager.page_tensor(
                test_tensor,
                "test_tensor",
                model._multimodal_pager.DataType.TEXT,
                model._multimodal_pager.PaginationPriority.HIGH
            )
            assert_true(success)
            
            # Access the tensor
            retrieved = model._multimodal_pager.access_tensor("test_tensor")
            assert_is_not_none(retrieved)
            assertTrue(torch.equal(test_tensor))
            
            # Clean up
            model.cleanup()
    
    def pagination_with_different_modalities(self)():
        """Test pagination with different data modalities."""
        config = Qwen3VL2BConfig()
        
        # Override model path to avoid downloading
        config.model_path = "dummy_path"
        
        # Enable pagination
        config.enable_intelligent_pagination = True
        config.pagination_swap_directory = str(Path(test_dir) / "tensor_swap")
        
        # Disable other heavy optimizations for faster testing
        config.use_flash_attention_2 = False
        config.use_sparse_attention = False
        config.use_sliding_window_attention = False
        config.use_paged_attention = False
        config.enable_disk_offloading = False
        config.enable_activation_offloading = False
        config.use_tensor_decomposition = False
        config.use_structured_pruning = False
        config.use_ml_optimizations = False
        config.use_modular_optimizations = False
        
        # Mock the model loading to avoid actual download
        .mock as mock
        with mock.patch('transformers.AutoModelForVision2Seq.from_pretrained') as mock_model) as mock_tokenizer, \
             mock.patch('transformers.AutoImageProcessor.from_pretrained') as mock_processor:
            
            # Create mock model
            mock_model.return_value = mock.Mock()
            mock_model.return_value.gradient_checkpointing_enable = mock.Mock()
            mock_model.return_value.generate = mock.Mock(return_value=torch.tensor([[1, 2, 3]]))
            mock_model.return_value.config = mock.Mock()
            mock_model.return_value.config.hidden_size = 2048
            mock_model.return_value.config.num_attention_heads = 16
            
            # Create mock tokenizer
            mock_tokenizer.return_value = mock.Mock()
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.pad_token = 151643
            mock_tokenizer.return_value.eos_token = 151643
            mock_tokenizer.return_value.encode = mock.Mock(return_value=[1, 2, 3])
            
            # Create mock processor
            mock_processor.return_value = mock.Mock()
            
            # Initialize the model
            model = Qwen3VL2BModel(config)
            
            # Test different modalities
            modalities = [
                (torch.randn(10, 10), "text_tensor", model._multimodal_pager.DataType.TEXT),
                (torch.randn(3, 224, 224), "image_tensor", model._multimodal_pager.DataType.IMAGE),
                (torch.randn(100, 128), "audio_tensor", model._multimodal_pager.DataType.AUDIO),
                (torch.randn(50, 512), "embedding_tensor", model._multimodal_pager.DataType.EMBEDDINGS),
                (torch.randn(20, 256), "activation_tensor", model._multimodal_pager.DataType.ACTIVATIONS),
                (torch.randn(16, 8, 100, 64), "kv_cache_tensor", model._multimodal_pager.DataType.KV_CACHE),
            ]
            
            for tensor, tensor_id, data_type in modalities:
                success = model._multimodal_pager.page_tensor(
                    tensor,
                    tensor_id,
                    data_type,
                    model._multimodal_pager.PaginationPriority.MEDIUM
                )
                assert_true(success)
                
                # Access the tensor
                retrieved = model._multimodal_pager.access_tensor(tensor_id)
                assert_is_not_none(retrieved)
                assertTrue(torch.equal(tensor))
            
            # Clean up
            model.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)