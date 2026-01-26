"""
Test for Qwen3-4B-Instruct-2507 model with intelligent unimodal pagination system.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import shutil
from pathlib import Path
import torch

from ..model import Qwen34BInstruct2507Model
from ..config import Qwen34BInstruct2507Config

# TestQwen34BInstruct2507ModelWithUnimodalPagination

    """Test cases for Qwen3-4B-Instruct-2507 model with unimodal pagination."""

    def setup_helper():
        """Set up test fixtures before each test method."""
        test_dir = tempfile.mkdtemp()
        swap_dir = Path(test_dir) / "text_tensor_swap"
        swap_dir.mkdir(exist_ok=True)

    def cleanup_helper():
        """Tear down test fixtures after each test method."""
        shutil.rmtree(test_dir)

    def model_initialization_with_unimodal_pagination(self)():
        """Test initializing the model with unimodal pagination enabled."""
        config = Qwen34BInstruct2507Config()
        
        # Enable unimodal pagination
        config.enable_intelligent_pagination = True
        config.pagination_swap_directory = str(swap_dir)
        config.pagination_page_size_mb = 8
        config.pagination_eviction_policy = "intelligent"
        config.pagination_max_memory_ratio = 0.6
        
        # Create model with pagination enabled
        model = Qwen34BInstruct2507Model(config)
        
        # Check that pagination system was initialized
        assert_is_not_none(model._pagination_system)
        assertIsNotNone(model._unimodal_pager)
        
        # Test pagination functionality
        test_tensor = torch.randn(10)
        success = model._unimodal_pager.page_tensor(
            test_tensor,
            "test_tensor",
            model._unimodal_pager.pagination_system.TextDataType.TEXT_EMBEDDINGS,
            priority=model._unimodal_pager.pagination_system.PaginationPriority.HIGH
        )
        
        assert_true(success)
        
        # Access the tensor
        retrieved_tensor = model._unimodal_pager.access_tensor("test_tensor")
        assert_is_not_none(retrieved_tensor)
        assertTrue(torch.equal(test_tensor))
        
        # Clean up
        model.cleanup()

    def unimodal_pagination_with_different_text_data_types(self)():
        """Test unimodal pagination with different text data types."""
        config = Qwen34BInstruct2507Config()
        
        # Enable unimodal pagination
        config.enable_intelligent_pagination = True
        config.pagination_swap_directory = str(swap_dir)
        config.pagination_page_size_mb = 8
        config.pagination_eviction_policy = "intelligent"
        config.pagination_max_memory_ratio = 0.6
        
        # Create model with pagination enabled
        model = Qwen34BInstruct2507Model(config)
        
        # Check that pagination system was initialized
        assertIsNotNone(model._pagination_system)
        assertIsNotNone(model._unimodal_pager)
        
        # Test different text data types
        from ...common.unimodal_tensor_pagination import TextDataType
        
        text_data_types = [
            TextDataType.TEXT_EMBEDDINGS,
            TextDataType.TEXT_ACTIVATIONS,
            TextDataType.TEXT_KV_CACHE,
            TextDataType.TEXT_ATTENTION_WEIGHTS,
            TextDataType.TEXT_MLP_WEIGHTS,
            TextDataType.TEXT_LAYERNORM_WEIGHTS,
            TextDataType.TEXT_INPUT_TOKENS,
            TextDataType.TEXT_OUTPUT_TOKENS,
            TextDataType.TEXT_HIDDEN_STATES
        ]
        
        for i, data_type in enumerate(text_data_types):
            tensor = torch.randn(5, 5)
            tensor_id = f"test_tensor_{data_type.value}_{i}"
            
            success = model._unimodal_pager.page_tensor(
                tensor,
                tensor_id,
                data_type,
                priority=model._unimodal_pager.pagination_system.PaginationPriority.MEDIUM
            )
            
            assert_true(success)
            
            retrieved = model._unimodal_pager.access_tensor(tensor_id)
            assert_is_not_none(retrieved)
            assert_true(torch.equal(tensor), f"Retrieved {data_type.value} tensor differs from original")
        
        # Clean up
        model.cleanup()

    def unimodal_pagination_priority_handling(self)():
        """Test priority handling in unimodal pagination."""
        config = Qwen34BInstruct2507Config()
        
        # Enable unimodal pagination
        config.enable_intelligent_pagination = True
        config.pagination_swap_directory = str(swap_dir)
        config.pagination_page_size_mb = 8
        config.pagination_eviction_policy = "intelligent"
        config.pagination_max_memory_ratio = 0.6
        
        # Create model with pagination enabled
        model = Qwen34BInstruct2507Model(config)
        
        # Check that pagination system was initialized
        assert_is_not_none(model._pagination_system)
        assertIsNotNone(model._unimodal_pager)
        
        # Test priority handling
        from ...common.unimodal_tensor_pagination import TextDataType)
        tensor_id = "priority_test"
        
        success = model._unimodal_pager.page_tensor(
            test_tensor,
            tensor_id,
            TextDataType.TEXT_EMBEDDINGS,
            priority=PaginationPriority.LOW
        )
        
        assert_true(success)
        
        # Check initial priority
        current_priority = model._unimodal_pager.get_tensor_priority(tensor_id)
        assert_equal(current_priority)
        
        # Update priority
        model._unimodal_pager.set_tensor_priority(tensor_id, PaginationPriority.HIGH)
        updated_priority = model._unimodal_pager.get_tensor_priority(tensor_id)
        assert_equal(updated_priority, PaginationPriority.HIGH)
        
        # Clean up
        model.cleanup()

    def unimodal_pagination_pin_unpin_functionality(self)():
        """Test pin/unpin functionality in unimodal pagination."""
        config = Qwen34BInstruct2507Config()
        
        # Enable unimodal pagination
        config.enable_intelligent_pagination = True
        config.pagination_swap_directory = str(swap_dir)
        config.pagination_page_size_mb = 8
        config.pagination_eviction_policy = "intelligent"
        config.pagination_max_memory_ratio = 0.6
        
        # Create model with pagination enabled
        model = Qwen34BInstruct2507Model(config)
        
        # Check that pagination system was initialized
        assert_is_not_none(model._pagination_system)
        assertIsNotNone(model._unimodal_pager)
        
        # Test pin/unpin functionality
        from ...common.unimodal_tensor_pagination import TextDataType)
        tensor_id = "pin_test"
        
        success = model._unimodal_pager.page_tensor(
            test_tensor,
            tensor_id,
            TextDataType.TEXT_EMBEDDINGS,
            priority=PaginationPriority.HIGH
        )
        
        assert_true(success)
        
        # Pin the tensor
        pin_success = model._unimodal_pager.pin_tensor(tensor_id)
        assertTrue(pin_success)
        
        # Unpin the tensor
        unpin_success = model._unimodal_pager.unpin_tensor(tensor_id)
        assertTrue(unpin_success)
        
        # Clean up
        model.cleanup()

if __name__ == '__main__':
    run_tests(test_functions)