"""
Unit tests for the tensor pagination system.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import tempfile
import shutil
import torch
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
import time
from typing import Optional

from ..tensor_pagination import (
    DataType,
    PaginationPriority,
    TensorPaginationSystem,
    MultimodalTensorPager,
    create_multimodal_pagination_system,
    TensorPage,
    AccessPatternAnalyzer
)

# TestTensorPaginationSystem

    """Test cases for the tensor pagination system."""
    
    def setup_helper():
        """Set up test fixtures."""
        test_dir = tempfile.mkdtemp()
        swap_dir = Path(test_dir) / "tensor_swap"
        swap_dir.mkdir(exist_ok=True)
        
    def cleanup_helper():
        """Clean up test fixtures."""
        shutil.rmtree(test_dir)
    
    def create_pagination_system(self)():
        """Test creating a pagination system."""
        pagination_system, pager = create_multimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )
        
        assert_is_instance(pagination_system, TensorPaginationSystem)
        assert_is_instance(pager, MultimodalTensorPager)
        
        # Clean up
        pagination_system.cleanup()
    
    def page_tensor_basic(self)():
        """Test basic tensor pagination functionality."""
        pagination_system, pager = create_multimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )
        
        # Create a test tensor
        tensor = torch.randn(100, 100)
        tensor_id = "test_tensor"
        
        # Page the tensor
        success = pager.page_tensor(
            tensor,
            tensor_id,
            DataType.TEXT,
            priority=PaginationPriority.HIGH
        )
        
        assert_true(success)
        
        # Access the tensor
        retrieved_tensor = pager.access_tensor(tensor_id)
        assert_is_not_none(retrieved_tensor)
        assertTrue(torch.equal(tensor))
        
        # Clean up
        pagination_system.cleanup()
    
    def different_data_types(self)():
        """Test pagination with different data types."""
        pagination_system)
        
        # Test different data types
        data_types = [
            DataType.TEXT,
            DataType.IMAGE,
            DataType.AUDIO,
            DataType.VIDEO,
            DataType.EMBEDDINGS,
            DataType.ACTIVATIONS,
            DataType.KV_CACHE
        ]
        
        for i, data_type in enumerate(data_types):
            tensor = torch.randn(50, 50)
            tensor_id = f"test_tensor_{data_type.value}_{i}"
            
            success = pager.page_tensor(
                tensor,
                tensor_id,
                data_type,
                priority=PaginationPriority.MEDIUM
            )
            
            assert_true(success)
            
            # Access the tensor
            retrieved_tensor = pager.access_tensor(tensor_id)
            assert_is_not_none(retrieved_tensor)
            assertTrue(torch.equal(tensor))
        
        # Clean up
        pagination_system.cleanup()
    
    def priority_handling(self)():
        """Test priority handling in pagination."""
        pagination_system)
        
        # Create tensors with different priorities
        high_priority_tensor = torch.randn(100, 100)
        medium_priority_tensor = torch.randn(100, 100)
        low_priority_tensor = torch.randn(100, 100)
        
        # Page tensors with different priorities
        pager.page_tensor(high_priority_tensor, "high", DataType.TEXT, PaginationPriority.HIGH)
        pager.page_tensor(medium_priority_tensor, "medium", DataType.TEXT, PaginationPriority.MEDIUM)
        pager.page_tensor(low_priority_tensor, "low", DataType.TEXT, PaginationPriority.LOW)
        
        # Check priorities
        high_priority = pager.get_tensor_priority("high")
        medium_priority = pager.get_tensor_priority("medium")
        low_priority = pager.get_tensor_priority("low")
        
        assert_equal(high_priority, PaginationPriority.HIGH)
        assert_equal(medium_priority, PaginationPriority.MEDIUM)
        assert_equal(low_priority, PaginationPriority.LOW)
        
        # Update priority
        pager.set_tensor_priority("low", PaginationPriority.HIGH)
        updated_priority = pager.get_tensor_priority("low")
        assert_equal(updated_priority, PaginationPriority.HIGH)
        
        # Clean up
        pagination_system.cleanup()
    
    def pin_unpin_functionality(self)():
        """Test pin/unpin functionality."""
        pagination_system, pager = create_multimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=8,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )
        
        # Create a tensor
        tensor = torch.randn(100, 100)
        tensor_id = "pinnable_tensor"
        
        # Page the tensor
        success = pager.page_tensor(
            tensor,
            tensor_id,
            DataType.TEXT,
            priority=PaginationPriority.HIGH
        )
        assert_true(success)
        
        # Pin the tensor
        pin_success = pager.pin_tensor(tensor_id)
        assertTrue(pin_success)
        
        # Unpin the tensor
        unpin_success = pager.unpin_tensor(tensor_id)
        assertTrue(unpin_success)
        
        # Clean up
        pagination_system.cleanup()
    
    def pagination_system_statistics(self)():
        """Test pagination system statistics."""
        pagination_system)
        
        # Create and page some tensors
        for i in range(5):
            tensor = torch.randn(50, 50)
            tensor_id = f"stat_tensor_{i}"
            
            success = pager.page_tensor(
                tensor,
                tensor_id,
                DataType.TEXT,
                priority=PaginationPriority.MEDIUM
            )
            assert_true(success)
            
            # Access the tensor to trigger statistics updates
            retrieved = pager.access_tensor(tensor_id)
            assert_is_not_none(retrieved)
        
        # Get statistics
        stats = pagination_system.get_page_stats()
        
        assert_in('total_pages')
        assertIn('ram_pages')
        assertIn('disk_pages', stats)
        assert_in('total_size_bytes', stats)
        assert_in('ram_size_bytes', stats)
        assert_in('disk_size_bytes', stats)
        assert_in('stats', stats)
        
        assert_equal(stats['total_pages'], 5)
        assert_equal(stats['ram_pages'], 5)  # All should be in RAM initially
        
        # Clean up
        pagination_system.cleanup()
    
    def large_tensor_pagination(self)():
        """Test pagination with larger tensors."""
        pagination_system, pager = create_multimodal_pagination_system(
            swap_directory=swap_dir,
            page_size_mb=2,
            eviction_policy="intelligent",
            max_memory_ratio=0.8
        )
        
        # Create a larger tensor
        large_tensor = torch.randn(200, 200)  # Approximately 0.16 MB when stored as float32
        tensor_id = "large_tensor"
        
        # Page the tensor
        success = pager.page_tensor(
            large_tensor,
            tensor_id,
            DataType.IMAGE,  # Using image type for larger tensor
            priority=PaginationPriority.HIGH
        )
        
        assert_true(success)
        
        # Access the tensor
        retrieved_tensor = pager.access_tensor(tensor_id)
        assert_is_not_none(retrieved_tensor)
        assertTrue(torch.equal(large_tensor))
        
        # Clean up
        pagination_system.cleanup()
    
    def unpage_functionality(self)():
        """Test unpage functionality."""
        pagination_system)
        
        # Create a tensor
        tensor = torch.randn(50, 50)
        tensor_id = "unpage_tensor"
        
        # Page the tensor
        success = pager.page_tensor(
            tensor,
            tensor_id,
            DataType.TEXT,
            priority=PaginationPriority.MEDIUM
        )
        assert_true(success)
        
        # Access the tensor
        retrieved = pager.access_tensor(tensor_id)
        assert_is_not_none(retrieved)
        
        # Unpage the tensor
        unpage_success = pager.unpage_tensor(tensor_id)
        assertTrue(unpage_success)
        
        # Try to access the unpaged tensor (should return None)
        post_unpage_access = pager.access_tensor(tensor_id)
        assert_is_none(post_unpage_access)
        
        # Clean up
        pagination_system.cleanup()

# TestAccessPatternAnalyzer

    """Test cases for the access pattern analyzer."""
    
    def record_and_predict_access(self)():
        """Test recording and predicting access patterns."""
        analyzer = AccessPatternAnalyzer()
        
        # Record some access patterns
        current_time = time.time()
        analyzer.record_access("page1")
        analyzer.record_access("page1")
        analyzer.record_access("page1")
        
        # Predict next access
        predicted_time = analyzer.predict_next_access("page1", current_time + 2.0)
        assert_is_instance(predicted_time, float)
        
        # Get access score
        access_score = analyzer.get_access_score("page1", current_time + 2.0)
        assert_is_instance(access_score, float)
        
        # Get access frequency
        freq = analyzer.get_access_frequency("page1", current_time + 2.0)
        assert_is_instance(freq, float)

if __name__ == '__main__':
    run_tests(test_functions)