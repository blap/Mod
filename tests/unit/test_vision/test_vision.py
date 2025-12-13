"""
Basic test for vision functionality
"""
import pytest
import torch
from vision.hierarchical_vision_processor import HierarchicalVisionProcessor


def test_hierarchical_vision_processor():
    """Test hierarchical vision processor functionality."""
    processor = HierarchicalVisionProcessor()
    assert processor is not None