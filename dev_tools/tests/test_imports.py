#!/usr/bin/env python
"""
Test script to verify that all import paths have been updated correctly.
"""
import sys
import os

# Add the src directory to the Python path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    print("Testing import paths after updates...")

    # Test core imports
    try:
        from models.config import Qwen3VLConfig
        print("src.qwen3_vl.models.config import successful")
    except ImportError as e:
        print(f"Failed to import models.config: {e}")

    try:
        from models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        print("src.qwen3_vl.models.modeling_qwen3_vl import successful")
    except ImportError as e:
        print(f"Failed to import models.modeling_qwen3_vl: {e}")

    # Test components imports
    try:
        from src.qwen3_vl.components.configuration import Qwen3VLConfig as ComponentQwen3VLConfig
        print("src.qwen3_vl.components.configuration import successful")
    except ImportError as e:
        print(f"Failed to import src.qwen3_vl.components.configuration: {e}")

    try:
        from src.qwen3_vl.components.attention.attention_mechanisms import Qwen3VLAttention
        print("src.qwen3_vl.components.attention.attention_mechanisms import successful")
    except ImportError as e:
        print(f"Failed to import src.qwen3_vl.components.attention.attention_mechanisms: {e}")

    # Test optimization imports
    try:
        from src.qwen3_vl.optimization.cpu_optimizations import OptimizedDataLoader
        print("src.qwen3_vl.optimization.cpu_optimizations import successful")
    except ImportError as e:
        print(f"Failed to import src.qwen3_vl.optimization.cpu_optimizations: {e}")

    # Test core imports
    try:
        from src.qwen3_vl.core.config import Qwen3VLConfig as CoreQwen3VLConfig
        print("src.qwen3_vl.core.config import successful")
    except ImportError as e:
        print(f"Failed to import src.qwen3_vl.core.config: {e}")

    # Test utils imports
    try:
        from src.qwen3_vl.utils.cuda_error_handler import CUDAErrorHandler
        print("src.qwen3_vl.utils.cuda_error_handler import successful")
    except ImportError as e:
        print(f"Failed to import src.qwen3_vl.utils.cuda_error_handler: {e}")

    # Test vision imports
    try:
        from vision.hierarchical_vision_processor import HierarchicalVisionProcessor
        print("src.qwen3_vl.vision.hierarchical_vision_processor import successful")
    except ImportError as e:
        print(f"Failed to import vision.hierarchical_vision_processor: {e}")

    # Test multimodal imports
    try:
        from src.qwen3_vl.multimodal.cross_modal_compression import CrossModalMemoryCompressor
        print("src.qwen3_vl.multimodal.cross_modal_compression import successful")
    except ImportError as e:
        print(f"Failed to import src.qwen3_vl.multimodal.cross_modal_compression: {e}")

    print("\nImport testing completed!")

if __name__ == "__main__":
    test_imports()