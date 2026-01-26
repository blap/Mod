"""
Test to verify that all models have been updated with streaming capabilities
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn as nn
from src.inference_pio.models.glm_4_7.model import GLM47Model
from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.qwen3_4b_instruct_2507.model import Qwen34BInstruct2507Model
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_coder_30b.model import Qwen3Coder30BModel
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig
from src.inference_pio.models.qwen3_vl_2b.model import Qwen3VL2BModel
from src.inference_pio.models.qwen3_vl_2b.config import Qwen3VL2BConfig


def test_glm47_streaming_methods():
    """Test that GLM-4-7 model has streaming methods."""
    print("Testing GLM-4-7 model streaming methods...")

    # Create a minimal config for testing
    config = GLM47Config(
        model_path="dummy_path",  # This will fail to load but that's OK for testing method existence
        torch_dtype="float16",
        device_map="cpu"
    )

    # Create model instance
    try:
        model = GLM47Model(config)
    except:
        # If model fails to initialize due to missing model files, that's OK for this test
        # We just want to check if the methods exist
        print("  Could not initialize GLM-4-7 model (expected for test), checking class methods...")
        # Check if the methods exist in the class
        assert hasattr(GLM47Model, 'setup_streaming_computation'), "setup_streaming_computation method not found"
        assert hasattr(GLM47Model, 'submit_stream_request'), "submit_stream_request method not found"
        assert hasattr(GLM47Model, 'generate_stream'), "generate_stream method not found"
        print("  All streaming methods found in GLM-4-7 class!")
        return

    # If model was created successfully, check if methods exist
    assert hasattr(model, 'setup_streaming_computation'), "setup_streaming_computation method not found"
    assert hasattr(model, 'submit_stream_request'), "submit_stream_request method not found"
    assert hasattr(model, 'generate_stream'), "generate_stream method not found"
    print("  All streaming methods found in GLM-4-7 instance!")


def test_qwen3_4b_streaming_methods():
    """Test that Qwen3-4b-instruct-2507 model has streaming methods."""
    print("Testing Qwen3-4b-instruct-2507 model streaming methods...")

    # Create a minimal config for testing
    config = Qwen34BInstruct2507Config(
        model_path="dummy_path",  # This will fail to load but that's OK for testing method existence
        torch_dtype="float16",
        device_map="cpu"
    )

    # Create model instance
    try:
        model = Qwen34BInstruct2507Model(config)
    except:
        # If model fails to initialize due to missing model files, that's OK for this test
        print("  Could not initialize Qwen3-4b-instruct-2507 model (expected for test), checking class methods...")
        # Check if the methods exist in the class
        assert hasattr(Qwen34BInstruct2507Model, 'setup_streaming_computation'), "setup_streaming_computation method not found"
        assert hasattr(Qwen34BInstruct2507Model, 'submit_stream_request'), "submit_stream_request method not found"
        assert hasattr(Qwen34BInstruct2507Model, 'generate_stream'), "generate_stream method not found"
        print("  All streaming methods found in Qwen3-4b-instruct-2507 class!")
        return

    # If model was created successfully, check if methods exist
    assert hasattr(model, 'setup_streaming_computation'), "setup_streaming_computation method not found"
    assert hasattr(model, 'submit_stream_request'), "submit_stream_request method not found"
    assert hasattr(model, 'generate_stream'), "generate_stream method not found"
    print("  All streaming methods found in Qwen3-4b-instruct-2507 instance!")


def test_qwen3_coder_streaming_methods():
    """Test that Qwen3-coder-30b model has streaming methods."""
    print("Testing Qwen3-coder-30b model streaming methods...")

    # Create a minimal config for testing
    config = Qwen3Coder30BConfig(
        model_path="dummy_path",  # This will fail to load but that's OK for testing method existence
        torch_dtype="float16",
        device_map="cpu"
    )

    # Create model instance
    try:
        model = Qwen3Coder30BModel(config)
    except:
        # If model fails to initialize due to missing model files, that's OK for this test
        print("  Could not initialize Qwen3-coder-30b model (expected for test), checking class methods...")
        # Check if the methods exist in the class
        assert hasattr(Qwen3Coder30BModel, 'setup_streaming_computation'), "setup_streaming_computation method not found"
        assert hasattr(Qwen3Coder30BModel, 'submit_stream_request'), "submit_stream_request method not found"
        assert hasattr(Qwen3Coder30BModel, 'generate_stream'), "generate_stream method not found"
        print("  All streaming methods found in Qwen3-coder-30b class!")
        return

    # If model was created successfully, check if methods exist
    assert hasattr(model, 'setup_streaming_computation'), "setup_streaming_computation method not found"
    assert hasattr(model, 'submit_stream_request'), "submit_stream_request method not found"
    assert hasattr(model, 'generate_stream'), "generate_stream method not found"
    print("  All streaming methods found in Qwen3-coder-30b instance!")


def test_qwen3_vl_streaming_methods():
    """Test that Qwen3-vl-2b model has streaming methods."""
    print("Testing Qwen3-vl-2b model streaming methods...")

    # Create a minimal config for testing
    config = Qwen3VL2BConfig(
        model_path="dummy_path",  # This will fail to load but that's OK for testing method existence
        torch_dtype="float16",
        device_map="cpu"
    )

    # Create model instance
    try:
        model = Qwen3VL2BModel(config)
    except:
        # If model fails to initialize due to missing model files, that's OK for this test
        print("  Could not initialize Qwen3-vl-2b model (expected for test), checking class methods...")
        # Check if the methods exist in the class
        assert hasattr(Qwen3VL2BModel, 'setup_streaming_computation'), "setup_streaming_computation method not found"
        assert hasattr(Qwen3VL2BModel, 'submit_stream_request'), "submit_stream_request method not found"
        assert hasattr(Qwen3VL2BModel, 'generate_stream'), "generate_stream method not found"
        print("  All streaming methods found in Qwen3-vl-2b class!")
        return

    # If model was created successfully, check if methods exist
    assert hasattr(model, 'setup_streaming_computation'), "setup_streaming_computation method not found"
    assert hasattr(model, 'submit_stream_request'), "submit_stream_request method not found"
    assert hasattr(model, 'generate_stream'), "generate_stream method not found"
    print("  All streaming methods found in Qwen3-vl-2b instance!")


def main():
    """Run all tests."""
    print("Testing that all models have been updated with streaming capabilities...\n")

    test_glm47_streaming_methods()
    print()

    test_qwen3_4b_streaming_methods()
    print()

    test_qwen3_coder_streaming_methods()
    print()

    test_qwen3_vl_streaming_methods()
    print()

    print("All models have been successfully updated with streaming computation capabilities!")


if __name__ == "__main__":
    main()