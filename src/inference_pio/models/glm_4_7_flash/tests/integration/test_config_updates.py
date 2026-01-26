"""
Final test to verify async unimodal processing with updated configurations.
"""
from src.inference_pio.test_utils import (assert_equal, assert_not_equal, assert_true, assert_false, assert_is_none, assert_is_not_none, assert_in, assert_not_in, assert_greater, assert_less, assert_is_instance, assert_raises, run_tests)


import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from src.inference_pio.models.glm_4_7.config import GLM47Config
from src.inference_pio.models.qwen3_4b_instruct_2507.config import Qwen34BInstruct2507Config
from src.inference_pio.models.qwen3_coder_30b.config import Qwen3Coder30BConfig

def test_config_async_params():
    """Test that all configs have the async processing parameters."""
    print("Testing config async processing parameters...")
    
    # Test GLM-4-7 config
    glm_config = GLM47Config()
    assert hasattr(glm_config, 'enable_async_unimodal_processing'), "GLM config missing enable_async_unimodal_processing"
    assert hasattr(glm_config, 'async_max_concurrent_requests'), "GLM config missing async_max_concurrent_requests"
    assert hasattr(glm_config, 'async_buffer_size'), "GLM config missing async_buffer_size"
    assert hasattr(glm_config, 'async_batch_timeout'), "GLM config missing async_batch_timeout"
    assert hasattr(glm_config, 'enable_async_batching'), "GLM config missing enable_async_batching"
    assert hasattr(glm_config, 'async_processing_device'), "GLM config missing async_processing_device"
    print("[PASS] GLM-4-7 config has all async processing parameters")

    # Test Qwen3-4b config
    qwen3_4b_config = Qwen34BInstruct2507Config()
    assert hasattr(qwen3_4b_config, 'enable_async_unimodal_processing'), "Qwen3-4b config missing enable_async_unimodal_processing"
    assert hasattr(qwen3_4b_config, 'async_max_concurrent_requests'), "Qwen3-4b config missing async_max_concurrent_requests"
    assert hasattr(qwen3_4b_config, 'async_buffer_size'), "Qwen3-4b config missing async_buffer_size"
    assert hasattr(qwen3_4b_config, 'async_batch_timeout'), "Qwen3-4b config missing async_batch_timeout"
    assert hasattr(qwen3_4b_config, 'enable_async_batching'), "Qwen3-4b config missing enable_async_batching"
    assert hasattr(qwen3_4b_config, 'async_processing_device'), "Qwen3-4b config missing async_processing_device"
    print("[PASS] Qwen3-4b config has all async processing parameters")

    # Test Qwen3-coder config
    qwen3_coder_config = Qwen3Coder30BConfig()
    assert hasattr(qwen3_coder_config, 'enable_async_unimodal_processing'), "Qwen3-coder config missing enable_async_unimodal_processing"
    assert hasattr(qwen3_coder_config, 'async_max_concurrent_requests'), "Qwen3-coder config missing async_max_concurrent_requests"
    assert hasattr(qwen3_coder_config, 'async_buffer_size'), "Qwen3-coder config missing async_buffer_size"
    assert hasattr(qwen3_coder_config, 'async_batch_timeout'), "Qwen3-coder config missing async_batch_timeout"
    assert hasattr(qwen3_coder_config, 'enable_async_batching'), "Qwen3-coder config missing enable_async_batching"
    assert hasattr(qwen3_coder_config, 'async_processing_device'), "Qwen3-coder config missing async_processing_device"
    print("[PASS] Qwen3-coder config has all async processing parameters")

    print("\nAll configurations have been successfully updated with async processing parameters!")


if __name__ == "__main__":
    test_config_async_params()