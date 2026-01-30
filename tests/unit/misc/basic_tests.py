"""
Basic Tests for Inference-PIO Self-Contained Plugins

This module provides basic tests to verify that the self-contained plugins work correctly.
"""

import torch
from inference_pio.test_utils import (
    assert_is_not_none,
    assert_equal,
    assert_in,
    run_tests
)
from inference_pio.models.glm_4_7_flash import GLM_4_7_Flash_Plugin, create_glm_4_7_flash_plugin
from inference_pio.models.qwen3_coder_30b import Qwen3_Coder_30B_Plugin, create_qwen3_coder_30b_plugin
from inference_pio.models.qwen3_vl_2b import Qwen3_VL_2B_Instruct_Plugin, create_qwen3_vl_2b_instruct_plugin
from inference_pio.models.qwen3_4b_instruct_2507 import Qwen3_4B_Instruct_2507_Plugin, create_qwen3_4b_instruct_2507_plugin


def test_glm47_flash_plugin_creation():
    """Test that the GLM-4.7-Flash plugin is created successfully."""
    plugin = create_glm_4_7_flash_plugin()
    assert_is_not_none(plugin)
    assert_equal(plugin.metadata.name, "GLM-4.7-Flash")


def test_glm47_model_info():
    """Test that model info can be retrieved."""
    plugin = create_glm_4_7_flash_plugin()
    info = plugin.get_model_info()
    assert_in("name", info)
    assert_in("model_type", info)
    assert_equal(info["name"], "GLM-4.7-Flash")


def test_qwen3_coder_30b_plugin_creation():
    """Test that the Qwen3-Coder-30B plugin is created successfully."""
    plugin = create_qwen3_coder_30b_plugin()
    assert_is_not_none(plugin)
    assert_equal(plugin.metadata.name, "Qwen3-Coder-30B")


def test_qwen3_coder_30b_model_info():
    """Test that model info can be retrieved."""
    plugin = create_qwen3_coder_30b_plugin()
    info = plugin.get_model_info()
    assert_in("name", info)
    assert_in("model_type", info)
    assert_equal(info["name"], "Qwen3-Coder-30B")


def test_qwen3_vl_2b_plugin_creation():
    """Test that the Qwen3-VL-2B-Instruct plugin is created successfully."""
    plugin = create_qwen3_vl_2b_instruct_plugin()
    assert_is_not_none(plugin)
    assert_equal(plugin.metadata.name, "Qwen3-VL-2B-Instruct")


def test_qwen3_vl_2b_model_info():
    """Test that model info can be retrieved."""
    plugin = create_qwen3_vl_2b_instruct_plugin()
    info = plugin.get_model_info()
    assert_in("name", info)
    assert_in("model_type", info)
    assert_equal(info["name"], "Qwen3-VL-2B-Instruct")


def test_qwen3_4b_instruct_2507_plugin_creation():
    """Test that the Qwen3-4B-Instruct-2507 plugin is created successfully."""
    plugin = create_qwen3_4b_instruct_2507_plugin()
    assert_is_not_none(plugin)
    assert_equal(plugin.metadata.name, "Qwen3-4B-Instruct-2507")


def test_qwen3_4b_instruct_2507_model_info():
    """Test that model info can be retrieved."""
    plugin = create_qwen3_4b_instruct_2507_plugin()
    info = plugin.get_model_info()
    assert_in("name", info)
    assert_in("model_type", info)
    assert_equal(info["name"], "Qwen3-4B-Instruct-2507")


if __name__ == '__main__':
    # Run the tests using custom test utilities
    test_functions = [
        test_glm47_plugin_creation,
        test_glm47_model_info,
        test_qwen3_coder_30b_plugin_creation,
        test_qwen3_coder_30b_model_info,
        test_qwen3_vl_2b_plugin_creation,
        test_qwen3_vl_2b_model_info,
        test_qwen3_4b_instruct_2507_plugin_creation,
        test_qwen3_4b_instruct_2507_model_info
    ]
    run_tests(test_functions)