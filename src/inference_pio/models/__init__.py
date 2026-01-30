"""
Models Package for Inference-PIO System

This module provides the models package for the Inference-PIO system.
"""

from .glm_4_7_flash.plugin import GLM_4_7_Flash_Plugin, create_glm_4_7_flash_plugin
from .qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin, create_qwen3_coder_30b_plugin
from .qwen3_vl_2b.plugin import Qwen3_VL_2B_Instruct_Plugin, create_qwen3_vl_2b_instruct_plugin
from .qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin, create_qwen3_4b_instruct_2507_plugin
from .paddleocr_vl_1_5.plugin import PaddleOCRVL15Plugin, create_paddleocr_vl_1_5_plugin

__all__ = [
    # GLM-4.7
    "GLM_4_7_Flash_Plugin",
    "create_glm_4_7_flash_plugin",

    # Qwen3-Coder-30B
    "Qwen3_Coder_30B_Plugin",
    "create_qwen3_coder_30b_plugin",

    # Qwen3-VL-2B
    "Qwen3_VL_2B_Instruct_Plugin",
    "create_qwen3_vl_2b_instruct_plugin",

    # Qwen3-4B-Instruct-2507
    "Qwen3_4B_Instruct_2507_Plugin",
    "create_qwen3_4b_instruct_2507_plugin",

    # PaddleOCR-VL-1.5
    "PaddleOCRVL15Plugin",
    "create_paddleocr_vl_1_5_plugin",
]

# Lazy loading functions to avoid immediate import issues
def get_glm_4_7_flash_plugin():
    from .glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
    return create_glm_4_7_flash_plugin()

def get_qwen3_coder_30b_plugin():
    from .qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
    return create_qwen3_coder_30b_plugin()

def get_qwen3_vl_2b_plugin():
    from .qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
    return create_qwen3_vl_2b_instruct_plugin()

def get_qwen3_4b_instruct_2507_plugin():
    from .qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
    return create_qwen3_4b_instruct_2507_plugin()

def get_paddleocr_vl_1_5_plugin():
    from .paddleocr_vl_1_5.plugin import create_paddleocr_vl_1_5_plugin
    return create_paddleocr_vl_1_5_plugin()
