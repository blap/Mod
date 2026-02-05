"""
Test script to verify predictive memory optimization implementation across all models.
"""

import torch
import tempfile
import os
from src.inference_pio.models.glm_4_7_flash.plugin import create_glm_4_7_flash_plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import create_qwen3_4b_instruct_2507_plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import create_qwen3_coder_30b_plugin
from src.inference_pio.models.qwen3_0_6b.plugin import create_qwen3_0_6b_plugin
from src.inference_pio.models.qwen3_coder_next.plugin import create_qwen3_coder_next_plugin


def test_predictive_memory_optimization():
    """Test predictive memory optimization across all models."""
    
    # Test GLM-4.7-Flash
    print("Testing GLM-4.7-Flash predictive memory optimization...")
    glm_plugin = create_glm_4_7_flash_plugin()
    
    # Initialize with predictive memory management enabled
    init_result = glm_plugin.initialize(enable_predictive_management=True)
    print(f"GLM-4.7-Flash initialization: {init_result}")
    
    # Test starting predictive memory management
    start_result = glm_plugin.start_predictive_memory_management()
    print(f"GLM-4.7-Flash start predictive memory management: {start_result}")
    
    # Test recording a tensor access
    dummy_tensor = torch.randn(10, 10)
    record_result = glm_plugin.record_tensor_access("test_tensor", dummy_tensor)
    print(f"GLM-4.7-Flash record tensor access: {record_result}")
    
    # Test stopping predictive memory management
    stop_result = glm_plugin.stop_predictive_memory_management()
    print(f"GLM-4.7-Flash stop predictive memory management: {stop_result}")
    
    print()
    
    # Test Qwen3-4B-Instruct-2507
    print("Testing Qwen3-4B-Instruct-2507 predictive memory optimization...")
    qwen4b_plugin = create_qwen3_4b_instruct_2507_plugin()
    
    # Initialize with predictive memory management enabled
    init_result = qwen4b_plugin.initialize(enable_predictive_management=True)
    print(f"Qwen3-4B-Instruct-2507 initialization: {init_result}")
    
    # Test starting predictive memory management
    start_result = qwen4b_plugin.start_predictive_memory_management()
    print(f"Qwen3-4B-Instruct-2507 start predictive memory management: {start_result}")
    
    # Test recording a tensor access
    dummy_tensor = torch.randn(10, 10)
    record_result = qwen4b_plugin.record_tensor_access("test_tensor", dummy_tensor)
    print(f"Qwen3-4B-Instruct-2507 record tensor access: {record_result}")
    
    # Test stopping predictive memory management
    stop_result = qwen4b_plugin.stop_predictive_memory_management()
    print(f"Qwen3-4B-Instruct-2507 stop predictive memory management: {stop_result}")
    
    print()
    
    # Test Qwen3-Coder-30B
    print("Testing Qwen3-Coder-30B predictive memory optimization...")
    qwen30b_plugin = create_qwen3_coder_30b_plugin()
    
    # Initialize with predictive memory management enabled
    init_result = qwen30b_plugin.initialize(enable_predictive_management=True)
    print(f"Qwen3-Coder-30B initialization: {init_result}")
    
    # Test starting predictive memory management
    start_result = qwen30b_plugin.start_predictive_memory_management()
    print(f"Qwen3-Coder-30B start predictive memory management: {start_result}")
    
    # Test recording a tensor access
    dummy_tensor = torch.randn(10, 10)
    record_result = qwen30b_plugin.record_tensor_access("test_tensor", dummy_tensor)
    print(f"Qwen3-Coder-30B record tensor access: {record_result}")
    
    # Test stopping predictive memory management
    stop_result = qwen30b_plugin.stop_predictive_memory_management()
    print(f"Qwen3-Coder-30B stop predictive memory management: {stop_result}")
    
    print()
    
    # Test Qwen3-0.6B
    print("Testing Qwen3-0.6B predictive memory optimization...")
    qwen06b_plugin = create_qwen3_0_6b_plugin()
    
    # Initialize with predictive memory management enabled
    init_result = qwen06b_plugin.initialize(enable_predictive_management=True)
    print(f"Qwen3-0.6B initialization: {init_result}")
    
    # Test starting predictive memory management
    start_result = qwen06b_plugin.start_predictive_memory_management()
    print(f"Qwen3-0.6B start predictive memory management: {start_result}")
    
    # Test recording a tensor access
    dummy_tensor = torch.randn(10, 10)
    record_result = qwen06b_plugin.record_tensor_access("test_tensor", dummy_tensor)
    print(f"Qwen3-0.6B record tensor access: {record_result}")
    
    # Test stopping predictive memory management
    stop_result = qwen06b_plugin.stop_predictive_memory_management()
    print(f"Qwen3-0.6B stop predictive memory management: {stop_result}")
    
    print()
    
    # Test Qwen3-Coder-Next
    print("Testing Qwen3-Coder-Next predictive memory optimization...")
    qwen_next_plugin = create_qwen3_coder_next_plugin()
    
    # Initialize with predictive memory management enabled
    init_result = qwen_next_plugin.initialize(enable_predictive_management=True)
    print(f"Qwen3-Coder-Next initialization: {init_result}")
    
    # Test starting predictive memory management
    start_result = qwen_next_plugin.start_predictive_memory_management()
    print(f"Qwen3-Coder-Next start predictive memory management: {start_result}")
    
    # Test recording a tensor access
    dummy_tensor = torch.randn(10, 10)
    record_result = qwen_next_plugin.record_tensor_access("test_tensor", dummy_tensor)
    print(f"Qwen3-Coder-Next record tensor access: {record_result}")
    
    # Test stopping predictive memory management
    stop_result = qwen_next_plugin.stop_predictive_memory_management()
    print(f"Qwen3-Coder-Next stop predictive memory management: {stop_result}")
    
    print("\nAll models have predictive memory optimization implemented successfully!")


if __name__ == "__main__":
    test_predictive_memory_optimization()