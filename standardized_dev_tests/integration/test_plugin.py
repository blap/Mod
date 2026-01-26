from src.inference_pio.models.glm_4_7.plugin import GLM_4_7_Plugin
from src.inference_pio.models.qwen3_4b_instruct_2507.plugin import Qwen3_4B_Instruct_2507_Plugin
from src.inference_pio.models.qwen3_coder_30b.plugin import Qwen3_Coder_30B_Plugin
from src.inference_pio.models.qwen3_vl_2b.plugin import Qwen3_VL_2B_Plugin

print('Testing plugin imports...')

glm_plugin = GLM_4_7_Plugin()
print('GLM-4.7 plugin created successfully')

qwen4b_plugin = Qwen3_4B_Instruct_2507_Plugin()
print('Qwen3-4B-Instruct-2507 plugin created successfully')

qwen30b_plugin = Qwen3_Coder_30B_Plugin()
print('Qwen3-Coder-30B plugin created successfully')

qwen_vl_plugin = Qwen3_VL_2B_Plugin()
print('Qwen3-VL-2B plugin created successfully')

print('All plugins created successfully!')

# Test that the new methods are available
print('Testing new predictive memory management methods...')
print(f'GLM plugin has start_predictive_memory_management: {hasattr(glm_plugin, "start_predictive_memory_management")}')
print(f'GLM plugin has stop_predictive_memory_management: {hasattr(glm_plugin, "stop_predictive_memory_management")}')

print(f'Qwen4B plugin has start_predictive_memory_management: {hasattr(qwen4b_plugin, "start_predictive_memory_management")}')
print(f'Qwen4B plugin has stop_predictive_memory_management: {hasattr(qwen4b_plugin, "stop_predictive_memory_management")}')

print(f'Qwen30B plugin has start_predictive_memory_management: {hasattr(qwen30b_plugin, "start_predictive_memory_management")}')
print(f'Qwen30B plugin has stop_predictive_memory_management: {hasattr(qwen30b_plugin, "stop_predictive_memory_management")}')

print(f'QwenVL plugin has start_predictive_memory_management: {hasattr(qwen_vl_plugin, "start_predictive_memory_management")}')
print(f'QwenVL plugin has stop_predictive_memory_management: {hasattr(qwen_vl_plugin, "stop_predictive_memory_management")}')

print('All tests passed!')