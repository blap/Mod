"""
Final integration test for Qwen3-VL with all memory optimizations
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.modeling_qwen3_vl_integrated import Qwen3VLForConditionalGeneration
from src.qwen3_vl.optimization.integrated_memory_manager import create_optimized_memory_manager
from src.components.configuration.config import Qwen3VLConfig

print('Testing Qwen3-VL with integrated memory optimizations...')

# Test basic functionality
config = Qwen3VLConfig()
config.hidden_size = 256
config.num_attention_heads = 4
config.num_hidden_layers = 2
config.vocab_size = 1000
config.max_position_embeddings = 256
config.vision_hidden_size = 256
config.vision_num_attention_heads = 4
config.vision_num_hidden_layers = 2
config.vision_image_size = 224
config.vision_patch_size = 14
config.vision_num_channels = 3

# Create memory manager
hardware_config = {
    'cpu_model': 'Intel i5-10210U',
    'gpu_model': 'NVIDIA SM61',
    'memory_size': 8 * 1024 * 1024 * 1024,  # 8GB
    'storage_type': 'nvme'
}
memory_manager = create_optimized_memory_manager(hardware_config)

# Create model
model = Qwen3VLForConditionalGeneration(config, memory_manager=memory_manager)
print('Model created successfully with integrated memory optimizations')

# Test forward pass
import torch
input_ids = torch.randint(0, config.vocab_size, (1, 10))
pixel_values = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output = model(input_ids=input_ids, pixel_values=pixel_values)
    print(f'Forward pass successful, output shape: {output.shape}')

print('All tests passed successfully!')