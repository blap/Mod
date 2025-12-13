"""Simple test for the configuration system."""

from src.qwen3_vl.core.config import Qwen3VLConfig
import tempfile
import os

# Test basic configuration
config = Qwen3VLConfig(num_hidden_layers=16, num_attention_heads=16)
print(f"Original config: {config.num_hidden_layers} layers, {config.num_attention_heads} heads")

# Save to file
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    temp_file = f.name

try:
    config.save_to_file(temp_file)
    
    # Load from file using from_pretrained
    loaded_config = Qwen3VLConfig.from_pretrained(temp_file)
    
    print(f"Loaded config: {loaded_config.num_hidden_layers} layers, {loaded_config.num_attention_heads} heads")
    
    # Verify values match
    assert loaded_config.num_hidden_layers == 16
    assert loaded_config.num_attention_heads == 16
    print("SUCCESS: Configuration loaded correctly with proper values")
finally:
    os.unlink(temp_file)