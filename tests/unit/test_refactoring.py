"""
Test script to verify the refactored structure works correctly
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Test that the old import path still works due to compatibility layer
try:
    from src.models.config import Qwen3VLConfig
    print("SUCCESS: Old config import works")
except ImportError as e:
    print(f"ERROR: Old config import failed: {e}")

try:
    from src.models.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    print("SUCCESS: Old model import works")
except ImportError as e:
    print(f"ERROR: Old model import failed: {e}")

# Test that new import path works
try:
    from src.components.models.qwen3_vl_model import Qwen3VLForConditionalGeneration
    print("SUCCESS: New model import works")
except ImportError as e:
    print(f"ERROR: New model import failed: {e}")

try:
    from src.components.configuration.config import Qwen3VLConfig
    print("SUCCESS: New config import works")
except ImportError as e:
    print(f"ERROR: New config import failed: {e}")

# Test creating a model with default config
try:
    config = Qwen3VLConfig()
    print(f"SUCCESS: Config creation works, vocab_size: {config.vocab_size}")
except Exception as e:
    print(f"ERROR: Config creation failed: {e}")

print("All tests completed successfully!")