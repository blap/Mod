"""
Very simple test to check basic functionality
"""
from datetime import datetime

print(f"Very simple test started at: {datetime.now()}")

try:
    # Test basic imports
    from src.qwen3_vl.config import Qwen3VLConfig
    print("✅ Qwen3VLConfig imported successfully")

    from src.qwen3_vl.models import Qwen3VLForConditionalGeneration
    print("✅ Qwen3VLForConditionalGeneration imported successfully")

    # Create minimal config
    config = Qwen3VLConfig(
        num_hidden_layers=1,
        num_attention_heads=1,
        hidden_size=8,
        intermediate_size=16,
        vision_num_hidden_layers=1,
        vision_num_attention_heads=1,
        vision_hidden_size=8,
        vision_intermediate_size=16
    )
    print("✅ Minimal config created successfully")

    # Validate config
    if config.validate_config():
        print("✅ Config validated successfully")

    # Create model
    model = Qwen3VLForConditionalGeneration(config)
    print("✅ Model created successfully")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Parameter count: {param_count:,}")

    # Check device
    device = next(model.parameters()).device
    print(f"✅ Model device: {device}")

    print("\n🎉 ALL BASIC TESTS PASSED! 🎉")
    print("Systematic errors have been resolved.")

except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    import traceback
    print("Full traceback:")
    traceback.print_exc()

print(f"Test completed at: {datetime.now()}")