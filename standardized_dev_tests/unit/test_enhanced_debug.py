"""
Enhanced debug test to identify the exact location of the problem with flush
"""
import sys
import os
import traceback
from datetime import datetime

# Force output to be unbuffered
sys.stdout.flush()
sys.stderr.flush()

# Add current directory to path
sys.path.insert(0, os.getcwd())

print(f'Enhanced debug test started at: {datetime.now()}', flush=True)

try:
    print('Step 1: Testing basic import...', flush=True)
    from src.qwen3_vl.config import Qwen3VLConfig
    print('✅ Step 1 passed: Config import successful', flush=True)

    print('Step 2: Testing models import...', flush=True)
    from src.qwen3_vl.models import Qwen3VLForConditionalGeneration
    print('✅ Step 2 passed: Models import successful', flush=True)

    print('Step 3: Creating minimal config...', flush=True)
    config = Qwen3VLConfig(
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_size=64,
        intermediate_size=128,
        vision_num_hidden_layers=2,
        vision_num_attention_heads=2,
        vision_hidden_size=32,
        vision_intermediate_size=64,
        use_mixed_precision=False
    )
    print('✅ Step 3 passed: Config creation successful', flush=True)

    print('Step 4: Validating config...', flush=True)
    if config.validate_config():
        print('✅ Step 4 passed: Config validation successful', flush=True)

    print('Step 5: Creating model...', flush=True)
    model = Qwen3VLForConditionalGeneration(config)
    print('✅ Step 5 passed: Model creation successful', flush=True)

    print('Step 6: Checking parameters...', flush=True)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'✅ Step 6 passed: Parameter count: {param_count:,}', flush=True)

    print('Step 7: Checking device...', flush=True)
    device = next(model.parameters()).device
    print(f'✅ Step 7 passed: Model device: {device}', flush=True)

    print('\n🎉 ALL STEPS COMPLETED SUCCESSFULLY! 🎉', flush=True)

except Exception as e:
    print(f'❌ ERROR at some step: {str(e)}', flush=True)
    print('Full traceback:', flush=True)
    print(traceback.format_exc(), flush=True)
    print(f'Error type: {type(e).__name__}', flush=True)