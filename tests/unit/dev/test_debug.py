"""
Debug test to identify the exact location of the problem
"""
import sys
import os
import traceback
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.getcwd())

print(f'Debug test started at: {datetime.now()}')

try:
    print('Step 1: Testing basic import...')
    from src.qwen3_vl.config import Qwen3VLConfig
    print('✅ Step 1 passed: Config import successful')

    print('Step 2: Testing models import...')
    from src.qwen3_vl.models import Qwen3VLForConditionalGeneration
    print('✅ Step 2 passed: Models import successful')

    print('Step 3: Creating minimal config...')
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
    print('✅ Step 3 passed: Config creation successful')

    print('Step 4: Validating config...')
    if config.validate_config():
        print('✅ Step 4 passed: Config validation successful')

    print('Step 5: Creating model...')
    model = Qwen3VLForConditionalGeneration(config)
    print('✅ Step 5 passed: Model creation successful')

    print('Step 6: Checking parameters...')
    param_count = sum(p.numel() for p in model.parameters())
    print(f'✅ Step 6 passed: Parameter count: {param_count:,}')

    print('Step 7: Checking device...')
    device = next(model.parameters()).device
    print(f'✅ Step 7 passed: Model device: {device}')

    print('\n🎉 ALL STEPS COMPLETED SUCCESSFULLY! 🎉')

except Exception as e:
    print(f'❌ ERROR at some step: {str(e)}')
    print('Full traceback:')
    print(traceback.format_exc())
    print(f'Error type: {type(e).__name__}')