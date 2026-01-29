
import re
from pathlib import Path

def fix_imports_specific_optimization(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content

    # Fix from src.inference_pio.optimization.profiles import ...
    new_content = re.sub(r'from src\.inference_pio\.optimization\.profiles import',
                         r'from src.inference_pio.common.optimization_profiles import', new_content)

    new_content = re.sub(r'from src\.inference_pio\.optimization import',
                         r'from src.inference_pio.common.optimization_profiles import', new_content)

    # Note: 'src.inference_pio.common.optimization_profiles' might not contain the specific classes like QuantizationOptimization
    # They might be in 'src.inference_pio.common.quantization' or similar.
    # But let's try mapping to common first as a fallback.

    if new_content != content:
        print(f"Fixed optimization imports in {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

def main():
    test_dir = Path('tests/unit/common/optimization/specific')
    for p in test_dir.glob('**/*.py'):
        fix_imports_specific_optimization(p)

if __name__ == '__main__':
    main()
