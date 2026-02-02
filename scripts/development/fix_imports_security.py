import re
from pathlib import Path


def fix_imports_security(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content

    # Fix from src.inference_pio.security import ...
    new_content = re.sub(
        r"from src\.inference_pio\.security import",
        r"from src.inference_pio.common.security_manager import",
        new_content,
    )

    # Fix from src.inference_pio.security.resource_management import ...
    new_content = re.sub(
        r"from src\.inference_pio\.security\.resource_management import",
        r"from src.inference_pio.common.security_manager import",
        new_content,
    )

    # Fix from src.inference_pio.security.security_contexts import ...
    new_content = re.sub(
        r"from src\.inference_pio\.security\.security_contexts import",
        r"from src.inference_pio.common.security_manager import",
        new_content,
    )

    if new_content != content:
        print(f"Fixed security imports in {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)


def main():
    test_dir = Path("tests/unit/common/security")
    for p in test_dir.glob("**/*.py"):
        fix_imports_security(p)


if __name__ == "__main__":
    main()
