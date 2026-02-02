import re
from pathlib import Path


def fix_imports_security_manual(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = content

    # Fix from src.inference_pio.security.context import SecurityContext
    new_content = re.sub(
        r"from src\.inference_pio\.security\.context import SecurityContext",
        r"from src.inference_pio.common.security_manager import SecurityContext",
        new_content,
    )

    # Fix from src.inference_pio.security.manager import SecurityManager
    new_content = re.sub(
        r"from src\.inference_pio\.security\.manager import SecurityManager",
        r"from src.inference_pio.common.security_manager import SecurityManager",
        new_content,
    )

    # Fix from src.inference_pio.security.resource_manager import ResourceManager
    new_content = re.sub(
        r"from src\.inference_pio\.security\.resource_manager import ResourceManager",
        r"from src.inference_pio.common.memory_manager import MemoryManager as ResourceManager",
        new_content,
    )
    # Assuming ResourceManager was renamed or merged. Or maybe check if it exists in security_manager.
    # Actually, looking at file list, there is 'memory_manager.py' and 'security_manager.py'.
    # There is no 'resource_manager.py' in common.
    # Maybe it was in 'plugin_isolation.py' or 'memory_manager.py'.
    # Let's map to memory_manager for now as a guess, or security_manager.

    if new_content != content:
        print(f"Fixed security imports in {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)


def main():
    test_dir = Path("tests/unit/common/security")
    for p in test_dir.glob("**/*.py"):
        fix_imports_security_manual(p)


if __name__ == "__main__":
    main()
