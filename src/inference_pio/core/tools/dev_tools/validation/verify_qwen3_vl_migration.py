#!/usr/bin/env python
"""
Verification Script for Qwen3-VL-2B Code Migration

This script verifies that all Qwen3-VL-2B specific code has been properly moved from the common directory
to the model-specific plugin directory.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_common_directory_for_qwen3_vl_code():
    """Check the common directory for any remaining Qwen3-VL-2B specific code."""
    print("Checking common directory for Qwen3-VL-2B specific code...")

    common_dir = Path("C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/common")
    qwen3_vl_keywords = [
        "qwen3.*vl.*2b",
        "qwen3_vl_2b",
        "Qwen3VL2B",
        "qwen3vl2b",
        "qwen3.*vl",
        "qwen3_vl",
        "Qwen3VL",
    ]

    found_files = []

    for py_file in common_dir.rglob("*.py"):
        try:
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().lower()

                for keyword in qwen3_vl_keywords:
                    if (
                        keyword.replace(".*", "") in content
                    ):  # Simple check without regex
                        found_files.append((py_file, keyword))
                        break
        except:
            continue  # Skip files that can't be read

    if found_files:
        print("WARNING: Found Qwen3-VL-2B specific code in common directory:")
        for file_path, keyword in found_files:
            print(f"  - {file_path} (contains: {keyword})")
        return False
    else:
        print("[SUCCESS] No Qwen3-VL-2B specific code found in common directory")
        return True


def check_model_directory_for_qwen3_vl_code():
    """Check the Qwen3-VL-2B model directory for the migrated code."""
    print("\nChecking Qwen3-VL-2B model directory for migrated code...")

    model_dir = Path(
        "C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/models/qwen3_vl_2b"
    )

    expected_files = ["model.py", "config.py", "plugin.py", "__init__.py"]

    missing_files = []
    for file_path in expected_files:
        full_path = model_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print("ERROR: Missing expected files in model directory:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print(
            "[SUCCESS] All expected Qwen3-VL-2B specific files found in model directory"
        )
        return True


def verify_imports_work():
    """Verify that imports work correctly from the model directory."""
    print("\nVerifying imports from model directory...")

    try:
        # Test importing the main model components
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "qwen3_vl_2b",
            "C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/models/qwen3_vl_2b/__init__.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        print("[SUCCESS] Qwen3-VL-2B module imported successfully")
        return True

    except ImportError as e:
        print(f"ERROR: Import failed: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error during import verification: {e}")
        return False


def main():
    """Main verification function."""
    print("Verifying Qwen3-VL-2B code migration...\n")

    all_checks_passed = True

    # Check 1: Verify no Qwen3-VL-2B specific code remains in common directory
    if not check_common_directory_for_qwen3_vl_code():
        all_checks_passed = False

    # Check 2: Verify all expected files exist in model directory
    if not check_model_directory_for_qwen3_vl_code():
        all_checks_passed = False

    # Check 3: Verify imports work correctly
    if not verify_imports_work():
        all_checks_passed = False

    print("\n" + "=" * 50)
    if all_checks_passed:
        print("[SUCCESS] ALL VERIFICATION CHECKS PASSED!")
        print(
            "Qwen3-VL-2B specific code has been successfully migrated to the model plugin directory."
        )
        print("Generic code remains in the common directory.")
    else:
        print("[FAILURE] SOME VERIFICATION CHECKS FAILED!")
        print("Please review the issues listed above.")
    print("=" * 50)

    return all_checks_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
