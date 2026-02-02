"""
Simple Migration Script for Qwen3-VL-2B Specific Code

This script identifies and moves Qwen3-VL-2B specific code from the common directory
to the model-specific plugin directory.
"""

import os
import shutil
from pathlib import Path


def move_qwen3_vl_specific_files():
    """
    Move Qwen3-VL-2B specific files from common to model-specific directory.
    """
    print("Starting migration of Qwen3-VL-2B specific code...")

    # Define source and destination directories
    common_dir = Path("C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/common")
    model_dir = Path(
        "C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/models/qwen3_vl_2b"
    )

    # Identify files with Qwen3-VL-2B specific code in the common directory
    files_to_move = []

    for file_path in common_dir.glob("*.py"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Look for Qwen3-VL-2B specific identifiers
            if any(
                phrase in content.lower()
                for phrase in [
                    "qwen3vl2b",
                    "qwen3_vl_2b",
                    "qwen3-vl-2b",
                    "qwen3 vl 2b",
                    "qwen3vl",
                    "qwen3_vl",
                ]
            ):
                files_to_move.append(file_path.name)

    print(f"Found {len(files_to_move)} files with Qwen3-VL-2B specific code:")
    for file in files_to_move:
        print(f"  - {file}")

    # Move each file to the model-specific directory
    for file_name in files_to_move:
        src_path = common_dir / file_name
        dst_path = model_dir / file_name

        if src_path.exists():
            print(f"\nMoving {file_name} from common to model-specific directory...")

            # If destination already exists, we'll skip to avoid overwriting
            if dst_path.exists():
                print(
                    f"  Warning: {dst_path} already exists, skipping migration for this file"
                )
                continue

            try:
                shutil.move(str(src_path), str(dst_path))
                print(f"  Successfully moved {file_name}")
            except Exception as e:
                print(f"  Error moving {file_name}: {e}")

    print("\nMigration completed!")


if __name__ == "__main__":
    move_qwen3_vl_specific_files()
    print("\nQwen3-VL-2B specific code has been moved to the model plugin directory!")
