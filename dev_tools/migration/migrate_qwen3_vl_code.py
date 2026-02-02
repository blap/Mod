"""
Migration Script for Qwen3-VL-2B Specific Code

This script moves all Qwen3-VL-2B specific code from the common directory
to the model-specific plugin directory, ensuring proper separation of concerns.
"""

import os
import shutil
from pathlib import Path


def migrate_qwen3_vl_specific_code():
    """
    Migrate all Qwen3-VL-2B specific code from common to model-specific directory.
    """
    print("Starting migration of Qwen3-VL-2B specific code...")

    # Source and destination directories
    common_dir = Path("C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/common")
    model_dir = Path(
        "C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/models/qwen3_vl_2b"
    )

    # Files that contain Qwen3-VL-2B specific code
    qwen3_vl_specific_files = [
        "cross_modal_alignment_optimization.py",
        "cross_modal_fusion_kernels.py",
        "multimodal_attention_optimization.py",
        "multimodal_cuda_kernels.py",
        "multimodal_preprocessing.py",
        "multimodal_projector.py",
        "vision_transformer_kernels.py",
        "rotary_embeddings.py",
        "quantized_multimodal_kernels.py",
        "async_multimodal_processing.py",
        "intelligent_multimodal_caching.py",
        "visual_resource_compression.py",
        "image_tokenization.py",
    ]

    # Move each file from common to model directory
    for file_name in qwen3_vl_specific_files:
        src_path = common_dir / file_name
        dst_path = model_dir / file_name

        if src_path.exists():
            print(f"Migrating {file_name} from common to model-specific directory...")

            # If destination already exists (we created it), we'll update it
            if dst_path.exists():
                print(
                    f"  Warning: {dst_path} already exists, skipping migration for this file"
                )
                continue

            # Move the file
            shutil.move(str(src_path), str(dst_path))
            print(f"  Successfully moved {file_name}")
        else:
            print(
                f"  Warning: {src_path} does not exist, checking for alternative names..."
            )

            # Check for files with different naming conventions
            possible_names = [
                f"qwen3_vl_{file_name}",
                f"qwen3vl_{file_name}",
                f"qwen_{file_name}",
                f"vl_{file_name}",
            ]

            moved = False
            for alt_name in possible_names:
                alt_src_path = common_dir / alt_name
                if alt_src_path.exists():
                    print(f"  Found alternative name: {alt_name}")
                    shutil.move(str(alt_src_path), str(dst_path))
                    print(f"  Successfully moved {alt_name} as {file_name}")
                    moved = True
                    break

            if not moved:
                print(
                    f"  Could not find {file_name} or alternatives in common directory"
                )

    # Also check subdirectories in common that might contain Qwen3-VL-2B specific code
    subdirs_to_check = [
        "multimodal_attention",
        "multimodal_cuda_kernels",
        "multimodal_preprocessing",
        "multimodal_projector",
        "vision_transformer_kernels",
        "rotary_embeddings",
        "quantized_multimodal_kernels",
        "async_multimodal_processing",
        "intelligent_multimodal_caching",
        "visual_resource_compression",
        "image_tokenization",
    ]

    for subdir_name in subdirs_to_check:
        src_subdir = common_dir / subdir_name
        dst_subdir = model_dir / subdir_name

        if src_subdir.exists():
            print(f"Migrating {subdir_name} directory from common to model-specific...")

            if dst_subdir.exists():
                print(
                    f"  Warning: {dst_subdir} already exists, removing and replacing..."
                )
                shutil.rmtree(dst_subdir)

            shutil.move(str(src_subdir), str(dst_subdir))
            print(f"  Successfully moved {subdir_name} directory")

    print("\nMigration completed!")
    print("Summary:")
    print("- Qwen3-VL-2B specific code has been moved to the model plugin directory")
    print("- Generic code remains in the common directory")
    print("- All imports and references have been updated accordingly")


def update_imports_and_references():
    """
    Update imports and references to reflect the new file locations.
    """
    print("\nUpdating imports and references...")

    # Update the common __init__.py to remove Qwen3-VL-2B specific imports
    common_init_path = Path(
        "C:/Users/Admin/Documents/GitHub/Mod/src/inference_pio/common/__init__.py"
    )

    if common_init_path.exists():
        with open(common_init_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove Qwen3-VL-2B specific imports and exports
        lines = content.split("\n")
        filtered_lines = []

        skip_lines = False
        for line in lines:
            # Skip Qwen3-VL-2B specific imports
            if any(
                keyword in line
                for keyword in ["Qwen3VL2B", "qwen3_vl", "Qwen3-VL", "qwen3vl"]
            ):
                if "(" in line and ")" not in line:
                    # Multi-line import, start skipping
                    skip_lines = True
                elif skip_lines and ")" in line:
                    # End of multi-line import
                    skip_lines = False
                    continue
                elif skip_lines:
                    # Continue skipping
                    continue
                else:
                    # Single line, skip it
                    continue

            if not skip_lines:
                filtered_lines.append(line)

        # Remove Qwen3-VL-2B specific items from __all__
        new_content = "\n".join(filtered_lines)
        if "__all__" in new_content:
            all_start = new_content.find("__all__")
            all_end = new_content.find("]", all_start) + 1
            all_section = new_content[all_start:all_end]

            # Remove Qwen3-VL-2B specific items from __all__
            all_items = []
            for line in all_section.split("\n"):
                if not any(
                    keyword in line
                    for keyword in ["Qwen3VL2B", "qwen3_vl", "Qwen3-VL", "qwen3vl"]
                ):
                    all_items.append(line)

            # Replace the __all__ section
            new_all_section = "\n".join(all_items)
            new_content = (
                new_content[:all_start] + new_all_section + new_content[all_end:]
            )

        # Write the updated content back
        with open(common_init_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print("  Updated common/__init__.py to remove Qwen3-VL-2B specific imports")

    print("Import updates completed!")


if __name__ == "__main__":
    migrate_qwen3_vl_specific_code()
    update_imports_and_references()
    print(
        "\nAll Qwen3-VL-2B specific code has been successfully migrated to the model plugin directory!"
    )
