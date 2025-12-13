#!/usr/bin/env python3
"""
Script to clean up Python cache directories (__pycache__, .pytest_cache)
that may have been created before proper .gitignore configuration.
"""

import os
import shutil
from pathlib import Path


def clean_cache_directories(root_path="."):
    """
    Remove common Python cache directories from the project.
    
    Args:
        root_path (str): Root path of the project (default: current directory)
    """
    root = Path(root_path)
    
    # Directories to remove
    cache_dirs = ["__pycache__", ".pytest_cache"]
    
    removed_dirs = []
    failed_removals = []
    
    # Walk through the directory tree and remove cache directories
    for cache_dir in cache_dirs:
        for path in root.rglob(cache_dir):
            if path.is_dir():
                try:
                    shutil.rmtree(path)
                    removed_dirs.append(str(path))
                    print(f"Removed: {path}")
                except Exception as e:
                    failed_removals.append((str(path), str(e)))
                    print(f"Failed to remove {path}: {e}")
    
    print(f"\nSummary:")
    print(f"- Successfully removed {len(removed_dirs)} directories")
    if failed_removals:
        print(f"- Failed to remove {len(failed_removals)} directories:")
        for path, error in failed_removals:
            print(f"  {path}: {error}")
    else:
        print("- All cache directories removed successfully!")


if __name__ == "__main__":
    print("Cleaning up Python cache directories...")
    clean_cache_directories()
    print("\nCache cleanup completed!")