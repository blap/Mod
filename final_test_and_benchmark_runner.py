#!/usr/bin/env python
"""
Final Test and Benchmark Runner for Real Models

This script runs both real model tests and benchmarks with proper H drive detection
and model loading. It ensures that all functionality works with real models and datasets.
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def run_tests():
    """Run the updated real model tests."""
    print("=" * 80)
    print("RUNNING UPDATED REAL MODEL TESTS")
    print("=" * 80)
    
    try:
        # Run the updated real model tests
        result = subprocess.run([
            sys.executable, 
            str(Path(__file__).parent / "updated_real_model_tests.py")
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Test exit code: {result.returncode}")
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Tests timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def run_benchmarks():
    """Run the updated real model benchmarks."""
    print("\n" + "=" * 80)
    print("RUNNING UPDATED REAL MODEL BENCHMARKS")
    print("=" * 80)
    
    try:
        # Run the updated real model benchmarks
        result = subprocess.run([
            sys.executable, 
            str(Path(__file__).parent / "updated_real_model_benchmarks.py")
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print(f"Benchmark exit code: {result.returncode}")
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("Benchmarks timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        return False


def check_h_drive_access():
    """Check if H drive is accessible and has model files."""
    print("\n" + "=" * 80)
    print("CHECKING H DRIVE ACCESSIBILITY")
    print("=" * 80)
    
    try:
        import platform
        if platform.system() == 'Windows':
            h_drive_path = Path("H:/")
            if h_drive_path.exists():
                print("✓ H drive exists")
                
                # Check for model directories
                model_dirs = []
                for item in h_drive_path.iterdir():
                    if item.is_dir() and any(keyword in item.name.lower() for keyword in 
                                           ['qwen', 'glm', 'model', 'ai']):
                        model_dirs.append(item.name)
                
                if model_dirs:
                    print(f"✓ Found potential model directories: {model_dirs}")
                    return True
                else:
                    print("⚠ No obvious model directories found on H drive")
                    return True  # H drive exists but may not have models yet
            else:
                print("⚠ H drive not accessible")
                return False
        else:
            # On non-Windows systems, check common mount points
            common_mounts = ["/mnt/h", "/media/h", "/drives/h"]
            for mount in common_mounts:
                mount_path = Path(mount)
                if mount_path.exists():
                    print(f"✓ Found potential H drive mount at: {mount}")
                    
                    # Check for model directories
                    model_dirs = []
                    for item in mount_path.iterdir():
                        if item.is_dir() and any(keyword in item.name.lower() for keyword in 
                                               ['qwen', 'glm', 'model', 'ai']):
                            model_dirs.append(item.name)
                    
                    if model_dirs:
                        print(f"✓ Found potential model directories: {model_dirs}")
                        return True
                    else:
                        print("⚠ No obvious model directories found at mount point")
                        return True
            
            print("⚠ No H drive mount point found")
            return False
            
    except Exception as e:
        print(f"Error checking H drive: {e}")
        return False


def main():
    """Main function to run comprehensive tests and benchmarks."""
    print("Starting comprehensive test and benchmark execution...")
    
    start_time = time.time()
    
    # Check H drive accessibility first
    h_drive_accessible = check_h_drive_access()
    
    # Run tests
    tests_passed = run_tests()
    
    # Run benchmarks
    benchmarks_passed = run_benchmarks()
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"H Drive Accessible: {'YES' if h_drive_accessible else 'NO'}")
    print(f"Tests Passed: {'YES' if tests_passed else 'NO'}")
    print(f"Benchmarks Passed: {'YES' if benchmarks_passed else 'NO'}")
    print(f"Total Execution Time: {duration:.2f} seconds")
    
    if tests_passed and benchmarks_passed:
        print("\n✓ All tests and benchmarks completed successfully!")
        return 0
    else:
        print("\n⚠ Some tests or benchmarks failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)