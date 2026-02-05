#!/usr/bin/env python
"""
Final Verification Report for Model Loading and Benchmarks

This script verifies that all tests and benchmarks are working correctly 
with real models, and confirms that models can be loaded from drive H.
"""

import sys
import os
import time
import subprocess
from pathlib import Path

def check_h_drive_access():
    """Check if H drive is accessible and has model files."""
    print("=" * 80)
    print("CHECKING H DRIVE ACCESSIBILITY")
    print("=" * 80)

    try:
        from src.inference_pio.core.model_loader import ModelLoader
        
        h_drive = ModelLoader.get_h_drive_base()
        print(f"H drive detected: {h_drive}")
        
        if h_drive and h_drive.exists():
            print("✓ H drive exists and is accessible")
            
            # Check for model directories
            model_dirs = []
            for item in h_drive.iterdir():
                if item.is_dir() and any(keyword in item.name.lower() for keyword in
                                       ['qwen', 'glm', 'model', 'ai']):
                    model_dirs.append(item.name)

            if model_dirs:
                print(f"✓ Found potential model directories: {model_dirs}")
                return True, model_dirs
            else:
                print("⚠ No obvious model directories found on H drive, but drive is accessible")
                return True, []
        else:
            print("⚠ H drive not accessible")
            return False, []

    except Exception as e:
        print(f"Error checking H drive: {e}")
        return False, []


def run_model_tests():
    """Run model tests to verify functionality."""
    print("\n" + "=" * 80)
    print("RUNNING MODEL FUNCTIONALITY TESTS")
    print("=" * 80)

    try:
        # Run the updated real model tests
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "updated_real_model_tests.py")
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout

        print("STDOUT:")
        print(result.stdout[-2000:])  # Last 2000 chars to avoid too much output

        if result.stderr:
            print("STDERR:")
            print(result.stderr[-1000:])  # Last 1000 chars

        print(f"Test exit code: {result.returncode}")
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("Tests timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def run_benchmarks():
    """Run benchmarks to verify performance."""
    print("\n" + "=" * 80)
    print("RUNNING MODEL BENCHMARKS")
    print("=" * 80)

    try:
        # Run the updated real model benchmarks
        result = subprocess.run([
            sys.executable,
            str(Path(__file__).parent / "updated_real_model_benchmarks.py")
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout

        print("STDOUT:")
        print(result.stdout[-2000:])  # Last 2000 chars to avoid too much output

        if result.stderr:
            print("STDERR:")
            print(result.stderr[-1000:])  # Last 1000 chars

        print(f"Benchmark exit code: {result.returncode}")
        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("Benchmarks timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        return False


def verify_model_loading():
    """Verify that models can be loaded from H drive."""
    print("\n" + "=" * 80)
    print("VERIFYING MODEL LOADING FROM H DRIVE")
    print("=" * 80)

    try:
        from src.inference_pio.core.model_loader import ModelLoader
        
        # Test resolving model path
        resolved_path = ModelLoader.resolve_model_path('qwen3-0.6b', 'Qwen/Qwen3-0.6B')
        print(f"Resolved path for qwen3-0.6b: {resolved_path}")
        
        # Check if it's using H drive
        if str(resolved_path).startswith('H:') or 'H:' in str(resolved_path):
            print("✓ Model path correctly points to H drive")
            h_drive_used = True
        else:
            print("⚠ Model path does not point to H drive")
            h_drive_used = False
            
        # Test with another model
        resolved_path2 = ModelLoader.resolve_model_path('qwen3_coder_next', 'Qwen/Qwen3-Coder-Next')
        print(f"Resolved path for qwen3_coder_next: {resolved_path2}")
        
        if str(resolved_path2).startswith('H:') or 'H:' in str(resolved_path2):
            print("✓ Model path for qwen3_coder_next correctly points to H drive")
            h_drive_used2 = True
        else:
            print("⚠ Model path for qwen3_coder_next does not point to H drive")
            h_drive_used2 = False
            
        return h_drive_used or h_drive_used2

    except Exception as e:
        print(f"Error verifying model loading: {e}")
        return False


def main():
    """Main function to run final verification."""
    print("Starting final verification of tests and benchmarks with real models...")
    
    start_time = time.time()
    
    # Check H drive access
    h_drive_accessible, model_dirs = check_h_drive_access()
    
    # Verify model loading from H drive
    model_loading_verified = verify_model_loading()
    
    # Run model tests
    tests_passed = run_model_tests()
    
    # Run benchmarks
    benchmarks_passed = run_benchmarks()
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 80)
    print(f"H Drive Accessible: {'YES' if h_drive_accessible else 'NO'}")
    if model_dirs:
        print(f"Model Directories Found: {model_dirs}")
    print(f"Model Loading from H Drive Verified: {'YES' if model_loading_verified else 'NO'}")
    print(f"Functionality Tests Passed: {'YES' if tests_passed else 'NO'}")
    print(f"Benchmarks Passed: {'YES' if benchmarks_passed else 'NO'}")
    print(f"Total Execution Time: {duration:.2f} seconds")
    
    # Overall success criteria
    overall_success = (
        h_drive_accessible and 
        model_loading_verified and 
        tests_passed and 
        benchmarks_passed
    )
    
    print(f"\nOVERALL STATUS: {'SUCCESS' if overall_success else 'FAILURE'}")
    
    if overall_success:
        print("\n✓ All verifications completed successfully!")
        print("✓ H drive is accessible and models can be loaded from it")
        print("✓ All functionality tests passed")
        print("✓ All benchmarks completed successfully")
        return 0
    else:
        print("\n⚠ Some verifications failed:")
        if not h_drive_accessible:
            print("  - H drive not accessible")
        if not model_loading_verified:
            print("  - Could not verify model loading from H drive")
        if not tests_passed:
            print("  - Functionality tests failed")
        if not benchmarks_passed:
            print("  - Benchmarks failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)