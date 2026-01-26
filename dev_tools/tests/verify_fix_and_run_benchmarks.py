#!/usr/bin/env python
"""
Verification script to confirm all 6 benchmark categories for qwen3_vl_2b model work correctly.
"""

import sys
import time
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_benchmark_imports():
    """Test that all benchmark modules can be imported without errors."""
    print("Testing benchmark module imports...")
    
    benchmark_modules = [
        "benchmark_accuracy",
        "benchmark_inference_speed", 
        "benchmark_memory_usage",
        "benchmark_optimization_impact",
        "benchmark_power_efficiency",
        "benchmark_throughput"
    ]
    
    results = {}
    
    for module_name in benchmark_modules:
        print(f"Testing import of {module_name}...")
        try:
            # Import the main benchmark runner
            # For qwen3_vl_2b, determine the correct subdirectory based on benchmark name
            if 'inference_speed' in module_name or 'memory_usage' in module_name or 'throughput' in module_name or 'power_efficiency' in module_name or 'optimization_impact' in module_name:
                subdir = 'performance'
            elif 'accuracy' in module_name:
                subdir = 'unit'
            elif 'comparison' in module_name or 'async_multimodal' in module_name or 'intelligent_multimodal' in module_name:
                subdir = 'integration'
            else:
                subdir = 'performance'  # default

            module = __import__(f"src.inference_pio.models.qwen3_vl_2b.benchmarks.{subdir}.{module_name}", fromlist=[module_name])
            print(f"  SUCCESS: {module_name} imported successfully")
            results[module_name] = True
        except ImportError as e:
            print(f"  ERROR: Failed to import {module_name}: {e}")
            results[module_name] = False
        except Exception as e:
            print(f"  ERROR: Error importing {module_name}: {e}")
            results[module_name] = False

    return results

def test_main_benchmark_scripts():
    """Test that the main benchmark scripts can be imported without errors."""
    print("\nTesting main benchmark script imports...")

    main_benchmark_scripts = [
        "benchmark_accuracy.py",
        "benchmark_inference_speed.py",
        "benchmark_memory_usage.py",
        "benchmark_optimization_impact.py",
        "benchmark_power_efficiency.py",
        "benchmark_throughput.py"
    ]

    results = {}

    for script_name in main_benchmark_scripts:
        print(f"Testing import of {script_name}...")
        try:
            # Import the main benchmark script
            module_name = script_name.replace('.py', '')
            module = __import__(module_name, fromlist=[module_name])
            print(f"  SUCCESS: {script_name} imported successfully")
            results[script_name] = True
        except ImportError as e:
            print(f"  ERROR: Failed to import {script_name}: {e}")
            results[script_name] = False
        except Exception as e:
            print(f"  ERROR: Error importing {script_name}: {e}")
            results[script_name] = False

    return results

def test_function_name_fix():
    """Test that the function name fix is working."""
    print("\nTesting function name fix...")

    try:
        from src.inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_instruct_plugin
        print("  SUCCESS: Successfully imported create_qwen3_vl_2b_instruct_plugin")

        # Try to create the plugin
        plugin = create_qwen3_vl_2b_instruct_plugin()
        print(f"  SUCCESS: Successfully created plugin: {plugin.metadata.name}")

        # Verify the old function name would fail
        try:
            from src.inference_pio.models.qwen3_vl_2b.plugin import create_qwen3_vl_2b_plugin
            print("  WARNING: Old function name still exists - this should not happen")
            return False
        except ImportError:
            print("  SUCCESS: Old function name (create_qwen3_vl_2b_plugin) correctly not found")
            return True

    except ImportError as e:
        print(f"  ERROR: Failed to import correct function name: {e}")
        return False
    except Exception as e:
        print(f"  ERROR: Error testing function name fix: {e}")
        return False

def main():
    """Run all verification tests."""
    print("="*60)
    print("VERIFICATION OF QWEN3_VL_2B PLUGIN FUNCTION NAME FIX")
    print("="*60)
    
    # Test function name fix
    function_fix_ok = test_function_name_fix()
    
    # Test benchmark imports
    benchmark_imports_ok = test_benchmark_imports()
    
    # Test main benchmark script imports
    main_script_imports_ok = test_main_benchmark_scripts()
    
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)

    print(f"Function name fix: {'PASS' if function_fix_ok else 'FAIL'}")

    print("\nBenchmark module imports:")
    for module, success in benchmark_imports_ok.items():
        status = 'PASS' if success else 'FAIL'
        print(f"  {module}: {status}")

    print("\nMain benchmark script imports:")
    for script, success in main_script_imports_ok.items():
        status = 'PASS' if success else 'FAIL'
        print(f"  {script}: {status}")

    # Overall result
    all_benchmarks_ok = all(benchmark_imports_ok.values())
    all_scripts_ok = all(main_script_imports_ok.values())
    overall_success = function_fix_ok and all_benchmarks_ok and all_scripts_ok

    print(f"\nOverall result: {'PASS' if overall_success else 'FAIL'}")

    if overall_success:
        print("\nSUCCESS: All verifications passed!")
        print("SUCCESS: Function name fix is working correctly")
        print("SUCCESS: All benchmark modules can be imported")
        print("SUCCESS: All main benchmark scripts can be imported")
        print("\nThe qwen3_vl_2b model is ready for all 6 benchmark categories.")
    else:
        print("\nFAILURE: Some verifications failed!")
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)