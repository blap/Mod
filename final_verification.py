"""
Final Verification Script for Inference-PIO Tests

This script verifies that all updated tests work correctly with the new systems
and follow the new standards. It runs a representative sample of tests to ensure
everything is functioning properly.
"""

import subprocess
import sys
import os
from pathlib import Path
import json


def run_test_file(test_file_path):
    """Run a single test file and return the result."""
    try:
        result = subprocess.run([
            sys.executable, test_file_path
        ], capture_output=True, text=True, timeout=120)  # 2 minute timeout
        
        return {
            'file': test_file_path,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            'file': test_file_path,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Timeout expired',
            'success': False
        }
    except Exception as e:
        return {
            'file': test_file_path,
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }


def verify_updated_tests():
    """Verify that the updated tests work correctly."""
    updated_test_files = [
        'tests/unit/test_fixtures_system_updated.py',
        'tests/unit/common/test_contractual_interfaces_updated.py',
        'tests/integration/test_benchmark_discovery_updated.py',
        'tests/performance/test_performance_regression_system_updated.py'
    ]
    
    results = []
    
    print("Verifying updated tests...")
    for test_file in updated_test_files:
        full_path = Path(test_file)
        if full_path.exists():
            print(f"Running {test_file}...")
            result = run_test_file(str(full_path))
            results.append(result)
            
            if result['success']:
                print(f"  [PASS] PASSED")
            else:
                print(f"  [FAIL] FAILED - Return code: {result['returncode']}")
                if result['stderr']:
                    print(f"     Error: {result['stderr'][:200]}...")  # First 200 chars
        else:
            print(f"  ⚠️  File not found: {test_file}")
            results.append({
                'file': test_file,
                'returncode': -1,
                'stdout': '',
                'stderr': 'File not found',
                'success': False
            })
    
    return results


def verify_core_functionality_tests():
    """Verify core functionality tests still work."""
    core_test_files = [
        'src/inference_pio/test_fixtures.py',
        'src/inference_pio/test_utils.py'
    ]
    
    results = []
    
    print("\nVerifying core functionality tests...")
    for test_file in core_test_files:
        full_path = Path(test_file)
        if full_path.exists():
            print(f"Checking {test_file}...")
            # These are utility files, not meant to be run directly
            # Just verify they can be imported without errors
            try:
                import_result = subprocess.run([
                    sys.executable, '-c', f'import sys; sys.path.insert(0, "src"); import {test_file.replace("/", ".").replace(".py", "").replace("src.","")}'
                ], capture_output=True, text=True, timeout=30)
                
                result = {
                    'file': test_file,
                    'returncode': import_result.returncode,
                    'stdout': import_result.stdout,
                    'stderr': import_result.stderr,
                    'success': import_result.returncode == 0
                }
                results.append(result)
                
                if result['success']:
                    print(f"  ✅ IMPORT SUCCESSFUL")
                else:
                    print(f"  ❌ IMPORT FAILED - Return code: {result['returncode']}")
                    if result['stderr']:
                        print(f"     Error: {result['stderr'][:200]}...")
            except Exception as e:
                result = {
                    'file': test_file,
                    'returncode': -1,
                    'stdout': '',
                    'stderr': str(e),
                    'success': False
                }
                results.append(result)
                print(f"  ❌ ERROR: {str(e)}")
        else:
            print(f"  ⚠️  File not found: {test_file}")
            results.append({
                'file': test_file,
                'returncode': -1,
                'stdout': '',
                'stderr': 'File not found',
                'success': False
            })
    
    return results


def verify_standard_components():
    """Verify that standard components work with the new tests."""
    print("\nVerifying standard components...")
    
    # Test that the test utilities work correctly
    test_util_code = '''
from src.inference_pio.test_utils import *
import torch

def test_basic_assertions():
    """Test basic assertion functionality."""
    try:
        assert_equal(1, 1)
        assert_true(True)
        assert_false(False)
        assert_is_not_none("test")
        print("Basic assertions: PASSED")
        return True
    except Exception as e:
        print(f"Basic assertions: FAILED - {e}")
        return False

def test_tensor_assertions():
    """Test tensor-specific assertions."""
    try:
        t1 = torch.tensor([1, 2, 3])
        t2 = torch.tensor([1, 2, 3])
        assert_tensor_equal(t1, t2)
        print("Tensor assertions: PASSED")
        return True
    except Exception as e:
        print(f"Tensor assertions: FAILED - {e}")
        return False

def test_advanced_assertions():
    """Test advanced assertion functionality."""
    try:
        assert_between(5, 1, 10)
        assert_length([1, 2, 3], 3)
        assert_in("a", ["a", "b", "c"])
        print("Advanced assertions: PASSED")
        return True
    except Exception as e:
        print(f"Advanced assertions: FAILED - {e}")
        return False

if __name__ == "__main__":
    results = [
        test_basic_assertions(),
        test_tensor_assertions(),
        test_advanced_assertions()
    ]
    all_passed = all(results)
    print(f"All utility tests: {'PASSED' if all_passed else 'FAILED'}")
    exit(0 if all_passed else 1)
'''
    
    # Write test to temporary file and run it
    with open('temp_test_utils_verification.py', 'w') as f:
        f.write(test_util_code)
    
    try:
        result = subprocess.run([
            sys.executable, 'temp_test_utils_verification.py'
        ], capture_output=True, text=True, timeout=30)
        
        util_test_result = {
            'file': 'test_utils_verification',
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0
        }
        
        print(f"Test utilities verification: {'✅ PASSED' if util_test_result['success'] else '❌ FAILED'}")
        if not util_test_result['success']:
            print(f"  Error: {result.stderr}")
        
        return [util_test_result]
    except Exception as e:
        print(f"Test utilities verification: ❌ ERROR - {e}")
        return [{
            'file': 'test_utils_verification',
            'returncode': -1,
            'stdout': '',
            'stderr': str(e),
            'success': False
        }]
    finally:
        # Clean up
        if os.path.exists('temp_test_utils_verification.py'):
            os.remove('temp_test_utils_verification.py')


def generate_verification_report(updated_results, core_results, util_results):
    """Generate a verification report."""
    all_results = updated_results + core_results + util_results
    
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r['success'])
    failed_tests = total_tests - passed_tests
    
    report = {
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
        },
        'updated_tests': updated_results,
        'core_tests': core_results,
        'utility_tests': util_results,
        'all_tests': all_results
    }
    
    return report


def print_verification_report(report):
    """Print a human-readable verification report."""
    print("\n" + "="*80)
    print("FINAL VERIFICATION REPORT")
    print("="*80)
    
    summary = report['summary']
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.2f}%")
    
    print(f"\nUpdated Tests Status:")
    for result in report['updated_tests']:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {status}: {result['file']}")
    
    print(f"\nCore Components Status:")
    for result in report['core_tests']:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {status}: {result['file']}")
    
    print(f"\nUtility Functions Status:")
    for result in report['utility_tests']:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {status}: {result['file']}")
    
    if summary['failed_tests'] > 0:
        print(f"\nFAILED TEST DETAILS:")
        for result in report['all_tests']:
            if not result['success']:
                print(f"\n  File: {result['file']}")
                print(f"  Return Code: {result['returncode']}")
                if result['stderr']:
                    print(f"  Error: {result['stderr'][:200]}...")
    
    print("="*80)
    
    return summary['success_rate'] >= 95  # Require 95% success rate


def main():
    """Main verification function."""
    print("Starting final verification of updated tests...")
    
    # Verify updated tests
    updated_results = verify_updated_tests()
    
    # Verify core functionality
    core_results = verify_core_functionality_tests()
    
    # Verify utility functions
    util_results = verify_standard_components()
    
    # Generate and print report
    report = generate_verification_report(updated_results, core_results, util_results)
    print_verification_report(report)
    
    # Save detailed report
    with open('final_verification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to final_verification_report.json")
    
    # Determine overall success
    overall_success = report['summary']['success_rate'] >= 95
    
    if overall_success:
        print(f"\n🎉 VERIFICATION SUCCESSFUL! All systems are working correctly.")
        print(f"   Success rate of {report['summary']['success_rate']:.2f}% meets the 95% threshold.")
    else:
        print(f"\n⚠️  VERIFICATION PARTIALLY FAILED! Success rate is below 95%.")
        print(f"   Current success rate: {report['summary']['success_rate']:.2f}%")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)