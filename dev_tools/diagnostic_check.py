#!/usr/bin/env python
"""
Quick diagnostic test to check the health of the codebase
"""

import sys
import os
from pathlib import Path

def check_syntax_errors():
    """Check for syntax errors in critical files."""
    print("Checking for syntax errors in critical files...")
    
    critical_files = [
        "tests/test_utilities.py",
        "tests/test_runner.py", 
        "tests/framework.py",
        "tests/test_execution_config.py",
        "test_centralized_utilities.py"
    ]
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            try:
                compile(open(file_path).read(), file_path, 'exec')
                print(f"✓ {file_path} - OK")
            except SyntaxError as e:
                print(f"✗ {file_path} - SYNTAX ERROR: {e}")
                return False
        else:
            print(f"? {file_path} - NOT FOUND")
    
    return True

def check_imports():
    """Check if critical modules can be imported."""
    print("\nChecking critical imports...")
    
    modules_to_test = [
        ("tests.test_utilities", "assert_equal"),
        ("tests.test_runner", "TestRunner"),
        ("tests.framework", "TestFrameworkConfig"),
        ("tests.test_execution_config", "TestExecutor"),
    ]
    
    for module_name, attr_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            if hasattr(module, attr_name):
                print(f"✓ {module_name}.{attr_name} - OK")
            else:
                print(f"? {module_name}.{attr_name} - ATTR NOT FOUND")
        except ImportError as e:
            print(f"✗ {module_name} - IMPORT ERROR: {e}")
            return False
        except Exception as e:
            print(f"✗ {module_name} - ERROR: {e}")
            return False
    
    return True

def main():
    print("Running diagnostic checks on Qwen3-VL codebase...")
    print("="*60)
    
    syntax_ok = check_syntax_errors()
    imports_ok = check_imports()
    
    print("\n" + "="*60)
    if syntax_ok and imports_ok:
        print("✓ DIAGNOSTIC CHECKS PASSED - Codebase appears healthy")
        return 0
    else:
        print("✗ DIAGNOSTIC CHECKS FAILED - Issues detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())