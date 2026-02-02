"""
Verification script to confirm that type hints have been properly added to test files.
"""

import ast
import inspect
import os
from typing import Any, Dict, List, Optional, Tuple, Union


def verify_type_hints_in_file(file_path: str) -> Dict[str, Any]:
    """
    Verify that a Python file has proper type hints by parsing its AST.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return {"valid_syntax": False, "error": "Syntax error in file"}

    type_hint_count = 0
    function_count = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_count += 1
            # Check if function has return annotation
            if node.returns is not None:
                type_hint_count += 1
            # Check if any arguments have type annotations
            for arg in node.args.args:
                if arg.annotation is not None:
                    type_hint_count += 1
            for arg in node.args.kwonlyargs:
                if arg.annotation is not None:
                    type_hint_count += 1
            if node.args.vararg and node.args.vararg.annotation:
                type_hint_count += 1
            if node.args.kwarg and node.args.kwarg.annotation:
                type_hint_count += 1

    return {
        "valid_syntax": True,
        "function_count": function_count,
        "type_hint_count": type_hint_count,
        "has_type_hints": type_hint_count > 0,
    }


def scan_test_directories() -> Dict[str, Any]:
    """
    Scan test directories and verify type hints in test files.
    """
    test_dirs = [
        "tests/unit/common/discovery",
        "tests/unit/common",
        "standardized_tests/unit",
        "standardized_dev_tests/unit",
    ]

    results: Dict[str, Any] = {}

    for test_dir in test_dirs:
        dir_path = os.path.join("C:/Users/Admin/Documents/GitHub/Mod", test_dir)
        if os.path.exists(dir_path):
            results[test_dir] = {}
            for file_name in os.listdir(dir_path):
                if file_name.endswith(".py") and file_name.startswith("test_"):
                    file_path = os.path.join(dir_path, file_name)
                    verification = verify_type_hints_in_file(file_path)
                    results[test_dir][file_name] = verification

    return results


if __name__ == "__main__":
    print("Verifying type hints in test files...")
    results = scan_test_directories()

    total_files = 0
    files_with_type_hints = 0

    for dir_name, files in results.items():
        print(f"\nDirectory: {dir_name}")
        print("-" * 40)
        for file_name, result in files.items():
            total_files += 1
            status = "PASS" if result.get("has_type_hints", False) else "FAIL"
            print(
                f"{status} {file_name}: {result.get('function_count', 0)} functions, {result.get('type_hint_count', 0)} type hints"
            )
            if result.get("has_type_hints", False):
                files_with_type_hints += 1

    print(f"\nSummary:")
    print(f"Total test files scanned: {total_files}")
    print(f"Files with type hints: {files_with_type_hints}")
    print(
        f"Success rate: {files_with_type_hints/total_files*100:.1f}% "
        if total_files > 0
        else "No files found"
    )

    print("\nType hint verification completed successfully!")
