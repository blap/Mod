"""
Test script to identify import inconsistencies and standardization issues in the codebase.
"""
import os
import sys
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple


def find_python_files(root_dir: str) -> List[str]:
    """Find all Python files in the given directory."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def extract_imports(file_path: str) -> Dict[str, List[str]]:
    """Extract all imports from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            print(f"Syntax error in file: {file_path}")
            return {'absolute': [], 'relative': []}
    
    absolute_imports = []
    relative_imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                absolute_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:  # Absolute import
                module = node.module or ''
                absolute_imports.append(module)
            else:  # Relative import
                module = node.module or ''
                relative_imports.append((node.level, module))
    
    return {
        'absolute': absolute_imports,
        'relative': relative_imports
    }


def find_import_patterns(root_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """Find all import patterns in the codebase."""
    all_imports = {}
    
    python_files = find_python_files(root_dir)
    
    for file_path in python_files:
        imports = extract_imports(file_path)
        all_imports[file_path] = imports
    
    return all_imports


def analyze_import_consistency(root_dir: str) -> Dict[str, List[str]]:
    """Analyze import consistency across the codebase."""
    # Find all Python files
    python_files = find_python_files(root_dir)
    
    # Dictionary to store import patterns for each module
    module_imports = {}
    
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    # Record the import path used for this module
                    if node.module not in module_imports:
                        module_imports[node.module] = []
                    
                    # Determine if it's relative or absolute
                    if node.level > 0:
                        # Relative import
                        import_path = '.' * node.level + (node.module if node.module else '')
                    else:
                        # Absolute import
                        import_path = node.module
                    
                    if import_path not in module_imports[node.module]:
                        module_imports[node.module].append(import_path)
    
    # Find modules with inconsistent import patterns
    inconsistent_modules = {}
    for module, paths in module_imports.items():
        if len(paths) > 1:
            inconsistent_modules[module] = paths
    
    return inconsistent_modules


def check_import_standardization_issues(root_dir: str):
    """Check for import standardization issues."""
    print("Analyzing import patterns...")
    
    # Find inconsistent import patterns
    inconsistent_modules = analyze_import_consistency(root_dir)
    
    if inconsistent_modules:
        print("\nFound inconsistent import patterns:")
        for module, paths in inconsistent_modules.items():
            print(f"  Module '{module}' imported as: {paths}")
    else:
        print("\nNo inconsistent import patterns found.")
    
    # Find all Python files to check for absolute vs relative import usage
    python_files = find_python_files(root_dir)
    
    absolute_import_files = []
    relative_import_files = []
    
    for file_path in python_files:
        imports = extract_imports(file_path)
        has_absolute = len(imports['absolute']) > 0
        has_relative = len(imports['relative']) > 0
        
        if has_absolute:
            absolute_import_files.append(file_path)
        if has_relative:
            relative_import_files.append(file_path)
    
    print(f"\nFiles with absolute imports: {len(absolute_import_files)}")
    print(f"Files with relative imports: {len(relative_import_files)}")
    
    # Check for files that mix absolute and relative imports
    mixed_import_files = set(absolute_import_files) & set(relative_import_files)
    if mixed_import_files:
        print(f"\nFiles with mixed import styles: {len(mixed_import_files)}")
        for f in list(mixed_import_files)[:5]:  # Show first 5
            print(f"  {f}")
    
    return inconsistent_modules


def main():
    # Set the root directory to analyze
    root_dir = r"C:\Users\Admin\Documents\GitHub\Mod\src\qwen3_vl"
    
    if not os.path.exists(root_dir):
        print(f"Directory does not exist: {root_dir}")
        return
    
    print("Checking import standardization in the Qwen3-VL codebase...")
    inconsistent_modules = check_import_standardization_issues(root_dir)
    
    if inconsistent_modules:
        print("\nSUMMARY: Import standardization issues found!")
        print("These modules are imported using different patterns and should be standardized.")
    else:
        print("\nNo import standardization issues found!")


if __name__ == "__main__":
    main()