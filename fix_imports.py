"""
Script to identify and fix import inconsistencies in the codebase.
This script identifies files with problematic import patterns and fixes them.
"""
import os
import re
import ast
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


def normalize_import_path(file_path: str, import_path: str, level: int) -> str:
    """Normalize a relative import path to an absolute import path."""
    # Convert file path to a module path
    file_parts = Path(file_path).parts
    # Find the 'src' part to determine the base module
    src_index = -1
    for i, part in enumerate(file_parts):
        if part == 'src':
            src_index = i
            break
    
    if src_index == -1:
        return import_path  # Cannot determine base module
    
    # Get the module parts after 'src'
    module_parts = file_parts[src_index + 1:]
    
    # Calculate the target module path based on relative import
    if level == 0:
        # Already absolute
        return import_path
    else:
        # Calculate the base path by going up 'level' times
        base_parts = module_parts[:-level]
        
        if import_path:
            # Join the base path with the import path
            target_parts = base_parts + import_path.split('.')
        else:
            # Just go up by 'level' and stay in the same directory
            target_parts = base_parts
        
        # Construct the normalized path
        normalized = '.'.join(target_parts)
        return normalized


def standardize_imports_in_file(file_path: str) -> bool:
    """Standardize imports in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Parse the AST to identify import statements
    try:
        tree = ast.parse(original_content)
    except SyntaxError:
        print(f"Syntax error in file: {file_path}")
        return False
    
    # Find all import statements
    import_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(node)
    
    # Sort nodes by line number in reverse order so replacements don't affect positions
    import_nodes.sort(key=lambda x: x.lineno, reverse=True)
    
    modified_content = original_content
    lines = modified_content.splitlines()
    
    # Process each import statement
    for node in import_nodes:
        if isinstance(node, ast.ImportFrom) and node.module and node.level > 0:
            # This is a relative import, convert to absolute
            relative_path = node.module or ''
            
            # Calculate the absolute module path
            absolute_path = normalize_import_path(file_path, relative_path, node.level)
            
            # Replace the import statement
            start_line = node.lineno - 1
            end_line = getattr(node, 'end_lineno', start_line)  # Handle Python < 3.8
            
            # Get the original line
            original_line = lines[start_line]
            
            # Create the new import statement
            names = ', '.join(alias.name + (' as ' + alias.asname if alias.asname else '') 
                             for alias in node.names)
            new_import = f"from {absolute_path} import {names}"
            
            # Replace the line
            lines[start_line] = original_line.replace(
                f"from {'.' * node.level}{relative_path}", 
                f"from {absolute_path}"
            )
            
            # Update modified content
            modified_content = '\\n'.join(lines)
    
    # Write back if changed
    if modified_content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"Fixed imports in: {file_path}")
        return True
    
    return False


def standardize_imports_in_directory(root_dir: str):
    """Standardize imports in all Python files in the directory."""
    python_files = find_python_files(root_dir)
    
    fixed_count = 0
    total_count = 0
    
    for file_path in python_files:
        total_count += 1
        if standardize_imports_in_file(file_path):
            fixed_count += 1
    
    print(f"\\nProcessed {total_count} files, fixed imports in {fixed_count} files.")


def analyze_imports_detailed(root_dir: str):
    """Detailed analysis of import patterns."""
    python_files = find_python_files(root_dir)
    
    relative_imports = []
    absolute_imports = []
    
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue
        
        has_relative = False
        has_absolute = False
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                has_absolute = True
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    has_relative = True
                    relative_imports.append((file_path, node.module, node.level))
                else:
                    has_absolute = True
                    absolute_imports.append((file_path, node.module))
        
        # Print files with mixed import styles
        if has_relative and has_absolute:
            print(f"Mixed imports: {file_path}")
    
    print(f"\\nFound {len(relative_imports)} relative imports")
    print(f"Found {len(absolute_imports)} absolute imports")
    
    # Show some examples of relative imports
    print("\\nSample relative imports:")
    for i, (file_path, module, level) in enumerate(relative_imports[:10]):
        print(f"  {file_path}: from {'.' * level}{module}")
    
    if len(relative_imports) > 10:
        print(f"  ... and {len(relative_imports) - 10} more")


if __name__ == "__main__":
    root_dir = r"C:\\Users\\Admin\\Documents\\GitHub\\Mod\\src\\qwen3_vl"
    
    if not os.path.exists(root_dir):
        print(f"Directory does not exist: {root_dir}")
        exit(1)
    
    print("Analyzing import patterns...")
    analyze_imports_detailed(root_dir)
    
    print("\\nStandardizing imports...")
    standardize_imports_in_directory(root_dir)
    
    print("\\nImport standardization complete!")