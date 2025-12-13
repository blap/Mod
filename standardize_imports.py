"""
Fix import standardization issues in the Qwen3-VL codebase.
Convert all relative imports to absolute imports for consistency.
"""
import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple


def get_package_path(file_path: str, package_root: str) -> str:
    """Convert a file path to its corresponding package path."""
    file_path_obj = Path(file_path)
    package_root_obj = Path(package_root)
    
    # Get the relative path from package root
    relative_path = file_path_obj.relative_to(package_root_obj)
    
    # Convert to package notation (replace separators with dots, remove .py)
    parts = list(relative_path.parts)
    if parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]  # Remove .py extension
    
    return '.'.join(parts)


def convert_relative_to_absolute(file_path: str, package_root: str) -> bool:
    """Convert relative imports in a file to absolute imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        tree = ast.parse(original_content)
    except (SyntaxError, UnicodeDecodeError):
        return False
    
    # Find the package path for this file
    file_package_path = get_package_path(file_path, package_root)
    file_package_parts = file_package_path.split('.')
    
    # Find all relative imports
    relative_import_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            relative_import_nodes.append({
                'node': node,
                'line_start': node.lineno - 1,
                'line_end': getattr(node, 'end_lineno', node.lineno) - 1 if hasattr(node, 'end_lineno') else node.lineno - 1,
                'level': node.level,
                'module': node.module or '',
                'names': node.names
            })
    
    if not relative_import_nodes:
        return False  # No relative imports to fix
    
    # Process lines to replace relative imports with absolute ones
    lines = original_content.splitlines()
    
    # Process in reverse order to maintain line numbers during replacement
    for import_info in reversed(relative_import_nodes):
        node = import_info['node']
        level = import_info['level']
        module = import_info['module']
        
        # Calculate the absolute module path
        # level=1 means same directory or parent, level=2 means grandparent, etc.
        if level <= len(file_package_parts):
            # Go up 'level' number of parts
            base_parts = file_package_parts[:-level] if level > 0 else file_package_parts[:]
            if module:
                # Append the relative module path
                target_parts = base_parts + module.split('.') if module else base_parts
            else:
                # Just the base path (when using 'from .. import something')
                target_parts = base_parts
            absolute_module = '.'.join(target_parts)
        else:
            # If level is greater than available parts, we can't resolve it properly
            # Keep the relative import as is to avoid breaking the code
            print(f"Warning: Could not resolve relative import in {file_path}: from {'.' * level}{module}")
            continue
        
        # Reconstruct the import statement
        names_list = []
        for alias in node.names:
            name_part = alias.name
            if alias.asname:
                name_part += f" as {alias.asname}"
            names_list.append(name_part)
        
        new_import_line = f"from {absolute_module} import {', '.join(names_list)}"
        
        # Replace the line
        line_idx = import_info['line_start']
        lines[line_idx] = new_import_line
    
    # Write the modified content back
    new_content = '\\n'.join(lines)
    if new_content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed imports in: {file_path}")
        return True
    
    return False


def fix_all_imports_in_directory(directory: str):
    """Fix all relative imports in the specified directory."""
    print(f"Scanning directory: {directory}")
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    
    fixed_count = 0
    error_count = 0
    
    for file_path in python_files:
        try:
            if convert_relative_to_absolute(file_path, directory):
                fixed_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            error_count += 1
    
    print(f"\\nImport standardization complete!")
    print(f"Files processed: {len(python_files)}")
    print(f"Files with fixed imports: {fixed_count}")
    print(f"Files with errors: {error_count}")


def verify_import_consistency(directory: str):
    """Verify that all imports are now consistent."""
    print("\\nVerifying import consistency...")
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    inconsistent_files = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            continue
        
        has_relative = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                has_relative = True
                break
        
        if has_relative:
            inconsistent_files.append(file_path)
    
    if inconsistent_files:
        print(f"\\nFound {len(inconsistent_files)} files with remaining relative imports:")
        for f in inconsistent_files[:10]:  # Show first 10
            print(f"  {f}")
        if len(inconsistent_files) > 10:
            print(f"  ... and {len(inconsistent_files) - 10} more")
    else:
        print("\\nAll imports are now standardized!")
    
    return len(inconsistent_files) == 0


if __name__ == "__main__":
    # Set the root directory
    root_dir = r"C:\\Users\\Admin\\Documents\\GitHub\\Mod\\src\\qwen3_vl"
    
    if not os.path.exists(root_dir):
        print(f"Directory does not exist: {root_dir}")
        exit(1)
    
    print("Starting import standardization process...")
    print("="*50)
    
    # Fix all relative imports
    fix_all_imports_in_directory(root_dir)
    
    # Verify the results
    success = verify_import_consistency(root_dir)
    
    if success:
        print("\\n✅ Import standardization completed successfully!")
    else:
        print("\\n⚠️  Some relative imports remain. Please check the output above.")