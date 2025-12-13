"""
Proper import standardization script for the Qwen3-VL codebase.
Converts relative imports to absolute imports while preserving correct syntax.
"""
import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple


def get_absolute_module_path(file_path: str, relative_level: int, relative_module: str) -> str:
    """
    Convert a relative import to an absolute import path.
    
    Args:
        file_path: The path of the file containing the relative import
        relative_level: The number of dots in the relative import (1 for '.', 2 for '..', etc.)
        relative_module: The module being imported (after the dots)
    
    Returns:
        The absolute module path
    """
    # Convert file path to parts
    file_parts = Path(file_path).parts
    
    # Find the 'src' directory to establish the base package
    src_index = -1
    for i, part in enumerate(file_parts):
        if part == 'src':
            src_index = i
            break
    
    if src_index == -1:
        # If we can't find 'src', we can't convert relative to absolute reliably
        raise ValueError(f"Cannot determine absolute path for {file_path}: no 'src' directory found")
    
    # Get the parts from 'src' onwards
    module_parts = file_parts[src_index + 1:]
    
    # Remove the filename and split into package parts
    if module_parts[-1].endswith('.py'):
        module_parts = module_parts[:-1]
    
    # Go up 'relative_level' number of levels
    if relative_level > len(module_parts):
        # This is an invalid relative import
        raise ValueError(f"Invalid relative import in {file_path}: going up {relative_level} levels from {len(module_parts)} available")
    
    # Calculate the base path
    base_parts = module_parts[:-(relative_level)] if relative_level > 0 else module_parts
    
    # Combine with the relative module path
    if relative_module:
        target_parts = base_parts + tuple(relative_module.split('.')) if relative_module else base_parts
    else:
        target_parts = base_parts
    
    return '.'.join(target_parts)


def convert_file_imports(file_path: str) -> bool:
    """
    Convert relative imports in a file to absolute imports.
    
    Args:
        file_path: Path to the Python file to process
        
    Returns:
        True if the file was modified, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Parse the AST to identify import statements
        tree = ast.parse(original_content)
    except (SyntaxError, UnicodeDecodeError):
        return False
    
    # Find all ImportFrom nodes with relative imports
    relative_import_nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            relative_import_nodes.append(node)
    
    if not relative_import_nodes:
        return False  # No relative imports to fix
    
    # Split content into lines for easier manipulation
    lines = original_content.splitlines()
    
    # Process imports in reverse order to maintain line positions
    modified = False
    for node in sorted(relative_import_nodes, key=lambda n: n.lineno, reverse=True):
        try:
            # Get the absolute module path
            absolute_module = get_absolute_module_path(file_path, node.level, node.module or '')
            
            # Create the new import statement
            names = []
            for alias in node.names:
                if alias.asname:
                    names.append(f"{alias.name} as {alias.asname}")
                else:
                    names.append(alias.name)
            
            new_import = f"from {absolute_module} import {', '.join(names)}"
            
            # Replace the line (adjust for 0-indexing)
            line_index = node.lineno - 1
            original_line = lines[line_index]
            
            # Replace the import statement in the line
            # Find the 'from' statement in the line
            from_pattern = r'from\s+\.{' + str(node.level) + r'}' + (re.escape(node.module) if node.module else '') + r'\s+import'
            if re.search(from_pattern, original_line):
                # Replace the relative import with the absolute import
                new_line = re.sub(from_pattern, f"from {absolute_module} import", original_line)
                lines[line_index] = new_line
                modified = True
            else:
                # Alternative approach: reconstruct the whole import statement
                # Find where the 'from' statement starts and ends
                current_line = lines[line_index]
                
                # Look for the pattern in the current line
                start_pos = current_line.find(f"from {'.' * node.level}{node.module if node.module else ''} import")
                if start_pos != -1:
                    # Replace the import statement
                    lines[line_index] = f"from {absolute_module} import {', '.join(names)}"
                    modified = True
                else:
                    # Multi-line import - we need to handle this case
                    # For now, just skip it and log a warning
                    print(f"Warning: Complex multi-line import in {file_path}:{node.lineno}, skipping...")
        
        except ValueError as e:
            print(f"Warning: {e}, keeping original relative import in {file_path}:{node.lineno}")
            continue
    
    if modified:
        # Write the modified content back to the file
        new_content = '\n'.join(lines)
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Fixed imports in: {file_path}")
            return True
    
    return False


def standardize_imports_in_directory(directory: str):
    """
    Standardize imports in all Python files in the specified directory.
    """
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
            if convert_file_imports(file_path):
                fixed_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            error_count += 1
    
    print(f"\nImport standardization complete!")
    print(f"Files processed: {len(python_files)}")
    print(f"Files with fixed imports: {fixed_count}")
    print(f"Files with errors: {error_count}")


def verify_conversion(directory: str) -> bool:
    """
    Verify that the conversion worked properly by checking for remaining relative imports.
    """
    print("\nVerifying conversion...")
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    files_with_relative_imports = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for relative imports (patterns like 'from .' or 'from ..')
            if re.search(r'from\s+\.\.?', content):
                files_with_relative_imports.append(file_path)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    
    if files_with_relative_imports:
        print(f"\nFound {len(files_with_relative_imports)} files with remaining relative imports:")
        for f in files_with_relative_imports[:10]:
            print(f"  {f}")
        if len(files_with_relative_imports) > 10:
            print(f"  ... and {len(files_with_relative_imports) - 10} more")
        return False
    else:
        print("\n✅ Verification successful: No relative imports found!")
        return True


if __name__ == "__main__":
    # Set the root directory
    root_dir = r"C:\\Users\\Admin\\Documents\\GitHub\\Mod\\src\\qwen3_vl"
    
    if not os.path.exists(root_dir):
        print(f"Directory does not exist: {root_dir}")
        exit(1)
    
    print("Starting import standardization process...")
    print("="*50)
    
    # Standardize imports
    standardize_imports_in_directory(root_dir)
    
    # Verify the results
    success = verify_conversion(root_dir)
    
    if success:
        print("\n✅ Import standardization completed successfully!")
    else:
        print("\n⚠️  Some relative imports remain. Manual review required.")