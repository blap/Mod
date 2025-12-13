"""
Final import standardization script that properly converts relative imports to absolute imports
while maintaining code integrity.
"""
import os
import ast
import re
from pathlib import Path


def get_absolute_module_path(file_path: str, relative_level: int, relative_module: str) -> str:
    """
    Convert a relative import to an absolute import path.
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
        # If we can't find 'src', return the original relative import
        return f"{'.' * relative_level}{relative_module}" if relative_module else f"{'.' * relative_level}"
    
    # Get the parts from 'src' onwards
    module_parts = file_parts[src_index + 1:]
    
    # Remove the filename and split into package parts
    if module_parts and module_parts[-1].endswith('.py'):
        module_parts = module_parts[:-1]
    
    # Go up 'relative_level' number of levels
    if relative_level > len(module_parts):
        # This is an invalid relative import, return as-is
        return f"{'.' * relative_level}{relative_module}" if relative_module else f"{'.' * relative_level}"
    
    # Calculate the base path
    base_parts = module_parts[:-(relative_level)] if relative_level > 0 else module_parts
    
    # Combine with the relative module path
    if relative_module:
        target_parts = base_parts + tuple(relative_module.split('.'))
    else:
        target_parts = base_parts
    
    return '.'.join(target_parts)


def fix_imports_in_file(file_path: str) -> bool:
    """
    Fix relative imports in a single file by converting them to absolute imports.
    This function carefully preserves the code structure and only modifies import statements.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return False  # Skip files that can't be decoded as text
    
    # Parse the original content to identify relative imports
    try:
        tree = ast.parse(content)
    except SyntaxError:
        # If the file already has syntax errors, skip it
        print(f"Skipping {file_path} due to existing syntax errors")
        return False
    
    # Find all relative imports
    relative_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            relative_imports.append({
                'node': node,
                'lineno': node.lineno,
                'level': node.level,
                'module': node.module or '',
                'names': node.names
            })
    
    if not relative_imports:
        return False  # No relative imports to fix
    
    # Split content into lines to process each import
    lines = content.splitlines()
    
    # Process each relative import
    modified = False
    for import_info in reversed(relative_imports):  # Process in reverse to maintain line numbers
        node = import_info['node']
        lineno = import_info['lineno'] - 1  # Convert to 0-indexed
        
        # Get the absolute module path
        absolute_module = get_absolute_module_path(file_path, node.level, node.module or '')
        
        # Create the new import statement
        names_list = []
        for alias in node.names:
            if alias.asname:
                names_list.append(f"{alias.name} as {alias.asname}")
            else:
                names_list.append(alias.name)
        
        new_import = f"from {absolute_module} import {', '.join(names_list)}"
        
        # Replace the line containing the relative import
        original_line = lines[lineno]
        
        # Find the relative import pattern in the line
        relative_pattern = f"from {'.' * node.level}{node.module or ''} import"
        if relative_pattern in original_line:
            lines[lineno] = new_import
            modified = True
        else:
            # Try to find other variations of the pattern
            pattern = r'from\s+\.{1,' + str(node.level) + r'}' + (re.escape(node.module) if node.module else '') + r'\s+import'
            if re.search(pattern, original_line):
                lines[lineno] = new_import
                modified = True
            else:
                # Multi-line import or complex case - skip for now
                print(f"Complex import pattern in {file_path}:{lineno+1}, skipping...")
    
    if modified:
        new_content = '\\n'.join(lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed imports in: {file_path}")
        return True
    
    return False


def standardize_all_imports(root_dir: str):
    """
    Standardize all imports in the given directory.
    """
    print(f"Standardizing imports in: {root_dir}")
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    
    fixed_count = 0
    for file_path in python_files:
        try:
            if fix_imports_in_file(file_path):
                fixed_count += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"\\nImport standardization complete!")
    print(f"Files processed: {len(python_files)}")
    print(f"Files with fixed imports: {fixed_count}")


def verify_standardization(root_dir: str):
    """
    Verify that all imports have been standardized.
    """
    print("\\nVerifying import standardization...")
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    files_with_relative_imports = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for relative imports (from . or from ..)
            if re.search(r'^\s*from\s+\.\.?', content, re.MULTILINE):
                # Find which lines have relative imports
                lines = content.splitlines()
                for i, line in enumerate(lines, 1):
                    if re.match(r'^\s*from\s+\.', line.strip()):
                        files_with_relative_imports.append((file_path, i, line.strip()))
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    
    if files_with_relative_imports:
        print(f"\\nFound {len(files_with_relative_imports)} relative imports in {len(set(f[0] for f in files_with_relative_imports))} files:")
        for file_path, line_num, line_content in files_with_relative_imports[:20]:
            print(f"  {file_path}:{line_num} - {line_content}")
        if len(files_with_relative_imports) > 20:
            print(f"  ... and {len(files_with_relative_imports) - 20} more")
        return False
    else:
        print("\\n✅ All imports have been standardized successfully!")
        return True


if __name__ == "__main__":
    root_dir = r"C:\\Users\\Admin\\Documents\\GitHub\\Mod\\src\\qwen3_vl"
    
    if not os.path.exists(root_dir):
        print(f"Directory does not exist: {root_dir}")
        exit(1)
    
    print("Starting import standardization process...")
    print("=" * 60)
    
    # Standardize imports
    standardize_all_imports(root_dir)
    
    # Verify the results
    success = verify_standardization(root_dir)
    
    if success:
        print("\\n🎉 Import standardization completed successfully!")
        print("All relative imports have been converted to absolute imports.")
    else:
        print("\\n⚠️  Some relative imports remain. Please review the listed files.")