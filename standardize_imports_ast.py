"""
Correct import standardization script using AST transformations.
This properly modifies the AST and regenerates the code.
"""
import ast
import os
import sys
from pathlib import Path


class RelativeToAbsoluteImportTransformer(ast.NodeTransformer):
    """
    AST Transformer that converts relative imports to absolute imports.
    """
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        
        # Find the 'src' directory to establish the base package
        self.src_index = -1
        file_parts = self.file_path.parts
        for i, part in enumerate(file_parts):
            if part == 'src':
                self.src_index = i
                break
        
        if self.src_index == -1:
            raise ValueError(f"Cannot determine absolute path for {file_path}: no 'src' directory found")
    
    def visit_ImportFrom(self, node):
        """
        Convert relative imports to absolute imports.
        """
        if node.level > 0:  # This is a relative import
            try:
                # Get the module parts from the file path
                module_parts = self.file_path.parts[self.src_index + 1:]
                
                # Remove the filename and split into package parts
                if module_parts and module_parts[-1].endswith('.py'):
                    module_parts = module_parts[:-1]
                
                # Go up 'node.level' number of levels
                if node.level > len(module_parts):
                    # Invalid relative import, return as-is
                    return node
                
                base_parts = module_parts[:-(node.level)] if node.level > 0 else module_parts
                relative_module = node.module or ''
                
                # Combine with the relative module path
                if relative_module:
                    target_parts = base_parts + tuple(relative_module.split('.'))
                else:
                    target_parts = base_parts
                
                # Create the absolute module path
                absolute_module = '.'.join(target_parts)
                
                # Create a new ImportFrom node with the absolute module
                new_node = ast.ImportFrom(
                    module=absolute_module,
                    names=node.names,
                    level=0  # No longer a relative import
                )
                
                # Preserve line number and column offset
                new_node.lineno = node.lineno
                new_node.col_offset = node.col_offset
                
                return new_node
            except Exception:
                # If conversion fails, keep the original node
                return node
        
        # Not a relative import, return as-is
        return node


def convert_file_imports(file_path):
    """
    Convert relative imports in a file to absolute imports.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Parse the AST
        tree = ast.parse(original_content)
    except (SyntaxError, UnicodeDecodeError):
        print(f"Skipping {file_path} due to syntax error")
        return False
    
    # Transform the AST
    transformer = RelativeToAbsoluteImportTransformer(file_path)
    try:
        new_tree = transformer.visit(tree)
    except ValueError as e:
        print(f"Skipping {file_path} due to: {e}")
        return False
    
    # Check if any changes were made
    if ast.dump(tree) == ast.dump(new_tree):
        return False  # No changes made
    
    # Generate the new code
    try:
        new_content = ast.unparse(new_tree)  # Available in Python 3.9+
    except AttributeError:
        # For older Python versions, use a workaround
        import astor  # May need to install: pip install astor
        new_content = astor.to_source(new_tree)
    
    # Write the modified content back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Fixed imports in: {file_path}")
    return True


def standardize_imports_in_directory(directory):
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


def verify_conversion(directory):
    """
    Verify that the conversion worked properly.
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
            if 'from .' in content and ' import ' in content:
                # More precise check: find lines that start with relative imports
                lines = content.splitlines()
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if line.startswith('from .') and ' import ' in line:
                        files_with_relative_imports.append((file_path, line_num, line))
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    
    if files_with_relative_imports:
        print(f"\nFound {len(files_with_relative_imports)} relative imports in {len(set(f[0] for f in files_with_relative_imports))} files:")
        for file_path, line_num, line_content in files_with_relative_imports[:10]:
            print(f"  {file_path}:{line_num} - {line_content}")
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