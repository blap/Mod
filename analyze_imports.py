import os
import ast
from pathlib import Path

def find_python_files(root_dir):
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def analyze_relative_imports(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError):
        return []
    
    relative_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0:
            relative_imports.append({
                'line': node.lineno,
                'level': node.level,
                'module': node.module or '',
                'names': [alias.name + (' as ' + alias.asname if alias.asname else '') for alias in node.names]
            })
    
    return relative_imports

root_dir = r'C:\Users\Admin\Documents\GitHub\Mod\src\qwen3_vl'
python_files = find_python_files(root_dir)

all_relative_imports = []
for file_path in python_files:
    rel_imports = analyze_relative_imports(file_path)
    for imp in rel_imports:
        all_relative_imports.append((file_path, imp))

print(f'Found {len(all_relative_imports)} relative imports')
print('\nExamples of relative imports:')
for i, (file_path, imp) in enumerate(all_relative_imports[:20]):
    dots = '.' * imp['level']
    names = ', '.join(imp['names'])
    print(f'  {file_path}:{imp["line"]} - from {dots}{imp["module"]} import {names}')

print(f'\n  ... and {len(all_relative_imports) - 20} more')