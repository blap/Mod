"""
Test Analysis Script for Inference-PIO Project

This script analyzes all test files in the project to identify:
1. Current test structure and naming conventions
2. Test categories (unit, integration, performance)
3. Potential issues or inconsistencies
4. Coverage gaps
"""

import os
import sys
import ast
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple


def find_test_files(root_dir: str) -> List[str]:
    """Find all test files in the project."""
    test_files = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith('test_') or '_test' in file:
                if file.endswith('.py'):
                    test_files.append(os.path.join(root, file))
    
    return test_files


def analyze_test_file(file_path: str) -> Dict[str, Any]:
    """Analyze a single test file to extract metadata."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Extract test functions
        test_functions = []
        import_statements = []
        class_definitions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    test_functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node)
                    })
            elif isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
                import_statements.append(ast.unparse(node))
            elif isinstance(node, ast.ClassDef):
                class_definitions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'bases': [ast.unparse(base) for base in node.bases]
                })
        
        # Determine test category based on path
        path_parts = file_path.split(os.sep)
        category = 'unknown'
        if 'unit' in path_parts:
            category = 'unit'
        elif 'integration' in path_parts:
            category = 'integration'
        elif 'performance' in path_parts:
            category = 'performance'
        
        # Check for common test frameworks
        uses_pytest = 'pytest' in content
        uses_unittest = 'unittest' in content
        uses_custom_framework = 'test_utils' in content
        
        return {
            'file_path': file_path,
            'category': category,
            'test_functions': test_functions,
            'classes': class_definitions,
            'imports': import_statements,
            'uses_pytest': uses_pytest,
            'uses_unittest': uses_unittest,
            'uses_custom_framework': uses_custom_framework,
            'total_test_functions': len(test_functions),
            'has_docstrings': any(tf['docstring'] for tf in test_functions),
            'size': len(content)
        }
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'category': 'unknown',
            'test_functions': [],
            'classes': [],
            'imports': [],
            'uses_pytest': False,
            'uses_unittest': False,
            'uses_custom_framework': False,
            'total_test_functions': 0,
            'has_docstrings': False,
            'size': 0
        }


def analyze_project_tests(root_dir: str) -> Dict[str, Any]:
    """Analyze all tests in the project."""
    print("Analyzing test files...")
    
    test_files = find_test_files(root_dir)
    print(f"Found {len(test_files)} test files")
    
    all_analysis = []
    category_counts = Counter()
    framework_usage = Counter()
    total_functions = 0
    
    for file_path in test_files:
        analysis = analyze_test_file(file_path)
        all_analysis.append(analysis)
        
        category_counts[analysis['category']] += 1
        if analysis.get('uses_pytest'):
            framework_usage['pytest'] += 1
        if analysis.get('uses_unittest'):
            framework_usage['unittest'] += 1
        if analysis.get('uses_custom_framework'):
            framework_usage['custom'] += 1
            
        total_functions += analysis.get('total_test_functions', 0)
    
    # Identify potential issues
    issues = []
    for analysis in all_analysis:
        if analysis.get('error'):
            issues.append(f"Error parsing {analysis['file_path']}: {analysis['error']}")
        elif analysis.get('total_test_functions', 0) == 0:
            issues.append(f"No test functions found in {analysis['file_path']}")
    
    return {
        'summary': {
            'total_files': len(test_files),
            'total_functions': total_functions,
            'categories': dict(category_counts),
            'framework_usage': dict(framework_usage),
            'issues_found': len(issues)
        },
        'files': all_analysis,
        'issues': issues
    }


def print_analysis_report(analysis: Dict[str, Any]):
    """Print a human-readable analysis report."""
    print("\n" + "="*60)
    print("TEST ANALYSIS REPORT")
    print("="*60)
    
    summary = analysis['summary']
    print(f"Total test files: {summary['total_files']}")
    print(f"Total test functions: {summary['total_functions']}")
    print(f"Issues found: {summary['issues_found']}")
    
    print(f"\nTest Categories:")
    for category, count in summary['categories'].items():
        print(f"  {category}: {count}")
    
    print(f"\nTesting Frameworks Used:")
    for framework, count in summary['framework_usage'].items():
        print(f"  {framework}: {count}")
    
    if analysis['issues']:
        print(f"\nIssues Found:")
        for issue in analysis['issues'][:10]:  # Show first 10 issues
            print(f"  - {issue}")
        if len(analysis['issues']) > 10:
            print(f"  ... and {len(analysis['issues']) - 10} more issues")
    
    # Show largest test files
    large_files = sorted(analysis['files'], 
                        key=lambda x: x.get('size', 0), 
                        reverse=True)[:5]
    print(f"\nLargest test files:")
    for file_info in large_files:
        if not file_info.get('error'):
            print(f"  {file_info['file_path']}: {file_info['size']} bytes "
                  f"({file_info['total_test_functions']} functions)")


def run_test_validation_script():
    """Create and run a script to validate all tests."""
    validation_script = '''
import os
import sys
import subprocess
import tempfile
from pathlib import Path

def validate_test_file(file_path):
    """Validate a single test file by importing it."""
    try:
        # Create a temporary Python script that imports the test file
        temp_script = f"""
import sys
sys.path.insert(0, r'{os.path.dirname(file_path)}')
sys.path.insert(0, r'{str(Path(file_path).parent.parent.parent.absolute())}')

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("{Path(file_path).stem}", r"{file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
"""
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(temp_script)
            temp_file = f.name
        
        # Run the temporary script
        result = subprocess.run([sys.executable, temp_file], 
                              capture_output=True, text=True, timeout=30)
        
        # Clean up
        os.unlink(temp_file)
        
        output = result.stdout.strip()
        if output == "SUCCESS":
            return True, ""
        else:
            error_msg = output.replace("ERROR: ", "")
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)

def validate_all_tests():
    """Validate all test files in the project."""
    # Find all test files
    test_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.startswith('test_') or '_test' in file:
                if file.endswith('.py'):
                    test_files.append(os.path.join(root, file))
    
    print(f"Validating {len(test_files)} test files...")
    
    results = []
    for i, file_path in enumerate(test_files):
        print(f"[{i+1}/{len(test_files)}] Validating: {file_path}")
        success, error = validate_test_file(file_path)
        results.append((file_path, success, error))
        
        if not success:
            print(f"  ❌ FAILED: {error}")
        else:
            print(f"  ✅ OK")
    
    # Summary
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful
    
    print(f"\nValidation Summary:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(results)}")
    
    if failed > 0:
        print(f"\nFailed files:")
        for file_path, success, error in results:
            if not success:
                print(f"  - {file_path}: {error}")
    
    return results

if __name__ == "__main__":
    validate_all_tests()
'''
    
    # Write the validation script
    with open('validate_tests.py', 'w', encoding='utf-8') as f:
        f.write(validation_script)
    
    print("Created validation script: validate_tests.py")
    print("Run 'python validate_tests.py' to validate all tests.")


def main():
    """Main function to run the analysis."""
    project_root = os.getcwd()
    
    # Analyze the project tests
    analysis = analyze_project_tests(project_root)
    
    # Print the report
    print_analysis_report(analysis)
    
    # Create validation script
    run_test_validation_script()
    
    # Save detailed analysis to JSON
    with open('test_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\nDetailed analysis saved to test_analysis.json")


if __name__ == "__main__":
    main()