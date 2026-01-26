"""
Comprehensive Test Coverage Report Generator for Inference-PIO

This script generates a comprehensive report of test coverage across the Inference-PIO project,
analyzing unit, integration, and performance tests to ensure adequate coverage.
"""

import os
import sys
import ast
import json
import subprocess
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple
import importlib.util
import traceback


def find_source_files(root_dir: str) -> List[str]:
    """Find all source files in the project."""
    source_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # Skip test directories when looking for source files
        dirs[:] = [d for d in dirs if not d.startswith('test') and d != 'tests']
        
        for file in files:
            if file.endswith('.py') and not file.startswith('test_') and '_test' not in file:
                if 'test' not in root and 'tests' not in root:
                    source_files.append(os.path.join(root, file))
    
    return source_files


def find_test_files(root_dir: str) -> List[str]:
    """Find all test files in the project."""
    test_files = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.startswith('test_') or '_test' in file:
                if file.endswith('.py'):
                    test_files.append(os.path.join(root, file))
    
    return test_files


def analyze_source_file(file_path: str) -> Dict[str, Any]:
    """Analyze a source file to extract classes, functions, and methods."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append({
                            'name': item.name,
                            'line': item.lineno,
                            'is_public': not item.name.startswith('_') or item.name.startswith('__')
                        })
                
                classes.append({
                    'name': node.name,
                    'line': node.lineno,
                    'methods': methods,
                    'public_methods': [m for m in methods if m['is_public']]
                })
            
            elif isinstance(node, ast.FunctionDef) and not any(isinstance(n, ast.ClassDef) for n in ast.walk(node)):
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'is_public': not node.name.startswith('_') or node.name.startswith('__')
                })
        
        return {
            'file_path': file_path,
            'classes': classes,
            'functions': functions,
            'public_functions': [f for f in functions if f['is_public']],
            'total_classes': len(classes),
            'total_functions': len(functions),
            'public_classes': [c for c in classes if c['name'][0].isupper()],
            'public_functions_count': len([f for f in functions if f['is_public']])
        }
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'classes': [],
            'functions': [],
            'public_functions': [],
            'total_classes': 0,
            'total_functions': 0,
            'public_classes': [],
            'public_functions_count': 0
        }


def analyze_test_file(file_path: str) -> Dict[str, Any]:
    """Analyze a test file to extract test functions."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        test_functions = []
        imported_modules = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    test_functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node)
                    })
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.append(alias.name)
        
        return {
            'file_path': file_path,
            'test_functions': test_functions,
            'imported_modules': imported_modules,
            'total_test_functions': len(test_functions)
        }
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'test_functions': [],
            'imported_modules': [],
            'total_test_functions': 0
        }


def map_tests_to_sources(test_files: List[str], source_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map test files to source files based on imports and naming conventions."""
    test_mappings = defaultdict(list)
    source_to_tests = defaultdict(list)
    
    # Create a mapping from source file paths to their analysis
    source_map = {sa['file_path']: sa for sa in source_analysis}
    
    for test_file in test_files:
        test_analysis = analyze_test_file(test_file)
        
        # Look for potential source mappings based on imports
        potential_sources = set()
        for imp in test_analysis['imported_modules']:
            # Convert import path to file path
            imp_path = imp.replace('.', os.sep) + '.py'
            
            # Look for matching source files
            for source_path in source_map.keys():
                if imp_path in source_path or os.path.basename(imp_path).replace('.py', '') in source_path:
                    potential_sources.add(source_path)
        
        # Also try to match based on naming convention (test_filename -> filename)
        test_basename = os.path.basename(test_file).replace('test_', '').replace('_test', '').replace('.py', '')
        for source_path in source_map.keys():
            source_basename = os.path.basename(source_path).replace('.py', '')
            if test_basename in source_basename or source_basename in test_basename:
                potential_sources.add(source_path)
        
        # Associate test with potential sources
        for source_path in potential_sources:
            if source_path in source_map:
                test_mappings[source_path].append(test_file)
                source_to_tests[test_file].append(source_path)
    
    return {
        'test_mappings': dict(test_mappings),
        'source_to_tests': dict(source_to_tests)
    }


def calculate_coverage(source_analysis: List[Dict[str, Any]], test_mappings: Dict[str, List[str]]) -> Dict[str, Any]:
    """Calculate test coverage statistics."""
    total_public_elements = 0
    covered_elements = 0
    coverage_details = []
    
    for source in source_analysis:
        source_path = source['file_path']
        
        # Count public elements (classes and functions)
        public_classes = len(source['public_classes'])
        public_functions = source['public_functions_count']
        total_source_elements = public_classes + public_functions
        
        total_public_elements += total_source_elements
        
        # Check if this source has associated tests
        test_files = test_mappings.get(source_path, [])
        has_tests = len(test_files) > 0
        
        if has_tests:
            covered_elements += total_source_elements
        
        coverage_details.append({
            'source_file': source_path,
            'total_public_elements': total_source_elements,
            'has_tests': has_tests,
            'test_files': test_files
        })
    
    overall_coverage = (covered_elements / total_public_elements * 100) if total_public_elements > 0 else 0
    
    return {
        'total_public_elements': total_public_elements,
        'covered_elements': covered_elements,
        'overall_coverage_percent': overall_coverage,
        'coverage_details': coverage_details,
        'untested_files': [detail for detail in coverage_details if not detail['has_tests']]
    }


def generate_coverage_report():
    """Generate a comprehensive test coverage report."""
    project_root = os.getcwd()
    
    print("Analyzing source files...")
    source_files = find_source_files(project_root)
    print(f"Found {len(source_files)} source files")
    
    print("Analyzing test files...")
    test_files = find_test_files(project_root)
    print(f"Found {len(test_files)} test files")
    
    print("Analyzing source code structure...")
    source_analysis = []
    for source_file in source_files:
        analysis = analyze_source_file(source_file)
        source_analysis.append(analysis)
    
    print("Mapping tests to sources...")
    mapping_result = map_tests_to_sources(test_files, source_analysis)
    
    print("Calculating coverage...")
    coverage_result = calculate_coverage(source_analysis, mapping_result['test_mappings'])
    
    # Generate report
    report = {
        'summary': {
            'total_source_files': len(source_files),
            'total_test_files': len(test_files),
            'total_public_elements': coverage_result['total_public_elements'],
            'covered_elements': coverage_result['covered_elements'],
            'overall_coverage_percent': coverage_result['overall_coverage_percent'],
            'tested_files_count': len(source_files) - len(coverage_result['untested_files']),
            'untested_files_count': len(coverage_result['untested_files'])
        },
        'coverage_details': coverage_result['coverage_details'],
        'untested_files': coverage_result['untested_files'],
        'test_distribution': {
            'unit_tests': len([tf for tf in test_files if 'unit' in tf]),
            'integration_tests': len([tf for tf in test_files if 'integration' in tf]),
            'performance_tests': len([tf for tf in test_files if 'performance' in tf])
        }
    }
    
    return report


def print_coverage_report(report: Dict[str, Any]):
    """Print a human-readable coverage report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST COVERAGE REPORT")
    print("="*80)
    
    summary = report['summary']
    print(f"Total Source Files: {summary['total_source_files']}")
    print(f"Total Test Files: {summary['total_test_files']}")
    print(f"Total Public Elements: {summary['total_public_elements']}")
    print(f"Covered Elements: {summary['covered_elements']}")
    print(f"Overall Coverage: {summary['overall_coverage_percent']:.2f}%")
    print(f"Files with Tests: {summary['tested_files_count']}")
    print(f"Untested Files: {summary['untested_files_count']}")
    
    print(f"\nTest Distribution:")
    print(f"  Unit Tests: {report['test_distribution']['unit_tests']}")
    print(f"  Integration Tests: {report['test_distribution']['integration_tests']}")
    print(f"  Performance Tests: {report['test_distribution']['performance_tests']}")
    
    if report['untested_files']:
        print(f"\nUntested Files:")
        for untested in report['untested_files'][:10]:  # Show first 10
            print(f"  - {untested['source_file']} ({untested['total_public_elements']} public elements)")
        if len(report['untested_files']) > 10:
            print(f"  ... and {len(report['untested_files']) - 10} more")
    
    print(f"\nCoverage by File:")
    covered_files = [detail for detail in report['coverage_details'] if detail['has_tests']]
    for detail in covered_files[:10]:  # Show first 10
        test_count = len(detail['test_files'])
        print(f"  {detail['source_file']}: {detail['total_public_elements']} elements, {test_count} test files")
    if len(covered_files) > 10:
        print(f"  ... and {len(covered_files) - 10} more")
    
    print("="*80)


def run_pytest_coverage():
    """Attempt to run pytest with coverage if available."""
    try:
        # Check if pytest-cov is available
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'show', 'pytest-cov'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("\nNote: pytest-cov not found. Install with: pip install pytest-cov")
            return None
        
        # Run pytest with coverage
        print("\nRunning pytest with coverage...")
        result = subprocess.run([
            sys.executable, '-m', 'pytest', '--cov=src', '--cov-report=html', 
            '--cov-report=term-missing', '--cov-fail-under=0', 'tests/'
        ], capture_output=True, text=True)
        
        print("Pytest coverage output:")
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        return result.returncode == 0
    except Exception as e:
        print(f"Could not run pytest coverage: {e}")
        return None


def main():
    """Main function to generate and display the coverage report."""
    print("Generating comprehensive test coverage report...")
    
    # Generate structural coverage report
    report = generate_coverage_report()
    print_coverage_report(report)
    
    # Try to run actual coverage measurement
    print("\nAttempting to run actual coverage measurement...")
    pytest_success = run_pytest_coverage()
    
    if pytest_success is not None:
        print(f"Pytest coverage {'succeeded' if pytest_success else 'failed'}")
    else:
        print("Skipping actual coverage measurement due to missing dependencies")
    
    # Save detailed report to JSON
    with open('comprehensive_test_coverage.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to comprehensive_test_coverage.json")
    
    # Provide recommendations
    print(f"\nRECOMMENDATIONS:")
    if report['summary']['overall_coverage_percent'] < 80:
        print("- Overall coverage is below 80%, consider adding more tests")
    if report['summary']['untested_files_count'] > 0:
        print(f"- {report['summary']['untested_files_count']} files have no tests, prioritize these")
    if report['test_distribution']['unit_tests'] < report['test_distribution']['integration_tests']:
        print("- Consider adding more unit tests relative to integration tests for better granularity")


if __name__ == "__main__":
    main()