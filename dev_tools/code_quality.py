"""
Code Quality and Linting Utilities for Qwen3-VL Model

This module provides comprehensive code quality and linting tools to maintain 
high standards in the Qwen3-VL model codebase.
"""

import ast
import astor
import os
import re
import json
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import tokenize
from io import StringIO
import importlib.util


@dataclass
class LintIssue:
    """Represents a code quality issue found by the linter"""
    file_path: str
    line_number: int
    column: int
    severity: str  # 'error', 'warning', 'info'
    code: str
    message: str
    source_line: str = ""


class CodeQualityChecker:
    """Main code quality checker class"""
    
    def __init__(self):
        self.issues = []
        self.metrics = {}
        self.config = {
            'max_line_length': 120,
            'min_function_length': 2,
            'max_function_length': 50,
            'max_class_length': 500,
            'max_cyclomatic_complexity': 10,
            'allowed_imports': [],
            'forbidden_imports': [],
            'naming_conventions': {
                'function': r'^[a-z_][a-z0-9_]*$',
                'variable': r'^[a-z_][a-z0-9_]*$',
                'class': r'^[A-Z][a-zA-Z0-9]*$',
                'constant': r'^[A-Z][A-Z0-9_]*$'
            }
        }
    
    def check_file(self, file_path: str) -> List[LintIssue]:
        """Check a single Python file for quality issues"""
        issues = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the file
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            issues.append(LintIssue(
                file_path=file_path,
                line_number=e.lineno or 0,
                column=e.offset or 0,
                severity='error',
                code='SYNTAX_ERROR',
                message=f"Syntax error: {e.msg}",
                source_line=content.split('\n')[e.lineno - 1] if e.lineno else ""
            ))
            return issues
        
        # Perform various checks
        issues.extend(self._check_line_length(file_path, content))
        issues.extend(self._check_naming_conventions(file_path, tree))
        issues.extend(self._check_function_lengths(file_path, tree))
        issues.extend(self._check_class_lengths(file_path, tree))
        issues.extend(self._check_cyclomatic_complexity(file_path, tree))
        issues.extend(self._check_imports(file_path, tree))
        issues.extend(self._check_docstrings(file_path, tree))
        issues.extend(self._check_unused_variables(file_path, tree))
        
        self.issues.extend(issues)
        return issues
    
    def check_directory(self, directory_path: str, recursive: bool = True) -> List[LintIssue]:
        """Check all Python files in a directory"""
        issues = []
        
        pattern = "**/*.py" if recursive else "*.py"
        py_files = Path(directory_path).glob(pattern)
        
        for file_path in py_files:
            file_issues = self.check_file(str(file_path))
            issues.extend(file_issues)
        
        return issues
    
    def _check_line_length(self, file_path: str, content: str) -> List[LintIssue]:
        """Check for lines exceeding maximum length"""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > self.config['max_line_length']:
                issues.append(LintIssue(
                    file_path=file_path,
                    line_number=i,
                    column=0,
                    severity='warning',
                    code='LINE_TOO_LONG',
                    message=f"Line too long ({len(line)} > {self.config['max_line_length']})",
                    source_line=line
                ))
        
        return issues
    
    def _check_naming_conventions(self, file_path: str, tree: ast.AST) -> List[LintIssue]:
        """Check naming conventions"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not re.match(self.config['naming_conventions']['function'], node.name):
                    issues.append(LintIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        severity='warning',
                        code='INVALID_FUNCTION_NAME',
                        message=f"Invalid function name '{node.name}'. Should match: {self.config['naming_conventions']['function']}",
                        source_line=self._get_source_line(file_path, node.lineno)
                    ))
            
            elif isinstance(node, ast.AsyncFunctionDef):
                if not re.match(self.config['naming_conventions']['function'], node.name):
                    issues.append(LintIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        severity='warning',
                        code='INVALID_FUNCTION_NAME',
                        message=f"Invalid function name '{node.name}'. Should match: {self.config['naming_conventions']['function']}",
                        source_line=self._get_source_line(file_path, node.lineno)
                    ))
            
            elif isinstance(node, ast.ClassDef):
                if not re.match(self.config['naming_conventions']['class'], node.name):
                    issues.append(LintIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        severity='warning',
                        code='INVALID_CLASS_NAME',
                        message=f"Invalid class name '{node.name}'. Should match: {self.config['naming_conventions']['class']}",
                        source_line=self._get_source_line(file_path, node.lineno)
                    ))
            
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper():  # Constant
                            if not re.match(self.config['naming_conventions']['constant'], target.id):
                                issues.append(LintIssue(
                                    file_path=file_path,
                                    line_number=target.lineno,
                                    column=getattr(target, 'col_offset', 0),
                                    severity='warning',
                                    code='INVALID_CONSTANT_NAME',
                                    message=f"Invalid constant name '{target.id}'. Should match: {self.config['naming_conventions']['constant']}",
                                    source_line=self._get_source_line(file_path, target.lineno)
                                ))
                        else:  # Variable
                            if not re.match(self.config['naming_conventions']['variable'], target.id):
                                issues.append(LintIssue(
                                    file_path=file_path,
                                    line_number=target.lineno,
                                    column=getattr(target, 'col_offset', 0),
                                    severity='warning',
                                    code='INVALID_VARIABLE_NAME',
                                    message=f"Invalid variable name '{target.id}'. Should match: {self.config['naming_conventions']['variable']}",
                                    source_line=self._get_source_line(file_path, target.lineno)
                                ))
        
        return issues
    
    def _check_function_lengths(self, file_path: str, tree: ast.AST) -> List[LintIssue]:
        """Check function lengths"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Count lines in function body
                body_lines = set()
                for item in node.body:
                    for line in range(item.lineno, getattr(item, 'end_lineno', item.lineno) + 1):
                        body_lines.add(line)
                
                length = len(body_lines)
                
                if length > self.config['max_function_length']:
                    issues.append(LintIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        severity='warning',
                        code='FUNCTION_TOO_LONG',
                        message=f"Function '{node.name}' is too long ({length} > {self.config['max_function_length']})",
                        source_line=self._get_source_line(file_path, node.lineno)
                    ))
        
        return issues
    
    def _check_class_lengths(self, file_path: str, tree: ast.AST) -> List[LintIssue]:
        """Check class lengths"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Count lines in class body
                body_lines = set()
                for item in node.body:
                    for line in range(item.lineno, getattr(item, 'end_lineno', item.lineno) + 1):
                        body_lines.add(line)
                
                length = len(body_lines)
                
                if length > self.config['max_class_length']:
                    issues.append(LintIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        severity='warning',
                        code='CLASS_TOO_LONG',
                        message=f"Class '{node.name}' is too long ({length} > {self.config['max_class_length']})",
                        source_line=self._get_source_line(file_path, node.lineno)
                    ))
        
        return issues
    
    def _check_cyclomatic_complexity(self, file_path: str, tree: ast.AST) -> List[LintIssue]:
        """Check cyclomatic complexity of functions"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)
                
                if complexity > self.config['max_cyclomatic_complexity']:
                    issues.append(LintIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        severity='warning',
                        code='HIGH_COMPLEXITY',
                        message=f"Function '{node.name}' has high cyclomatic complexity ({complexity} > {self.config['max_cyclomatic_complexity']})",
                        source_line=self._get_source_line(file_path, node.lineno)
                    ))
        
        return issues
    
    def _calculate_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for item in ast.walk(node):
            if isinstance(item, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(item, ast.BoolOp):  # and/or
                complexity += len(item.values) - 1
        
        return complexity
    
    def _check_imports(self, file_path: str, tree: ast.AST) -> List[LintIssue]:
        """Check imports against allowed/forbidden lists"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    if self.config['forbidden_imports'] and module_name in self.config['forbidden_imports']:
                        issues.append(LintIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=getattr(node, 'col_offset', 0),
                            severity='warning',
                            code='FORBIDDEN_IMPORT',
                            message=f"Forbidden import: {module_name}",
                            source_line=self._get_source_line(file_path, node.lineno)
                        ))
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if self.config['forbidden_imports'] and module_name in self.config['forbidden_imports']:
                    issues.append(LintIssue(
                        file_path=file_path,
                        line_number=node.lineno,
                        column=getattr(node, 'col_offset', 0),
                        severity='warning',
                        code='FORBIDDEN_IMPORT',
                        message=f"Forbidden import: {module_name}",
                        source_line=self._get_source_line(file_path, node.lineno)
                    ))
        
        return issues
    
    def _check_docstrings(self, file_path: str, tree: ast.AST) -> List[LintIssue]:
        """Check for missing docstrings"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    docstring = ast.get_docstring(node)
                    if not docstring:
                        issues.append(LintIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=getattr(node, 'col_offset', 0),
                            severity='info',
                            code='MISSING_DOCSTRING',
                            message=f"Missing docstring for function '{node.name}'",
                            source_line=self._get_source_line(file_path, node.lineno)
                        ))
                except TypeError:
                    # Some nodes can't have docstrings, skip them
                    pass

            elif isinstance(node, ast.ClassDef):
                try:
                    docstring = ast.get_docstring(node)
                    if not docstring:
                        issues.append(LintIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            column=getattr(node, 'col_offset', 0),
                            severity='info',
                            code='MISSING_DOCSTRING',
                            message=f"Missing docstring for class '{node.name}'",
                            source_line=self._get_source_line(file_path, node.lineno)
                        ))
                except TypeError:
                    # Some nodes can't have docstrings, skip them
                    pass
        
        return issues
    
    def _check_unused_variables(self, file_path: str, tree: ast.AST) -> List[LintIssue]:
        """Check for unused variables"""
        issues = []
        
        # This is a simplified check - a full implementation would need more sophisticated analysis
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith('_'):
                        # Variables starting with _ are often meant to be unused
                        continue
                    elif isinstance(target, ast.Name):
                        # In a real implementation, we would check if the variable is used
                        # For now, we'll just skip this check to avoid false positives
                        pass
        
        return issues
    
    def _get_source_line(self, file_path: str, line_number: int) -> str:
        """Get a specific line from a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if 1 <= line_number <= len(lines):
                    return lines[line_number - 1].rstrip()
        except:
            pass
        return ""
    
    def get_report(self) -> Dict[str, Any]:
        """Get a summary report of all issues found"""
        error_count = sum(1 for issue in self.issues if issue.severity == 'error')
        warning_count = sum(1 for issue in self.issues if issue.severity == 'warning')
        info_count = sum(1 for issue in self.issues if issue.severity == 'info')
        
        return {
            'total_issues': len(self.issues),
            'errors': error_count,
            'warnings': warning_count,
            'info': info_count,
            'files_affected': len(set(issue.file_path for issue in self.issues)),
            'issues_by_file': self._group_issues_by_file()
        }
    
    def _group_issues_by_file(self) -> Dict[str, List[LintIssue]]:
        """Group issues by file"""
        grouped = {}
        for issue in self.issues:
            if issue.file_path not in grouped:
                grouped[issue.file_path] = []
            grouped[issue.file_path].append(issue)
        return grouped
    
    def save_report(self, output_path: str):
        """Save the report to a JSON file"""
        report = self.get_report()
        report['issues'] = [
            {
                'file_path': issue.file_path,
                'line_number': issue.line_number,
                'column': issue.column,
                'severity': issue.severity,
                'code': issue.code,
                'message': issue.message,
                'source_line': issue.source_line
            }
            for issue in self.issues
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)


class PyLintRunner:
    """Runner for external pylint tool"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
    
    def run_pylint(self, paths: List[str], output_format: str = 'json') -> Dict[str, Any]:
        """Run pylint on specified paths"""
        cmd = ['pylint']
        
        if self.config_file:
            cmd.extend(['--rcfile', self.config_file])
        
        cmd.extend(['--output-format', output_format])
        cmd.extend(paths)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if output_format == 'json':
                # Parse JSON output
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return basic info
                    return {
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode
                    }
            else:
                return {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
        except FileNotFoundError:
            # Pylint not installed
            return {
                'error': 'pylint not found. Install with: pip install pylint',
                'returncode': -1
            }


class Flake8Runner:
    """Runner for external flake8 tool"""
    
    def run_flake8(self, paths: List[str]) -> Dict[str, Any]:
        """Run flake8 on specified paths"""
        cmd = ['flake8']
        cmd.extend(['--format', 'json'])
        cmd.extend(paths)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse JSON output
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    # If JSON parsing fails, return basic info
                    return {
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode
                    }
            else:
                # If no output but no error, return empty list
                if result.returncode == 0:
                    return []
                else:
                    return {
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'returncode': result.returncode
                    }
        except FileNotFoundError:
            # Flake8 not installed
            return {
                'error': 'flake8 not found. Install with: pip install flake8',
                'returncode': -1
            }


class BlackFormatter:
    """Runner for external black formatter"""
    
    def format_code(self, code: str, line_length: int = 88) -> str:
        """Format code with black"""
        try:
            import black
            mode = black.FileMode(line_length=line_length)
            return black.format_str(code, mode=mode)
        except ImportError:
            # Black not installed
            return code
    
    def format_file(self, file_path: str, line_length: int = 88) -> bool:
        """Format a file with black"""
        try:
            import black
            
            with open(file_path, 'r') as f:
                original_content = f.read()
            
            mode = black.FileMode(line_length=line_length)
            formatted_content = black.format_str(original_content, mode=mode)
            
            if original_content != formatted_content:
                with open(file_path, 'w') as f:
                    f.write(formatted_content)
                return True
            
            return False
        except ImportError:
            # Black not installed
            return False


class CodeMetrics:
    """Calculate various code metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_file_metrics(self, file_path: str) -> Dict[str, Any]:
        """Calculate metrics for a single file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        lines = content.split('\n')
        comment_lines = 0
        docstring_lines = 0
        code_lines = 0
        
        # Count different types of lines
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#'):
                comment_lines += 1
            elif stripped:
                code_lines += 1
        
        # Count docstrings
        for node in ast.walk(tree):
            try:
                docstring = ast.get_docstring(node)
                if docstring:
                    docstring_lines += len(docstring.split('\n'))
            except TypeError:
                # Some nodes can't have docstrings, skip them
                continue
        
        # Count functions, classes, etc.
        functions = 0
        classes = 0
        imports = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions += 1
            elif isinstance(node, ast.ClassDef):
                classes += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports += 1
        
        metrics = {
            'file_path': file_path,
            'total_lines': len(lines),
            'code_lines': code_lines,
            'comment_lines': comment_lines,
            'docstring_lines': docstring_lines,
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'complexity': self._calculate_file_complexity(tree)
        }
        
        self.metrics[file_path] = metrics
        return metrics
    
    def _calculate_file_complexity(self, tree: ast.AST) -> int:
        """Calculate overall complexity of a file"""
        complexity = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity += self._calculate_function_complexity(node)
        
        return complexity
    
    def _calculate_function_complexity(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
        """Calculate complexity of a function"""
        complexity = 1  # Base complexity
        
        for item in ast.walk(node):
            if isinstance(item, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(item, ast.BoolOp):  # and/or
                complexity += len(item.values) - 1
        
        return complexity
    
    def calculate_directory_metrics(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """Calculate metrics for all Python files in a directory"""
        pattern = "**/*.py" if recursive else "*.py"
        py_files = Path(directory_path).glob(pattern)
        
        all_metrics = []
        total_metrics = {
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'docstring_lines': 0,
            'functions': 0,
            'classes': 0,
            'imports': 0,
            'complexity': 0
        }
        
        for file_path in py_files:
            file_metrics = self.calculate_file_metrics(str(file_path))
            all_metrics.append(file_metrics)
            
            # Update totals
            for key in total_metrics:
                if key != 'file_path':
                    total_metrics[key] += file_metrics[key]
        
        return {
            'files': all_metrics,
            'totals': total_metrics,
            'average_complexity_per_function': total_metrics['complexity'] / max(total_metrics['functions'], 1),
            'comment_ratio': total_metrics['comment_lines'] / max(total_metrics['code_lines'], 1),
            'docstring_ratio': total_metrics['docstring_lines'] / max(total_metrics['code_lines'], 1)
        }


class QualityReportGenerator:
    """Generate quality reports"""
    
    def __init__(self):
        pass
    
    def generate_text_report(self, issues: List[LintIssue]) -> str:
        """Generate a text report from issues"""
        report_lines = ["Code Quality Report", "="*50, ""]
        
        # Group by severity
        errors = [issue for issue in issues if issue.severity == 'error']
        warnings = [issue for issue in issues if issue.severity == 'warning']
        info = [issue for issue in issues if issue.severity == 'info']
        
        report_lines.append(f"Total Issues: {len(issues)}")
        report_lines.append(f"Errors: {len(errors)}")
        report_lines.append(f"Warnings: {len(warnings)}")
        report_lines.append(f"Info: {len(info)}")
        report_lines.append("")
        
        if errors:
            report_lines.append("ERRORS:")
            for issue in errors:
                report_lines.append(f"  {issue.file_path}:{issue.line_number}:{issue.column} - {issue.code}: {issue.message}")
            report_lines.append("")
        
        if warnings:
            report_lines.append("WARNINGS:")
            for issue in warnings:
                report_lines.append(f"  {issue.file_path}:{issue.line_number}:{issue.column} - {issue.code}: {issue.message}")
            report_lines.append("")
        
        if info:
            report_lines.append("INFO:")
            for issue in info:
                report_lines.append(f"  {issue.file_path}:{issue.line_number}:{issue.column} - {issue.code}: {issue.message}")
            report_lines.append("")
        
        # Group by file
        files = {}
        for issue in issues:
            if issue.file_path not in files:
                files[issue.file_path] = []
            files[issue.file_path].append(issue)
        
        report_lines.append("ISSUES BY FILE:")
        for file_path, file_issues in files.items():
            report_lines.append(f"  {file_path}: {len(file_issues)} issues")
            for issue in file_issues[:5]:  # Show first 5 issues per file
                report_lines.append(f"    {issue.line_number}:{issue.column} - {issue.code}: {issue.message[:50]}...")
            if len(file_issues) > 5:
                report_lines.append(f"    ... and {len(file_issues) - 5} more issues")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def generate_html_report(self, issues: List[LintIssue], output_path: str):
        """Generate an HTML report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .issue {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .error {{ border-left: 5px solid #d9534f; }}
                .warning {{ border-left: 5px solid #f0ad4e; }}
                .info {{ border-left: 5px solid #5bc0de; }}
                .severity {{ font-weight: bold; margin-right: 10px; }}
                .file-path {{ color: #666; font-family: monospace; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Code Quality Report</h1>
                <p>Total Issues: {{total_issues}}</p>
                <p>Errors: {{error_count}}, Warnings: {{warning_count}}, Info: {{info_count}}</p>
            </div>

            {{issue_list}}
        </body>
        </html>
        """
        
        issue_html = ""
        for issue in issues:
            severity_class = issue.severity
            severity_display = issue.severity.upper()
            
            issue_html += f"""
            <div class="issue {severity_class}">
                <div>
                    <span class="severity">{severity_display}</span>
                    <span class="file-path">{issue.file_path}:{issue.line_number}</span>
                </div>
                <div><strong>{issue.code}</strong>: {issue.message}</div>
                <div><em>{issue.source_line}</em></div>
            </div>
            """
        
        html_content = html_template.format(
            total_issues=len(issues),
            error_count=len([i for i in issues if i.severity == 'error']),
            warning_count=len([i for i in issues if i.severity == 'warning']),
            info_count=len([i for i in issues if i.severity == 'info']),
            issue_list=issue_html
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)


# Global instances
quality_checker = CodeQualityChecker()
pylint_runner = PyLintRunner()
flake8_runner = Flake8Runner()
black_formatter = BlackFormatter()
code_metrics = CodeMetrics()
report_generator = QualityReportGenerator()


def run_quality_checks(directory_path: str) -> Dict[str, Any]:
    """Run all quality checks on a directory"""
    print(f"Running quality checks on: {directory_path}")
    
    # Run internal checks
    issues = quality_checker.check_directory(directory_path)
    
    # Generate report
    report = quality_checker.get_report()
    
    # Calculate metrics
    metrics = code_metrics.calculate_directory_metrics(directory_path)
    
    # Combine results
    results = {
        'issues': issues,
        'report': report,
        'metrics': metrics
    }
    
    print(f"Found {len(issues)} issues in {report['files_affected']} files")
    print(f"Total errors: {report['errors']}, warnings: {report['warnings']}, info: {report['info']}")
    
    return results


def generate_quality_report(results: Dict[str, Any], output_dir: str = "quality_report"):
    """Generate a comprehensive quality report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate text report
    text_report = report_generator.generate_text_report(results['issues'])
    with open(os.path.join(output_dir, "quality_report.txt"), "w") as f:
        f.write(text_report)
    
    # Generate HTML report
    report_generator.generate_html_report(results['issues'], os.path.join(output_dir, "quality_report.html"))
    
    # Save detailed JSON report
    quality_checker.save_report(os.path.join(output_dir, "detailed_report.json"))
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results['metrics'], f, indent=2)
    
    print(f"Quality reports generated in: {output_dir}")


def example_quality_checking():
    """Example of quality checking"""
    print("=== Code Quality Checking Example ===")
    
    # Create a temporary file with some code issues
    test_code = '''
# This is a test file for quality checking

import os
import sys
import json

class BadClassName:  # Should be PascalCase
    def __init__(self):
        self.bad_attr_name = 42  # Should be snake_case
        very_long_variable_name_that_exceeds_reasonable_limits_and_should_be_shortened = "hello"
    
    def badFunctionName(self):  # Should be snake_case
        """This function does something."""
        x = 1
        y = 2
        if x > 0:
            if y > 0:
                if x + y > 1:
                    if x - y < 10:
                        if x * y < 100:
                            result = "complex logic"
        return result

def function_with_too_many_params(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p):
    """Function with too many parameters."""
    return a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p

very_long_line = "This is a very long line that exceeds the maximum allowed length for readability purposes and should be shortened to improve code quality and maintainability"

CONSTANT_NAME_SHOULD_BE_SNAKE = 100  # Should be UPPER_SNAKE_CASE

# This line is OK: short and readable
short_line = "OK"

class ReasonableClass:
    """A class with reasonable length and complexity."""
    
    def method1(self):
        return 1
    
    def method2(self):
        return 2
'''
    
    # Write test code to a temporary file
    with open("temp_quality_test.py", "w") as f:
        f.write(test_code)
    
    # Run quality checks
    results = run_quality_checks(".")
    
    # Generate reports
    generate_quality_report(results)
    
    # Clean up
    if os.path.exists("temp_quality_test.py"):
        os.remove("temp_quality_test.py")
    if os.path.exists("quality_report"):
        import shutil
        shutil.rmtree("quality_report")


def example_external_tools():
    """Example of using external quality tools"""
    print("\n=== External Quality Tools Example ===")
    
    # Check if external tools are available
    try:
        import pylint
        print("Pylint is available")
        
        # Run pylint on the current file
        pylint_results = pylint_runner.run_pylint(["temp_quality_test.py"], output_format='text')
        print("Pylint results:", pylint_results.get('stdout', 'No output'))
    except ImportError:
        print("Pylint not available. Install with: pip install pylint")
    
    try:
        import flake8
        print("Flake8 is available")
        
        # Run flake8 on the current file
        flake8_results = flake8_runner.run_flake8(["temp_quality_test.py"])
        print("Flake8 results:", flake8_results)
    except ImportError:
        print("Flake8 not available. Install with: pip install flake8")
    
    try:
        import black
        print("Black is available")
        
        # Format the test file
        formatted = black_formatter.format_file("temp_quality_test.py")
        print(f"Black formatting applied: {formatted}")
    except ImportError:
        print("Black not available. Install with: pip install black")


if __name__ == "__main__":
    example_quality_checking()
    example_external_tools()