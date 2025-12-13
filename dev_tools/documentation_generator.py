"""
Documentation Generation Tools for Qwen3-VL Model

This module provides tools to automatically generate documentation for the Qwen3-VL model,
including API documentation, architecture diagrams, and usage guides.
"""

import os
import ast
import inspect
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import markdown
from jinja2 import Template, Environment, FileSystemLoader


@dataclass
class FunctionDoc:
    """Documentation for a function"""
    name: str
    signature: str
    docstring: str
    parameters: List[Dict[str, str]]
    return_type: str
    examples: List[str]


@dataclass
class ClassDoc:
    """Documentation for a class"""
    name: str
    docstring: str
    methods: List[FunctionDoc]
    attributes: List[Dict[str, str]]
    inheritance: List[str]


@dataclass
class ModuleDoc:
    """Documentation for a module"""
    name: str
    docstring: str
    classes: List[ClassDoc]
    functions: List[FunctionDoc]
    imports: List[str]


class DocParser:
    """Parse Python code to extract documentation"""
    
    def __init__(self):
        self.modules = {}
    
    def parse_file(self, file_path: str) -> ModuleDoc:
        """Parse a Python file and extract documentation"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}")
        
        module_name = Path(file_path).stem
        module_docstring = ast.get_docstring(tree) or ""
        
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):  # Skip private functions
                    functions.append(self._parse_function(node))
            elif isinstance(node, ast.AsyncFunctionDef):
                if not node.name.startswith('_'):  # Skip private functions
                    functions.append(self._parse_function(node))
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith('_'):  # Skip private classes
                    classes.append(self._parse_class(node))
        
        module_doc = ModuleDoc(
            name=module_name,
            docstring=module_docstring,
            classes=classes,
            functions=functions,
            imports=imports
        )
        
        self.modules[module_name] = module_doc
        return module_doc
    
    def _parse_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> FunctionDoc:
        """Parse a function definition"""
        # Get signature
        args = []
        defaults = [ast.unparse(d) for d in node.args.defaults] if node.args.defaults else []
        
        # Handle regular arguments
        for i, arg in enumerate(node.args.args):
            default_idx = len(node.args.args) - len(defaults) + i
            default = defaults[i - (len(node.args.args) - len(defaults))] if default_idx >= 0 and default_idx < len(defaults) else None
            args.append({
                'name': arg.arg,
                'type': self._get_annotation_str(arg.annotation),
                'default': default
            })
        
        # Handle *args
        if node.args.vararg:
            args.append({
                'name': f"*{node.args.vararg.arg}",
                'type': self._get_annotation_str(node.args.vararg.annotation),
                'default': None
            })
        
        # Handle keyword-only arguments
        for i, arg in enumerate(node.args.kwonlyargs):
            default = ast.unparse(node.args.kw_defaults[i]) if i < len(node.args.kw_defaults) and node.args.kw_defaults[i] else None
            args.append({
                'name': arg.arg,
                'type': self._get_annotation_str(arg.annotation),
                'default': default
            })
        
        # Handle **kwargs
        if node.args.kwarg:
            args.append({
                'name': f"**{node.args.kwarg.arg}",
                'type': self._get_annotation_str(node.args.kwarg.annotation),
                'default': None
            })
        
        signature_parts = [f"{arg['name']}" + (f": {arg['type']}" if arg['type'] else "") + 
                          (f" = {arg['default']}" if arg['default'] is not None else "") for arg in args]
        signature = f"{node.name}({', '.join(signature_parts)})"
        
        if node.returns:
            signature += f" -> {self._get_annotation_str(node.returns)}"
        
        return FunctionDoc(
            name=node.name,
            signature=signature,
            docstring=ast.get_docstring(node) or "",
            parameters=args,
            return_type=self._get_annotation_str(node.returns) if node.returns else "",
            examples=[]
        )
    
    def _parse_class(self, node: ast.ClassDef) -> ClassDoc:
        """Parse a class definition"""
        methods = []
        attributes = []
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not item.name.startswith('_') or item.name in ['__init__', '__call__']:
                    methods.append(self._parse_function(item))
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append({
                            'name': target.id,
                            'type': self._get_annotation_str(item.value),
                            'value': ast.unparse(item.value)
                        })
        
        return ClassDoc(
            name=node.name,
            docstring=ast.get_docstring(node) or "",
            methods=methods,
            attributes=attributes,
            inheritance=[base.id if isinstance(base, ast.Name) else ast.unparse(base) for base in node.bases]
        )
    
    def _get_annotation_str(self, annotation) -> str:
        """Convert AST annotation to string"""
        if annotation is None:
            return ""
        return ast.unparse(annotation)


class ModelDocGenerator:
    """Generate documentation for PyTorch models"""
    
    def __init__(self):
        self.model_docs = {}
    
    def generate_model_doc(self, model: nn.Module, model_name: str = "Model") -> Dict[str, Any]:
        """Generate documentation for a PyTorch model"""
        doc = {
            'name': model_name,
            'class': model.__class__.__name__,
            'docstring': model.__doc__ or "",
            'layers': [],
            'parameters': {
                'total': sum(p.numel() for p in model.parameters()),
                'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'non_trainable': sum(p.numel() for p in model.parameters() if not p.requires_grad)
            },
            'architecture': self._analyze_architecture(model)
        }
        
        # Document each layer/module
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                layer_info = {
                    'name': name,
                    'type': module.__class__.__name__,
                    'parameters': sum(p.numel() for p in module.parameters(recurse=False)),
                    'trainable': any(p.requires_grad for p in module.parameters(recurse=False)),
                    'config': self._get_module_config(module)
                }
                doc['layers'].append(layer_info)
        
        self.model_docs[model_name] = doc
        return doc
    
    def _analyze_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model architecture"""
        architecture = {
            'layer_types': {},
            'connections': [],
            'input_shape': None,
            'output_shape': None
        }
        
        # Count layer types
        for name, module in model.named_modules():
            module_type = module.__class__.__name__
            architecture['layer_types'][module_type] = architecture['layer_types'].get(module_type, 0) + 1
        
        return architecture
    
    def _get_module_config(self, module: nn.Module) -> Dict[str, Any]:
        """Get configuration of a module"""
        config = {}
        
        # Get all attributes that are likely configuration parameters
        for attr_name in dir(module):
            if not attr_name.startswith('_'):
                attr = getattr(module, attr_name)
                if isinstance(attr, (int, float, str, bool, tuple, list)) and not callable(attr):
                    config[attr_name] = attr
        
        return config
    
    def generate_model_api_doc(self, model: nn.Module, model_name: str = "Model") -> str:
        """Generate API documentation for a model in markdown format"""
        model_doc = self.generate_model_doc(model, model_name)
        
        md_content = f"# {model_doc['name']} API Documentation\n\n"
        md_content += f"**Class**: {model_doc['class']}\n\n"
        
        if model_doc['docstring']:
            md_content += f"## Description\n{model_doc['docstring']}\n\n"
        
        md_content += f"## Parameters\n"
        md_content += f"- Total Parameters: {model_doc['parameters']['total']:,}\n"
        md_content += f"- Trainable Parameters: {model_doc['parameters']['trainable']:,}\n"
        md_content += f"- Non-trainable Parameters: {model_doc['parameters']['non_trainable']:,}\n\n"
        
        md_content += f"## Architecture\n"
        md_content += f"Layer Types:\n"
        for layer_type, count in model_doc['architecture']['layer_types'].items():
            md_content += f"- {layer_type}: {count}\n"
        md_content += "\n"
        
        md_content += f"## Layers\n"
        md_content += "| Name | Type | Parameters | Trainable |\n"
        md_content += "|------|------|------------|-----------|\n"
        
        for layer in model_doc['layers']:
            trainable_str = "YES" if layer['trainable'] else "NO"
            md_content += f"| {layer['name']} | {layer['type']} | {layer['parameters']:,} | {trainable_str} |\n"
        
        return md_content


class DocumentationGenerator:
    """Main documentation generator"""
    
    def __init__(self):
        self.doc_parser = DocParser()
        self.model_doc_generator = ModelDocGenerator()
        self.output_dir = "docs"
    
    def generate_code_docs(self, source_dir: str, output_dir: str = None):
        """Generate documentation for source code"""
        if output_dir:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Find all Python files
        py_files = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.join(root, file))
        
        # Parse each file
        modules = {}
        for file_path in py_files:
            try:
                module_doc = self.doc_parser.parse_file(file_path)
                modules[module_doc.name] = module_doc
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")
        
        # Generate documentation
        self._generate_html_docs(modules)
        self._generate_markdown_docs(modules)
    
    def _generate_html_docs(self, modules: Dict[str, ModuleDoc]):
        """Generate HTML documentation"""
        html_dir = os.path.join(self.output_dir, "html")
        os.makedirs(html_dir, exist_ok=True)
        
        # Create index page
        index_content = self._create_index_page(modules)
        with open(os.path.join(html_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(index_content)
        
        # Create individual module pages
        for module_name, module_doc in modules.items():
            module_content = self._create_module_page(module_name, module_doc)
            with open(os.path.join(html_dir, f"{module_name}.html"), "w", encoding="utf-8") as f:
                f.write(module_content)
    
    def _generate_markdown_docs(self, modules: Dict[str, ModuleDoc]):
        """Generate Markdown documentation"""
        md_dir = os.path.join(self.output_dir, "markdown")
        os.makedirs(md_dir, exist_ok=True)
        
        # Create main README
        readme_content = self._create_readme(modules)
        with open(os.path.join(md_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # Create individual module files
        for module_name, module_doc in modules.items():
            module_content = self._create_module_md(module_name, module_doc)
            with open(os.path.join(md_dir, f"{module_name}.md"), "w", encoding="utf-8") as f:
                f.write(module_content)
    
    def _create_index_page(self, modules: Dict[str, ModuleDoc]) -> str:
        """Create main index HTML page"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Qwen3-VL Documentation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2c3e50; }
                .module-list { list-style-type: none; padding: 0; }
                .module-item { margin: 10px 0; padding: 10px; border: 1px solid #ddd; }
                .module-link { font-size: 18px; font-weight: bold; text-decoration: none; color: #3498db; }
                .module-link:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>Qwen3-VL Model Documentation</h1>
            <p>Welcome to the documentation for the Qwen3-VL model and its development tools.</p>
            
            <h2>Modules</h2>
            <ul class="module-list">
            {% for module_name, module_doc in modules.items() %}
                <li class="module-item">
                    <a href="{{ module_name }}.html" class="module-link">{{ module_name }}</a>
                    <p>{{ module_doc.docstring[:100] + '...' if module_doc.docstring else 'No description' }}</p>
                </li>
            {% endfor %}
            </ul>
        </body>
        </html>
        """
        
        template = Template(html_template)
        return template.render(modules=modules)
    
    def _create_module_page(self, module_name: str, module_doc: ModuleDoc) -> str:
        """Create HTML page for a module"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ module_doc.name }} - Qwen3-VL Documentation</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2, h3 { color: #2c3e50; }
                .section { margin: 20px 0; }
                .function, .class { border: 1px solid #ddd; padding: 15px; margin: 10px 0; }
                .signature { font-family: monospace; background: #f8f9fa; padding: 5px; }
                .docstring { margin: 10px 0; }
                .params { margin: 10px 0; }
                .param { margin: 5px 0; }
                .code { background: #f4f4f4; padding: 10px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>Module: {{ module_doc.name }}</h1>
            
            {% if module_doc.docstring %}
            <div class="section">
                <h2>Description</h2>
                <div class="docstring">{{ module_doc.docstring }}</div>
            </div>
            {% endif %}
            
            {% if module_doc.imports %}
            <div class="section">
                <h2>Imports</h2>
                <ul>
                {% for imp in module_doc.imports %}
                    <li>{{ imp }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            {% if module_doc.classes %}
            <div class="section">
                <h2>Classes</h2>
                {% for class_doc in module_doc.classes %}
                <div class="class">
                    <h3>{{ class_doc.name }}</h3>
                    {% if class_doc.docstring %}
                    <div class="docstring">{{ class_doc.docstring }}</div>
                    {% endif %}
                    
                    {% if class_doc.inheritance %}
                    <p><strong>Inheritance:</strong> {{ class_doc.inheritance|join(' -> ') }}</p>
                    {% endif %}
                    
                    {% if class_doc.attributes %}
                    <h4>Attributes</h4>
                    <ul>
                    {% for attr in class_doc.attributes %}
                        <li><code>{{ attr.name }}</code>: {{ attr.type or 'Any' }} = {{ attr.value }}</li>
                    {% endfor %}
                    </ul>
                    {% endif %}
                    
                    {% if class_doc.methods %}
                    <h4>Methods</h4>
                    {% for method in class_doc.methods %}
                    <div class="function">
                        <div class="signature">{{ method.signature }}</div>
                        {% if method.docstring %}
                        <div class="docstring">{{ method.docstring }}</div>
                        {% endif %}
                        
                        {% if method.parameters %}
                        <div class="params">
                            <strong>Parameters:</strong>
                            <ul>
                            {% for param in method.parameters %}
                                <li class="param"><code>{{ param.name }}</code>: {{ param.type or 'Any' }}{% if param.default %} = {{ param.default }}{% endif %}</li>
                            {% endfor %}
                            </ul>
                        </div>
                        {% endif %}
                        
                        {% if method.return_type %}
                        <p><strong>Returns:</strong> {{ method.return_type }}</p>
                        {% endif %}
                    </div>
                    {% endfor %}
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            {% endif %}
            
            {% if module_doc.functions %}
            <div class="section">
                <h2>Functions</h2>
                {% for func_doc in module_doc.functions %}
                <div class="function">
                    <div class="signature">{{ func_doc.signature }}</div>
                    {% if func_doc.docstring %}
                    <div class="docstring">{{ func_doc.docstring }}</div>
                    {% endif %}
                    
                    {% if func_doc.parameters %}
                    <div class="params">
                        <strong>Parameters:</strong>
                        <ul>
                        {% for param in func_doc.parameters %}
                            <li class="param"><code>{{ param.name }}</code>: {{ param.type or 'Any' }}{% if param.default %} = {{ param.default }}{% endif %}</li>
                        {% endfor %}
                        </ul>
                    </div>
                    {% endif %}
                    
                    {% if func_doc.return_type %}
                    <p><strong>Returns:</strong> {{ func_doc.return_type }}</p>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </body>
        </html>
        """
        
        template = Template(html_template)
        return template.render(module_doc=module_doc)
    
    def _create_readme(self, modules: Dict[str, ModuleDoc]) -> str:
        """Create main README.md"""
        content = "# Qwen3-VL Model Documentation\n\n"
        content += "This documentation covers the Qwen3-VL model and its development tools.\n\n"
        
        content += "## Modules\n\n"
        for module_name, module_doc in modules.items():
            description = module_doc.docstring.split('\n')[0] if module_doc.docstring else "No description"
            content += f"- [{module_name}]({module_name}.md) - {description}\n"
        
        return content
    
    def _create_module_md(self, module_name: str, module_doc: ModuleDoc) -> str:
        """Create Markdown documentation for a module"""
        content = f"# {module_doc.name}\n\n"
        
        if module_doc.docstring:
            content += f"## Description\n\n{module_doc.docstring}\n\n"
        
        if module_doc.imports:
            content += "## Imports\n\n"
            for imp in module_doc.imports:
                content += f"- `{imp}`\n"
            content += "\n"
        
        if module_doc.classes:
            content += "## Classes\n\n"
            for class_doc in module_doc.classes:
                content += f"### {class_doc.name}\n\n"
                
                if class_doc.docstring:
                    content += f"{class_doc.docstring}\n\n"
                
                if class_doc.inheritance:
                    content += f"**Inheritance:** { ' -> '.join(class_doc.inheritance) }\n\n"
                
                if class_doc.attributes:
                    content += "#### Attributes\n\n"
                    for attr in class_doc.attributes:
                        content += f"- `{attr.name}`: {attr.type or 'Any'} = {attr.value}\n"
                    content += "\n"
                
                if class_doc.methods:
                    content += "#### Methods\n\n"
                    for method in class_doc.methods:
                        content += f"##### `{method.signature}`\n\n"
                        if method.docstring:
                            content += f"{method.docstring}\n\n"
                        
                        if method.parameters:
                            content += "**Parameters:**\n\n"
                            for param in method.parameters:
                                default_str = f" = {param.default}" if param.default is not None else ""
                                content += f"- `{param.name}`: {param.type or 'Any'}{default_str}\n"
                            content += "\n"
                        
                        if method.return_type:
                            content += f"**Returns:** {method.return_type}\n\n"
                    
                    content += "\n"
        
        if module_doc.functions:
            content += "## Functions\n\n"
            for func_doc in module_doc.functions:
                content += f"### `{func_doc.signature}`\n\n"
                if func_doc.docstring:
                    content += f"{func_doc.docstring}\n\n"
                
                if func_doc.parameters:
                    content += "**Parameters:**\n\n"
                    for param in func_doc.parameters:
                        default_str = f" = {param.default}" if param.default is not None else ""
                        content += f"- `{param.name}`: {param.type or 'Any'}{default_str}\n"
                    content += "\n"
                
                if func_doc.return_type:
                    content += f"**Returns:** {func_doc.return_type}\n\n"
        
        return content
    
    def generate_model_docs(self, model: nn.Module, model_name: str = "Qwen3-VL", output_path: str = None):
        """Generate documentation for a PyTorch model"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{model_name.lower()}_model.md")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate model documentation
        api_doc = self.model_doc_generator.generate_model_api_doc(model, model_name)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(api_doc)
        
        return api_doc
    
    def generate_usage_guide(self, guide_name: str, content: str, output_path: str = None):
        """Generate a usage guide"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"{guide_name.lower().replace(' ', '_')}.md")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def generate_config_docs(self, config: Dict[str, Any], output_path: str = None):
        """Generate documentation for configuration"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "configuration.md")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        content = "# Configuration Documentation\n\n"
        content += "This document describes the configuration options for the Qwen3-VL model.\n\n"
        
        content += self._dict_to_markdown(config, level=2)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _dict_to_markdown(self, d: Dict[str, Any], level: int = 2) -> str:
        """Convert a dictionary to markdown"""
        content = ""
        for key, value in d.items():
            header_prefix = "#" * level
            content += f"{header_prefix} {key}\n\n"
            
            if isinstance(value, dict):
                content += self._dict_to_markdown(value, level + 1)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        content += f"{header_prefix}# Item {i}\n\n"
                        content += self._dict_to_markdown(item, level + 1)
                    else:
                        content += f"- {item}\n"
                content += "\n"
            else:
                content += f"{value}\n\n"
        
        return content
    
    def generate_api_reference(self, output_path: str = None):
        """Generate API reference documentation"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "api_reference.md")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        content = "# API Reference\n\n"
        content += "This section documents the public API of the Qwen3-VL model development tools.\n\n"
        
        # Add API reference content here based on parsed modules
        # For now, we'll create a template
        content += "## Available Modules\n\n"
        for module_name in self.doc_parser.modules:
            content += f"- [{module_name}](#{module_name.replace('_', '-')})\n"
        
        content += "\n## Module Details\n\n"
        for module_name, module_doc in self.doc_parser.modules.items():
            content += f"### {module_name}\n\n"
            
            if module_doc.docstring:
                content += f"{module_doc.docstring}\n\n"
            
            if module_doc.functions:
                content += "#### Functions\n\n"
                for func in module_doc.functions:
                    content += f"- `{func.signature}` - {func.docstring.split('.')[0] if func.docstring else 'No description'}\n"
                content += "\n"
            
            if module_doc.classes:
                content += "#### Classes\n\n"
                for cls in module_doc.classes:
                    content += f"- `{cls.name}` - {cls.docstring.split('.')[0] if cls.docstring else 'No description'}\n"
                content += "\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)


class DocumentationTemplate:
    """Template system for documentation"""
    
    def __init__(self):
        self.templates = {}
    
    def register_template(self, name: str, template: str):
        """Register a documentation template"""
        self.templates[name] = Template(template)
    
    def render_template(self, name: str, **kwargs) -> str:
        """Render a registered template with provided data"""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        return self.templates[name].render(**kwargs)
    
    def create_model_template(self):
        """Create a template for model documentation"""
        template = """
# {{ model_name }} Model Documentation

**Architecture:** {{ architecture }}

**Parameters:** {{ total_params | int | format_number }} total ({{ trainable_params | int | format_number }} trainable)

## Description
{{ description }}

## Architecture Details
{{ architecture_details }}

## Usage Example
```python
import torch
from qwen3_vl import {{ model_name }}

model = {{ model_name }}()
# Example usage
```

## Configuration
{{ configuration | to_yaml }}
"""
        self.register_template('model_doc', template)


def create_usage_guide_template():
    """Create a template for usage guides"""
    guide_template = """
# {{ title }}

{{ description }}

## Prerequisites
{{ prerequisites }}

## Installation
```bash
{{ installation_commands }}
```

## Quick Start
```python
{{ quick_start_code }}
```

## Advanced Usage
{{ advanced_usage }}

## Configuration
{{ configuration_options }}

## Troubleshooting
{{ troubleshooting }}
"""
    return guide_template


def example_documentation_generation():
    """Example of documentation generation"""
    print("=== Documentation Generation Example ===")
    
    # Create a documentation generator
    doc_gen = DocumentationGenerator()
    
    # Create a simple model for documentation
    class SimpleModel(nn.Module):
        """
        A simple example model for documentation generation.
        
        This model demonstrates how documentation tools work with PyTorch models.
        """
        def __init__(self, input_size: int = 784, hidden_size: int = 128, output_size: int = 10):
            """
            Initialize the SimpleModel.
            
            Args:
                input_size: Size of input features
                hidden_size: Size of hidden layer
                output_size: Size of output layer
            """
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, output_size)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the model.
            
            Args:
                x: Input tensor of shape (batch_size, input_size)
                
            Returns:
                Output tensor of shape (batch_size, output_size)
            """
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleModel()
    
    # Generate model documentation
    model_doc = doc_gen.generate_model_docs(model, "SimpleModel", "docs/simple_model.md")
    print("Generated model documentation")
    
    # Generate a usage guide
    guide_content = """
# Qwen3-VL Quick Start Guide

This guide shows how to get started with the Qwen3-VL model.

## Installation

```bash
pip install torch qwen3-vl
```

## Basic Usage

```python
import torch
from qwen3_vl import Qwen3VLModel

model = Qwen3VLModel.from_pretrained("qwen3-vl-base")
inputs = model.preprocess_images_and_text(images, text)
outputs = model(inputs)
```
"""
    doc_gen.generate_usage_guide("Quick Start", guide_content)
    print("Generated usage guide")
    
    # Generate configuration documentation
    config = {
        "model": {
            "name": "qwen3_vl_2b",
            "transformer_layers": 32,
            "attention_heads": 32,
            "hidden_size": 2560
        },
        "training": {
            "batch_size": 1,
            "learning_rate": 5e-5,
            "optimizer": "adamw"
        },
        "optimization": {
            "memory_efficient": True,
            "use_mixed_precision": True,
            "kv_cache_compression": True
        }
    }
    doc_gen.generate_config_docs(config)
    print("Generated configuration documentation")
    
    # Create a simple API reference
    doc_gen.generate_api_reference()
    print("Generated API reference")


def example_code_parsing():
    """Example of code documentation parsing"""
    print("\n=== Code Documentation Parsing Example ===")
    
    # Create a temporary Python file to parse
    test_code = '''
"""
Test module for documentation generation.
This module contains example classes and functions.
"""

import torch
import torch.nn as nn
from typing import Optional


class ExampleClass:
    """
    An example class for documentation.
    
    This class demonstrates how documentation is extracted from source code.
    """
    
    def __init__(self, param1: int, param2: str = "default"):
        """
        Initialize the ExampleClass.
        
        Args:
            param1: An integer parameter
            param2: A string parameter with default value
        """
        self.param1 = param1
        self.param2 = param2
        self.internal_state = 0
    
    def example_method(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor.
        
        Args:
            input_tensor: Input tensor to process
            
        Returns:
            Processed tensor
        """
        return input_tensor * self.param1


def example_function(x: float, y: float = 1.0) -> float:
    """
    An example function for documentation.
    
    Args:
        x: First parameter
        y: Second parameter with default
        
    Returns:
        Product of x and y
    """
    return x * y
'''
    
    # Write test code to a temporary file
    with open("temp_test_module.py", "w") as f:
        f.write(test_code)
    
    # Parse the file
    doc_parser = DocParser()
    module_doc = doc_parser.parse_file("temp_test_module.py")
    
    print(f"Parsed module: {module_doc.name}")
    print(f"Module docstring: {module_doc.docstring}")
    print(f"Found {len(module_doc.classes)} classes and {len(module_doc.functions)} functions")
    
    # Print class information
    for class_doc in module_doc.classes:
        print(f"\nClass: {class_doc.name}")
        print(f"Docstring: {class_doc.docstring}")
        print(f"Inheritance: {class_doc.inheritance}")
        print(f"Methods: {len(class_doc.methods)}")
    
    # Print function information
    for func_doc in module_doc.functions:
        print(f"\nFunction: {func_doc.name}")
        print(f"Signature: {func_doc.signature}")
        print(f"Docstring: {func_doc.docstring}")
    
    # Clean up
    if os.path.exists("temp_test_module.py"):
        os.remove("temp_test_module.py")


if __name__ == "__main__":
    example_documentation_generation()
    example_code_parsing()