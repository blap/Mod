# Docstring Standards

This document outlines the docstring standards for the Inference-PIO project.

## Format

All docstrings must follow the **Google style** format as specified in the CONTRIBUTING.md guidelines.

### Function/Method Docstrings

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of what the function does.

    Args:
        param1: Description of param1.
        param2: Description of param2 with default value.

    Returns:
        Description of the return value.

    Raises:
        ExceptionType: Description of when this exception is raised.
    """
    # Implementation
```

### Class Docstrings

```python
class ExampleClass:
    """
    Brief description of the class.

    Longer description of the class functionality and purpose.
    Include information about the main responsibilities of the class.
    """

    def __init__(self, param1: str):
        """
        Initialize the ExampleClass.

        Args:
            param1: Description of param1.
        """
        # Implementation
```

### Module Docstrings

```python
"""
Module name - Brief description.

Detailed description of the module's purpose and functionality.
List the main classes and functions exported by the module.
"""
```

## Content Requirements

### Functions/Methods
- **Brief summary**: First line should be a brief imperative statement
- **Args section**: Document all parameters with type hints
- **Returns section**: Document return value with type
- **Raises section**: Document all possible exceptions

### Classes
- **Brief summary**: First line should describe the class
- **Detailed description**: Explain the class's purpose and responsibilities
- **Constructor**: Document `__init__` parameters separately if complex

### Modules
- **Purpose**: Explain what the module does
- **Exports**: List main classes/functions
- **Dependencies**: Mention important dependencies if relevant

## Type Hints

All functions and methods must include type hints for parameters and return values:

```python
from typing import List, Dict, Optional, Union

def example_typed_function(
    items: List[str],
    mapping: Dict[str, int],
    optional_param: Optional[str] = None
) -> Union[str, int]:
    """
    Function with proper type hints.

    Args:
        items: List of strings to process.
        mapping: Dictionary mapping strings to integers.
        optional_param: Optional string parameter.

    Returns:
        Either a string or integer depending on input.
    """
    # Implementation
```

## Examples

Include usage examples when beneficial:

```python
def example_with_usage(items: List[str]) -> str:
    """
    Process a list of items and return a concatenated string.

    Args:
        items: List of strings to concatenate.

    Returns:
        Concatenated string with items separated by commas.

    Example:
        >>> example_with_usage(['a', 'b', 'c'])
        'a,b,c'
    """
    return ','.join(items)
```

## Self-Contained Architecture Compliance

Since each model plugin is completely independent with its own configuration, tests, and benchmarks, documentation should:

- Clearly indicate which components belong to which model
- Use relative imports in examples: `from src.inference_pio.models.model_name.component import ...`
- Reference model-specific configurations and optimizations
- Include model-specific usage examples
- Document model-specific parameters and options
