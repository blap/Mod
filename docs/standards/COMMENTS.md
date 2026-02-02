# Comment Standards

This document outlines the comment standards for the Inference-PIO project.

## Inline Comments

### Purpose
Inline comments should explain the "why" behind complex code, not the "what" which should be clear from the code itself.

### Format
- Use `# ` (hash followed by space) to begin inline comments
- Capitalize the first word of the comment
- Keep comments concise but informative

```python
# This is a good inline comment explaining why we're doing this
result = complex_calculation(x, y)
# TODO: Refactor this calculation to improve performance
```

### When to Use
- To explain complex algorithms or business logic
- To clarify non-obvious code behavior
- To mark temporary workarounds or areas for improvement
- To indicate assumptions or constraints

## Block Comments

### Purpose
Block comments explain larger sections of code or provide context for complex operations.

### Format
- Use `# ` at the beginning of each line
- Separate from code with blank lines above and below
- Use multiple lines for detailed explanations

```python
# The following block implements the attention mechanism
# with optimizations for memory usage. We use a sliding window
# approach to reduce computational complexity while maintaining
# accuracy for long sequences.

for i in range(len(sequence)):
    # Process each element in the sequence
    result = process_element(sequence[i])
```

## File Header Comments

### Purpose
File header comments provide essential information about the file's purpose and contents.

### Format
Each Python file should begin with a module docstring (not a comment), but additional header information can be included as needed:

```python
"""
Module Name - Brief Description

Detailed description of the module's purpose and functionality.
"""

# Author: Your Name
# Date: YYYY-MM-DD
# License: Apache 2.0
```

## TODO Comments

### Purpose
TODO comments mark incomplete functionality or areas that need future attention.

### Format
Use the format `# TODO: Description` with an optional assignee or issue reference:

```python
# TODO: Implement error handling for edge case
# TODO(user): Add unit tests for this function
# TODO(#123): Refactor this method to improve performance
```

## Model-Specific Comments

### Self-Contained Architecture
Since each model plugin is completely independent with its own configuration, tests, and benchmarks, comments should:

- Clearly indicate model-specific functionality
- Reference model-specific configurations and optimizations
- Explain model-specific parameters and options
- Mark areas that are unique to particular models

```python
# Qwen3-0.6B specific optimization: Apply special attention masking
# for this model's unique architecture requirements
if self.model_name == "qwen3_0_6b":
    apply_special_attention_mask(x)
```

## Performance-Related Comments

### Purpose
Comments related to performance optimizations should explain the rationale and expected benefits.

```python
# Apply fused kernel optimization to reduce memory allocations
# This reduces GPU memory usage by ~20% and improves throughput
result = fused_operation(x, y, z)
```

## Security-Related Comments

### Purpose
Comments related to security measures should explain the protection being implemented.

```python
# Validate file path to prevent directory traversal attacks
# Only allow access to files within the designated model directory
if not is_safe_path(file_path, allowed_directory):
    raise SecurityError("Unsafe file path detected")
```

## Testing-Related Comments

### Purpose
Comments related to testing should explain test scenarios and expectations.

```python
# Test with real-world data instead of mock data to ensure
# accurate performance measurements and realistic behavior
test_data = get_real_world_texts()[:10]
```