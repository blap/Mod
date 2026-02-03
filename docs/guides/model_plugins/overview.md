# Creating a Model Plugin: Overview

This guide outlines how to create a new model plugin for the Inference-PIO system. The core architectural principle of Inference-PIO is **Model Independence**.

## Model Independence

Each model is implemented as a completely self-contained unit. This means every model plugin must have its own:

*   **Configuration**: Defined in `config.py` (inheriting from `BaseConfig`).
*   **Implementation**: Core logic in `model.py` and `plugin.py`.
*   **Tests**: Unit, integration, and performance tests located in `tests/`.
*   **Benchmarks**: Performance measurement scripts in `benchmarks/`.
*   **Optimizations**: Model-specific optimization logic (e.g., custom attention patterns).

Models should **not** import code from other model directories. Shared logic should be placed in `src/inference_pio/common/`.

## Documentation Requirements

When creating a new model, you must follow the project's documentation standards:

### Docstrings
All files must follow the docstring standards specified in [DOCSTRINGS.md](../../standards/DOCSTRINGS.md):

-   **Modules**: Must include docstrings explaining the module's purpose.
-   **Classes**: Must include docstrings with class description and responsibilities.
-   **Methods/Functions**: Must include docstrings with Args, Returns, and Raises when applicable.

### Comments
Follow the comment standards specified in [COMMENTS.md](../../standards/COMMENTS.md):

-   Explanatory comments for complex code.
-   TODO markers for future functionality.
-   Comments on model-specific optimizations.
