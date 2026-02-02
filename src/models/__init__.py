"""
Models Package for Inference-PIO

This module provides access to all model implementations in the Inference-PIO system.
Implements automatic plugin discovery to scan subdirectories and register available plugins.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

# Get the current directory of this __init__.py file
current_dir = Path(__file__).parent

# Dictionary to store imported modules
discovered_modules = {}


def discover_and_register_plugins():
    """
    Discovers model plugins in subdirectories and registers them automatically.
    """
    # Add the parent directory to sys.path to enable relative imports from submodules
    src_dir = str(current_dir.parent)  # This is 'src'
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Explicitly add the models directory to sys.modules to ensure proper package recognition
    models_package_name = __name__  # 'src.models'
    if models_package_name not in sys.modules:
        sys.modules[models_package_name] = sys.modules[__name__]

    # Scan subdirectories to discover model plugins
    for item in current_dir.iterdir():
        if item.is_dir() and not item.name.startswith("__") and item.name != "test":
            # If it's a model type directory (language, vision_language, coding, specialized),
            # scan its subdirectories for actual models
            if item.name in ["language", "vision_language", "coding", "specialized"]:
                # Add the category to sys.modules as a package
                category_package_name = f"{__name__}.{item.name}"
                category_init = item / "__init__.py"
                if category_init.exists():
                    # Load the category package
                    category_spec = importlib.util.spec_from_file_location(
                        category_package_name, category_init
                    )
                    category_module = importlib.util.module_from_spec(category_spec)
                    sys.modules[category_package_name] = category_module
                    category_spec.loader.exec_module(category_module)

                for sub_item in item.iterdir():
                    if sub_item.is_dir() and not sub_item.name.startswith("__") and sub_item.name != "test":
                        # Look for __init__.py in the subdirectory
                        module_init = sub_item / "__init__.py"
                        if module_init.exists():
                            try:
                                # Import the submodule dynamically using importlib
                                module_name = f"{__name__}.{item.name}.{sub_item.name}"

                                # Create a module spec with proper package context to handle relative imports
                                spec = importlib.util.spec_from_file_location(
                                    module_name, module_init, submodule_search_locations=[str(sub_item)]
                                )

                                if spec is None:
                                    print(
                                        f"Warning: Could not create spec for model plugin '{sub_item.name}'",
                                        file=sys.stderr,
                                    )
                                    continue

                                # Create the module
                                module = importlib.util.module_from_spec(spec)

                                # Register the module in sys.modules before execution to handle relative imports
                                sys.modules[module_name] = module

                                # Execute the module to load its contents
                                try:
                                    spec.loader.exec_module(module)
                                except ValueError as e:
                                    # Handle relative import errors by temporarily patching the module
                                    # This occurs when modules use relative imports beyond the top level
                                    print(
                                        f"Info: Handling relative import issue for '{sub_item.name}': {e}",
                                        file=sys.stderr,
                                    )

                                    # Add the package to sys.modules to resolve relative import issues
                                    package_name = f"{__name__}.{item.name}.{sub_item.name}"
                                    if package_name not in sys.modules:
                                        sys.modules[package_name] = module

                                    # Retry execution
                                    spec.loader.exec_module(module)

                                # Store the module in our dictionary
                                discovered_modules[f"{item.name}.{sub_item.name}"] = module

                                # Add all items from the module's __all__ to this package's namespace
                                if hasattr(module, "__all__"):
                                    for attr_name in module.__all__:
                                        if hasattr(module, attr_name):
                                            attr_value = getattr(module, attr_name)
                                            setattr(sys.modules[__name__], attr_name, attr_value)

                            except ImportError as e:
                                print(
                                    f"Warning: Could not import model plugin '{sub_item.name}' from {item.name}: {e}",
                                    file=sys.stderr,
                                )
                            except Exception as e:
                                print(
                                    f"Warning: Error loading model plugin '{sub_item.name}' from {item.name}: {e}",
                                    file=sys.stderr,
                                )
            else:
                # Regular model directory (for backward compatibility or other models)
                # Look for __init__.py in the subdirectory
                module_init = item / "__init__.py"
                if module_init.exists():
                    try:
                        # Import the submodule dynamically using importlib
                        module_name = f"{__name__}.{item.name}"

                        # Create a module spec with proper package context to handle relative imports
                        spec = importlib.util.spec_from_file_location(
                            module_name, module_init, submodule_search_locations=[str(item)]
                        )

                        if spec is None:
                            print(
                                f"Warning: Could not create spec for model plugin '{item.name}'",
                                file=sys.stderr,
                            )
                            continue

                        # Create the module
                        module = importlib.util.module_from_spec(spec)

                        # Register the module in sys.modules before execution to handle relative imports
                        sys.modules[module_name] = module

                        # Execute the module to load its contents
                        try:
                            spec.loader.exec_module(module)
                        except ValueError as e:
                            # Handle relative import errors by temporarily patching the module
                            # This occurs when modules use relative imports beyond the top level
                            print(
                                f"Info: Handling relative import issue for '{item.name}': {e}",
                                file=sys.stderr,
                            )

                            # Add the package to sys.modules to resolve relative import issues
                            package_name = f"{__name__}.{item.name}"
                            if package_name not in sys.modules:
                                sys.modules[package_name] = module

                            # Retry execution
                            spec.loader.exec_module(module)

                        # Store the module in our dictionary
                        discovered_modules[item.name] = module

                        # Add all items from the module's __all__ to this package's namespace
                        if hasattr(module, "__all__"):
                            for attr_name in module.__all__:
                                if hasattr(module, attr_name):
                                    attr_value = getattr(module, attr_name)
                                    setattr(sys.modules[__name__], attr_name, attr_value)

                    except ImportError as e:
                        print(
                            f"Warning: Could not import model plugin '{item.name}': {e}",
                            file=sys.stderr,
                        )
                    except Exception as e:
                        print(
                            f"Warning: Error loading model plugin '{item.name}': {e}",
                            file=sys.stderr,
                        )


# Execute the discovery function
discover_and_register_plugins()

# Build __all__ dynamically from all discovered modules
__all__ = []
for module in discovered_modules.values():
    if hasattr(module, "__all__"):
        __all__.extend(module.__all__)
