"""
General utility functions for the Qwen3-VL model.

This module contains general-purpose utility functions that are commonly used
across the codebase but are not specifically related to tensor operations.
These utilities are centralized to reduce duplication and improve maintainability.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union, List, Dict, Any
import logging
import os
import json
import time
from pathlib import Path
import functools
import threading
from contextlib import contextmanager


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger with the specified name and level.

    This function provides a consistent way to create loggers throughout the
    codebase with proper formatting and configuration.

    Args:
        name: Name for the logger (typically __name__ of the module)
        level: Logging level (default: logging.INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers to the same logger
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], 
                         tensor_name: str = "tensor") -> bool:
    """
    Validate that a tensor has the expected shape.

    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape as a tuple
        tensor_name: Name for the tensor (for error messages)

    Returns:
        True if shape matches, False otherwise

    Raises:
        TypeError: If tensor is not a torch.Tensor
        ValueError: If expected_shape is not a tuple
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{tensor_name} must be a torch.Tensor, got {type(tensor)}")
    if not isinstance(expected_shape, tuple):
        raise ValueError(f"expected_shape must be a tuple, got {type(expected_shape)}")

    if tensor.shape == expected_shape:
        return True
    else:
        logging.warning(f"{tensor_name} shape mismatch: expected {expected_shape}, got {tensor.shape}")
        return False


def safe_tensor_operation(func):
    """
    Decorator to add error handling to tensor operations.

    This decorator catches common tensor-related errors and logs them,
    making debugging easier and providing better error messages.

    Args:
        func: Function to decorate

    Returns:
        Wrapped function with error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            logging.error(f"Runtime error in {func.__name__}: {e}")
            raise
        except ValueError as e:
            logging.error(f"Value error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper


def get_available_memory(device: Union[str, torch.device] = 'cuda') -> int:
    """
    Get available memory on the specified device.

    Args:
        device: Device to check memory for (default: 'cuda')

    Returns:
        Available memory in bytes
    """
    device = torch.device(device)
    
    if device.type == 'cuda' and torch.cuda.is_available():
        # For CUDA devices, return available memory
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        return total_memory - reserved_memory
    elif device.type == 'cpu':
        # For CPU, return available system memory
        import psutil
        return int(psutil.virtual_memory().available)
    else:
        # For other devices, return a reasonable default
        return 2 * 1024 * 1024 * 1024  # 2GB


def get_tensor_memory_size(tensor: torch.Tensor) -> int:
    """
    Calculate the memory size of a tensor in bytes.

    Args:
        tensor: Tensor to calculate size for

    Returns:
        Size in bytes
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(tensor)}")
    
    element_size = tensor.element_size()
    num_elements = tensor.nelement()
    return element_size * num_elements


def create_tensor_from_config(shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                            device: Union[str, torch.device] = 'cpu',
                            init_method: str = 'empty') -> torch.Tensor:
    """
    Create a tensor with the specified shape, dtype, and device.

    Args:
        shape: Shape of the tensor
        dtype: Data type of the tensor
        device: Device to create tensor on
        init_method: Initialization method ('empty', 'zeros', 'ones', 'rand', 'randn')

    Returns:
        Created tensor
    """
    if init_method == 'empty':
        return torch.empty(shape, dtype=dtype, device=device)
    elif init_method == 'zeros':
        return torch.zeros(shape, dtype=dtype, device=device)
    elif init_method == 'ones':
        return torch.ones(shape, dtype=dtype, device=device)
    elif init_method == 'rand':
        return torch.rand(shape, dtype=dtype, device=device)
    elif init_method == 'randn':
        return torch.randn(shape, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown init_method: {init_method}")


def get_nested_attribute(obj: Any, attr_path: str, default: Any = None) -> Any:
    """
    Get a nested attribute using dot notation.

    Args:
        obj: Object to get attribute from
        attr_path: Dot-separated path to attribute (e.g., "model.encoder.layers.0")
        default: Default value if attribute doesn't exist

    Returns:
        Attribute value or default
    """
    attrs = attr_path.split('.')
    current = obj
    
    for attr in attrs:
        try:
            current = getattr(current, attr)
        except AttributeError:
            return default
    
    return current


def set_nested_attribute(obj: Any, attr_path: str, value: Any) -> bool:
    """
    Set a nested attribute using dot notation.

    Args:
        obj: Object to set attribute on
        attr_path: Dot-separated path to attribute (e.g., "model.encoder.layers.0")
        value: Value to set

    Returns:
        True if successful, False otherwise
    """
    attrs = attr_path.split('.')
    if len(attrs) == 1:
        try:
            setattr(obj, attr_path, value)
            return True
        except AttributeError:
            return False
    
    try:
        parent = get_nested_attribute(obj, '.'.join(attrs[:-1]))
        if parent is not None:
            setattr(parent, attrs[-1], value)
            return True
    except:
        pass
    
    return False


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any], 
                overwrite: bool = False) -> Dict[str, Any]:
    """
    Merge two dictionaries with optional overwrite behavior.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (values that will potentially overwrite)
        overwrite: If True, values in dict2 will overwrite values in dict1

    Returns:
        Merged dictionary
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise TypeError("Both inputs must be dictionaries")
    
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and not overwrite:
            continue
        result[key] = value
    
    return result


def get_config_path(config_name: str) -> Path:
    """
    Get the path to a configuration file.

    Args:
        config_name: Name of the configuration file (with or without extension)

    Returns:
        Path object to the configuration file
    """
    # Look for config in multiple possible locations
    possible_paths = [
        Path(config_name),  # Direct path
        Path("configs") / config_name,  # configs directory
        Path("src/qwen3_vl/configs") / config_name,  # src config directory
        Path(".") / config_name,  # Current directory
    ]
    
    # Add .json extension if not provided
    if not config_name.endswith(('.json', '.yaml', '.yml', '.toml')):
        possible_paths.extend([
            Path(config_name + '.json'),
            Path("configs") / (config_name + '.json'),
            Path("src/qwen3_vl/configs") / (config_name + '.json'),
            Path(".") / (config_name + '.json'),
        ])
    
    for path in possible_paths:
        if path.exists():
            return path.resolve()
    
    # If not found, return the path in the current directory
    return Path(config_name).resolve()


def load_config(config_path: Union[str, Path], 
                config_type: str = 'json') -> Dict[str, Any]:
    """
    Load a configuration file.

    Args:
        config_path: Path to the configuration file
        config_type: Type of config file ('json', 'yaml', 'yml', 'toml')

    Returns:
        Loaded configuration as dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    if config_type.lower() in ['json']:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif config_type.lower() in ['yaml', 'yml']:
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML config files")
    else:
        raise ValueError(f"Unsupported config type: {config_type}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path],
                config_type: str = 'json') -> None:
    """
    Save a configuration file.

    Args:
        config: Configuration dictionary to save
        config_path: Path to save the configuration file
        config_type: Type of config file ('json', 'yaml', 'yml', 'toml')
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if config_type.lower() in ['json']:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    elif config_type.lower() in ['yaml', 'yml']:
        try:
            import yaml
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        except ImportError:
            raise ImportError("PyYAML is required to save YAML config files")
    else:
        raise ValueError(f"Unsupported config type: {config_type}")


def time_it(func):
    """
    Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function that logs execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
    return wrapper


@contextmanager
def temporary_seed(seed: int):
    """
    Context manager to temporarily set a random seed.

    Args:
        seed: Random seed to use temporarily
    """
    state = torch.get_rng_state()
    np_state = np.random.get_state()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    try:
        yield
    finally:
        torch.set_rng_state(state)
        np.random.set_state(np_state)


def get_device_count() -> int:
    """
    Get the number of available devices (GPUs).

    Returns:
        Number of available GPUs, or 1 if CUDA is not available
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 1


def get_device_name(device: Union[str, torch.device] = 'cuda') -> str:
    """
    Get the name of a device.

    Args:
        device: Device to get name for

    Returns:
        Device name as string
    """
    device = torch.device(device)
    
    if device.type == 'cuda':
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(device)
        else:
            return "cuda (unavailable)"
    else:
        return str(device)


def is_tensor_on_device(tensor: torch.Tensor, device: Union[str, torch.device]) -> bool:
    """
    Check if a tensor is on the specified device.

    Args:
        tensor: Tensor to check
        device: Device to check against

    Returns:
        True if tensor is on device, False otherwise
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Input must be a torch.Tensor, got {type(tensor)}")
    
    return tensor.device == torch.device(device)


def move_tensors_to_device(tensors: List[torch.Tensor], 
                          device: Union[str, torch.device]) -> List[torch.Tensor]:
    """
    Move a list of tensors to the specified device.

    Args:
        tensors: List of tensors to move
        device: Device to move tensors to

    Returns:
        List of tensors on the specified device
    """
    if not isinstance(tensors, (list, tuple)):
        raise TypeError(f"tensors must be a list or tuple, got {type(tensors)}")
    
    device = torch.device(device)
    return [tensor.to(device) for tensor in tensors]


def create_thread_pool_executor(max_workers: Optional[int] = None):
    """
    Create a ThreadPoolExecutor with appropriate max_workers.

    Args:
        max_workers: Maximum number of worker threads (default: number of CPUs)

    Returns:
        ThreadPoolExecutor instance
    """
    try:
        from concurrent.futures import ThreadPoolExecutor
        import multiprocessing
        
        if max_workers is None:
            max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
        
        return ThreadPoolExecutor(max_workers=max_workers)
    except ImportError:
        raise ImportError("concurrent.futures is required for ThreadPoolExecutor")


def atomic_operation(func):
    """
    Decorator to make a function thread-safe with a lock.

    Args:
        func: Function to make thread-safe

    Returns:
        Thread-safe wrapped function
    """
    lock = threading.Lock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    return wrapper


def get_cache_dir() -> Path:
    """
    Get the default cache directory for the model.

    Returns:
        Path to cache directory
    """
    cache_dir = Path.home() / ".cache" / "qwen3_vl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def is_debug_mode() -> bool:
    """
    Check if the application is running in debug mode.

    Returns:
        True if debug mode is enabled, False otherwise
    """
    return os.getenv('QWEN3_VL_DEBUG', '').lower() in ('1', 'true', 'yes', 'on')


def get_debug_logger() -> Optional[logging.Logger]:
    """
    Get a logger that only logs in debug mode.

    Returns:
        Logger if debug mode is enabled, None otherwise
    """
    if is_debug_mode():
        return get_logger("qwen3_vl.debug", logging.DEBUG)
    return None


__all__ = [
    "get_logger",
    "validate_tensor_shape", 
    "safe_tensor_operation",
    "get_available_memory",
    "get_tensor_memory_size",
    "create_tensor_from_config",
    "get_nested_attribute",
    "set_nested_attribute",
    "merge_dicts",
    "get_config_path",
    "load_config",
    "save_config",
    "time_it",
    "temporary_seed",
    "get_device_count",
    "get_device_name",
    "is_tensor_on_device",
    "move_tensors_to_device",
    "create_thread_pool_executor",
    "atomic_operation",
    "get_cache_dir",
    "is_debug_mode",
    "get_debug_logger"
]