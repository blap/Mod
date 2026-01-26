"""
Test Fixtures System for Inference-PIO

This module provides a comprehensive fixture system for managing test data and resources
across different test modules. It includes fixture decorators, context managers,
and resource management utilities to ensure consistent test setup and teardown.
"""

import os
import tempfile
import shutil
import torch
import numpy as np
from typing import Any, Dict, List, Callable, Generator, Optional, Union
from contextlib import contextmanager
from functools import wraps
import atexit
import gc


class FixtureManager:
    """
    Centralized fixture manager that handles the lifecycle of test fixtures.
    """
    
    def __init__(self):
        self._fixtures = {}
        self._active_fixtures = {}
        self._cleanup_callbacks = []
        
    def register_fixture(self, name: str, factory: Callable, scope: str = 'function'):
        """
        Register a fixture with the manager.
        
        Args:
            name: Name of the fixture
            factory: Factory function that creates the fixture
            scope: Scope of the fixture ('function', 'class', 'module', 'session')
        """
        self._fixtures[name] = {
            'factory': factory,
            'scope': scope,
            'instance': None,
            'initialized': False
        }
    
    def get_fixture(self, name: str, **kwargs):
        """
        Get a fixture instance, creating it if necessary.
        """
        if name not in self._fixtures:
            raise ValueError(f"Fixture '{name}' not registered")
            
        fixture_info = self._fixtures[name]
        
        # For function-scoped fixtures, always create a new instance
        if fixture_info['scope'] == 'function':
            return fixture_info['factory'](**kwargs)
        
        # For other scopes, cache the instance
        if not fixture_info['initialized']:
            fixture_info['instance'] = fixture_info['factory'](**kwargs)
            fixture_info['initialized'] = True
            
        return fixture_info['instance']
    
    def cleanup(self):
        """Clean up all active fixtures."""
        for fixture_info in self._fixtures.values():
            if fixture_info['initialized'] and fixture_info['instance'] is not None:
                if hasattr(fixture_info['instance'], 'cleanup'):
                    fixture_info['instance'].cleanup()
                elif hasattr(fixture_info['instance'], 'close'):
                    fixture_info['instance'].close()
                elif hasattr(fixture_info['instance'], '__del__'):
                    fixture_info['instance'].__del__()
                    
        # Execute registered cleanup callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception:
                pass  # Ignore cleanup errors
                
        self._fixtures.clear()
        self._active_fixtures.clear()
        self._cleanup_callbacks.clear()
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback to be executed during cleanup."""
        self._cleanup_callbacks.append(callback)


# Global fixture manager instance
_fixture_manager = FixtureManager()


def fixture(scope: str = 'function'):
    """
    Decorator to register a fixture function.
    
    Args:
        scope: Scope of the fixture ('function', 'class', 'module', 'session')
    """
    def decorator(func):
        _fixture_manager.register_fixture(func.__name__, func, scope)
        return func
    return decorator


def use_fixtures(*fixture_names):
    """
    Decorator to inject fixtures into a test function.
    
    Args:
        *fixture_names: Names of fixtures to inject
    """
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            # Inject fixtures as keyword arguments
            for name in fixture_names:
                if name not in kwargs:
                    kwargs[name] = _fixture_manager.get_fixture(name)
            
            return test_func(*args, **kwargs)
        return wrapper
    return decorator


class TemporaryDirectoryFixture:
    """
    A fixture that creates and manages a temporary directory.
    """
    
    def __init__(self, prefix: str = "inference_pio_test_", suffix: str = ""):
        self.prefix = prefix
        self.suffix = suffix
        self.temp_dir = None
        
    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp(prefix=self.prefix, suffix=self.suffix)
        return self.temp_dir
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def cleanup(self):
        """Explicit cleanup method."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)


@fixture(scope='function')
def temp_dir():
    """Function-scoped temporary directory fixture."""
    return TemporaryDirectoryFixture()


@fixture(scope='session')
def shared_temp_dir():
    """Session-scoped temporary directory fixture."""
    temp_dir = tempfile.mkdtemp(prefix="inference_pio_shared_")
    
    def cleanup():
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    _fixture_manager.register_cleanup_callback(cleanup)
    return temp_dir


class TensorFixture:
    """
    A fixture that creates and manages PyTorch tensors for testing.
    """
    
    def __init__(self, shapes_and_dtypes: List[tuple] = None):
        self.shapes_and_dtypes = shapes_and_dtypes or [
            ((2, 3), torch.float32),
            ((4, 5, 6), torch.float32),
            ((1, 10), torch.int64),
        ]
        self.tensors = {}
        self._create_tensors()
    
    def _create_tensors(self):
        """Create tensors based on predefined shapes and dtypes."""
        for i, (shape, dtype) in enumerate(self.shapes_and_dtypes):
            # For integer dtypes, use randint instead of randn
            if dtype in [torch.int64, torch.int32, torch.long]:
                tensor = torch.randint(low=-10, high=10, size=shape, dtype=dtype)
            else:
                tensor = torch.randn(shape, dtype=dtype)
            self.tensors[f'tensor_{i}'] = tensor
            self.tensors[f'zeros_{i}'] = torch.zeros(shape, dtype=dtype)
            self.tensors[f'ones_{i}'] = torch.ones(shape, dtype=dtype)
    
    def get_tensor(self, name: str):
        """Get a specific tensor by name."""
        return self.tensors.get(name)
    
    def get_random_tensors(self, count: int = 1):
        """Get random tensors."""
        keys = [k for k in self.tensors.keys() if k.startswith('tensor_')]
        selected_keys = keys[:count] if len(keys) >= count else keys
        return [self.tensors[k] for k in selected_keys]
    
    def cleanup(self):
        """Clean up tensors by deleting references and forcing garbage collection."""
        self.tensors.clear()
        gc.collect()


@fixture(scope='function')
def tensor_fixture():
    """Function-scoped tensor fixture."""
    return TensorFixture()


@fixture(scope='session')
def shared_tensor_fixture():
    """Session-scoped tensor fixture."""
    fixture = TensorFixture([
        ((10, 10), torch.float32),
        ((5, 5, 5), torch.float32),
        ((1, 100), torch.int64),
        ((20, 20), torch.float16),
        ((3, 3, 3, 3), torch.float32),
    ])
    
    def cleanup():
        fixture.cleanup()
    
    _fixture_manager.register_cleanup_callback(cleanup)
    return fixture


class MockModelFixture:
    """
    A fixture that creates mock model components for testing.
    """
    
    def __init__(self, model_type: str = "simple"):
        self.model_type = model_type
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.dataloader = self._create_dataloader()
    
    def _create_model(self):
        """Create a mock model based on type."""
        if self.model_type == "simple":
            # Simple linear model
            return torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 1)
            )
        elif self.model_type == "conv":
            # Simple CNN model
            return torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(),
                torch.nn.Linear(16, 10)
            )
        else:
            # Default to simple model
            return torch.nn.Sequential(
                torch.nn.Linear(10, 20),
                torch.nn.ReLU(),
                torch.nn.Linear(20, 1)
            )
    
    def _create_optimizer(self):
        """Create a mock optimizer."""
        return torch.optim.SGD(self.model.parameters(), lr=0.01)
    
    def _create_dataloader(self):
        """Create a mock dataloader-like object."""
        class MockDataLoader:
            def __init__(self, batch_size=4, num_batches=5):
                self.batch_size = batch_size
                self.num_batches = num_batches
            
            def __iter__(self):
                for i in range(self.num_batches):
                    yield torch.randn(self.batch_size, 10), torch.randint(0, 2, (self.batch_size,))
            
            def __len__(self):
                return self.num_batches
        
        return MockDataLoader()
    
    def cleanup(self):
        """Clean up model resources."""
        del self.model
        del self.optimizer
        del self.dataloader
        gc.collect()


@fixture(scope='function')
def mock_model_fixture():
    """Function-scoped mock model fixture."""
    return MockModelFixture()


@fixture(scope='class')
def class_mock_model_fixture():
    """Class-scoped mock model fixture."""
    return MockModelFixture()


class ConfigFixture:
    """
    A fixture that provides configuration objects for testing.
    """
    
    def __init__(self, config_type: str = "default"):
        self.config_type = config_type
        self.config = self._create_config()
    
    def _create_config(self):
        """Create a mock configuration."""
        if self.config_type == "default":
            return {
                'model_name': 'test_model',
                'batch_size': 4,
                'learning_rate': 0.001,
                'epochs': 10,
                'device': 'cpu',
                'precision': 'fp32',
                'optimization_level': 1,
                'enable_mixed_precision': False,
                'gradient_clipping': 1.0,
                'checkpoint_interval': 5,
                'log_level': 'INFO',
                'output_dir': './test_output',
                'seed': 42,
            }
        elif self.config_type == "optimized":
            return {
                'model_name': 'optimized_test_model',
                'batch_size': 8,
                'learning_rate': 0.01,
                'epochs': 20,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                'precision': 'fp16' if torch.cuda.is_available() else 'fp32',
                'optimization_level': 3,
                'enable_mixed_precision': torch.cuda.is_available(),
                'gradient_clipping': 0.5,
                'checkpoint_interval': 2,
                'log_level': 'DEBUG',
                'output_dir': './optimized_test_output',
                'seed': 123,
                'enable_tensor_core': torch.cuda.is_available(),
                'enable_activation_offloading': True,
                'enable_gradient_checkpointing': True,
            }
        else:
            return {}
    
    def get_config(self, key: str = None):
        """Get configuration value or entire config."""
        if key:
            return self.config.get(key)
        return self.config
    
    def update_config(self, updates: dict):
        """Update configuration values."""
        self.config.update(updates)
    
    def cleanup(self):
        """Clean up configuration resources."""
        self.config.clear()


@fixture(scope='function')
def config_fixture():
    """Function-scoped config fixture."""
    return ConfigFixture()


@fixture(scope='session')
def shared_config_fixture():
    """Session-scoped config fixture."""
    return ConfigFixture(config_type="optimized")


class DatabaseFixture:
    """
    A fixture that simulates database connections for testing.
    """
    
    def __init__(self, db_type: str = "mock"):
        self.db_type = db_type
        self.connection = self._create_connection()
        self.data = {}
    
    def _create_connection(self):
        """Create a mock database connection."""
        class MockConnection:
            def __init__(self):
                self.connected = True
            
            def close(self):
                self.connected = False
            
            def execute(self, query, params=None):
                # Mock execution
                return MockCursor()
        
        return MockConnection()
    
    def insert_data(self, table: str, data: dict):
        """Insert mock data."""
        if table not in self.data:
            self.data[table] = []
        self.data[table].append(data)
    
    def get_data(self, table: str):
        """Get mock data."""
        return self.data.get(table, [])
    
    def cleanup(self):
        """Clean up database resources."""
        if self.connection:
            self.connection.close()
        self.data.clear()


@fixture(scope='function')
def db_fixture():
    """Function-scoped database fixture."""
    return DatabaseFixture()


@contextmanager
def fixture_context(*fixture_names):
    """
    Context manager for using fixtures.
    
    Args:
        *fixture_names: Names of fixtures to use
    """
    fixtures = {}
    try:
        for name in fixture_names:
            fixtures[name] = _fixture_manager.get_fixture(name)
        yield fixtures
    finally:
        # Cleanup is handled by the fixture manager's registered callbacks
        pass


def get_fixture_manager():
    """
    Get the global fixture manager instance.
    """
    return _fixture_manager


def reset_fixture_manager():
    """
    Reset the fixture manager, clearing all fixtures and callbacks.
    """
    _fixture_manager.cleanup()


# Register cleanup at program exit
atexit.register(reset_fixture_manager)


# Utility functions for common fixture patterns
def create_tensor_with_shape(shape: tuple, dtype: torch.dtype = torch.float32, device: str = 'cpu'):
    """Utility to create a tensor with specific shape and properties."""
    return torch.randn(shape, dtype=dtype, device=device)


def create_mock_model_with_params(num_params: int = 100):
    """Utility to create a mock model with approximately specified number of parameters."""
    layers = []
    remaining_params = num_params
    
    # Create layers until we reach the desired parameter count
    while remaining_params > 0:
        # Create a layer with up to 50 params
        layer_size = min(remaining_params, 50)
        if remaining_params >= 10:
            layers.append(torch.nn.Linear(layer_size, layer_size))
            remaining_params -= layer_size * layer_size + layer_size  # weights + biases
        else:
            # Add a small layer to get closer to target
            layers.append(torch.nn.Linear(1, 1))
            remaining_params -= 2  # 1 weight + 1 bias
    
    return torch.nn.Sequential(*layers) if layers else torch.nn.Identity()


def cleanup_test_resources():
    """Global function to clean up test resources."""
    reset_fixture_manager()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()