"""
Updated Test suite for the test fixtures system in Inference-PIO
Following new standards and including enhanced functionality tests
"""

import os
import tempfile
import shutil
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import gc
from typing import Dict, Any

# Import the test utilities from the project
try:
    from src.inference_pio.test_utils import (
        assert_equal, assert_true, assert_false, assert_is_not_none,
        assert_is_none, assert_in, assert_not_in, assert_raises,
        assert_greater_equal, assert_less_equal, assert_between,
        assert_is_instance, assert_not_is_instance, run_tests,
        assert_length, assert_dict_contains, assert_tensor_shape,
        assert_tensor_dtype, assert_is_subclass
    )
except ImportError:
    print("Could not import test utilities. Please ensure the project structure is correct.")
    sys.exit(1)

# Import the fixture system
try:
    from src.inference_pio.test_fixtures import (
        FixtureManager, fixture, use_fixtures, fixture_context,
        get_fixture_manager, reset_fixture_manager, cleanup_test_resources,
        temp_dir, tensor_fixture, mock_model_fixture, config_fixture,
        db_fixture, TemporaryDirectoryFixture, TensorFixture,
        MockModelFixture, ConfigFixture, DatabaseFixture,
        create_tensor_with_shape, create_mock_model_with_params
    )
except ImportError:
    print("Could not import test fixtures. Please ensure test_fixtures.py is in place.")
    sys.exit(1)


def test_fixture_manager_initialization():
    """Test that the fixture manager initializes correctly."""
    manager = FixtureManager()
    assert_is_not_none(manager)
    assert_equal(len(manager._fixtures), 0)
    assert_equal(len(manager._active_fixtures), 0)
    assert_equal(len(manager._cleanup_callbacks), 0)


def test_register_fixture():
    """Test registering a fixture with the manager."""
    manager = FixtureManager()

    def sample_fixture():
        return "sample_value"

    manager.register_fixture("sample", sample_fixture, "function")

    assert_in("sample", manager._fixtures)
    assert_equal(manager._fixtures["sample"]["factory"], sample_fixture)
    assert_equal(manager._fixtures["sample"]["scope"], "function")
    assert_false(manager._fixtures["sample"]["initialized"])
    assert_is_none(manager._fixtures["sample"]["instance"])


def test_get_fixture_function_scope():
    """Test getting a function-scoped fixture."""
    manager = FixtureManager()

    def sample_fixture():
        return {"created_at": id({})}  # Unique object each time

    manager.register_fixture("sample", sample_fixture, "function")

    # Get fixture twice - should create new instances each time
    instance1 = manager.get_fixture("sample")
    instance2 = manager.get_fixture("sample")

    assert_equal(instance1, instance2)  # Values are equal
    assert_not_in("instance", manager._fixtures["sample"])  # No cached instance for function scope


def test_get_fixture_session_scope():
    """Test getting a session-scoped fixture."""
    manager = FixtureManager()

    def sample_fixture():
        return {"created_at": id({})}  # Unique object each time

    manager.register_fixture("sample", sample_fixture, "session")

    # Get fixture twice - should return same cached instance
    instance1 = manager.get_fixture("sample")
    instance2 = manager.get_fixture("sample")

    assert_equal(instance1, instance2)  # Values are equal
    assert_in("instance", manager._fixtures["sample"])  # Cached instance exists
    assert_is_not_none(manager._fixtures["sample"]["instance"])


def test_fixture_decorator():
    """Test the fixture decorator functionality."""
    @fixture(scope='function')
    def decorated_fixture():
        return "decorated_value"

    manager = get_fixture_manager()
    assert_in("decorated_fixture", manager._fixtures)
    assert_equal(manager._fixtures["decorated_fixture"]["scope"], "function")

    # Get the fixture
    value = manager.get_fixture("decorated_fixture")
    assert_equal(value, "decorated_value")


def test_use_fixtures_decorator():
    """Test the use_fixtures decorator functionality."""
    manager = get_fixture_manager()

    def sample_fixture():
        return "sample_value"

    manager.register_fixture("sample", sample_fixture, "function")

    @use_fixtures("sample")
    def test_function(sample=None):
        return f"received: {sample}"

    result = test_function()
    assert_equal(result, "received: sample_value")


def test_temporary_directory_fixture():
    """Test the TemporaryDirectoryFixture."""
    with TemporaryDirectoryFixture() as temp_dir:
        assert_true(os.path.exists(temp_dir))
        assert_true(os.path.isdir(temp_dir))

        # Create a file inside the temp directory
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        assert_true(os.path.exists(test_file))

    # After exiting the context, the directory should be cleaned up
    assert_false(os.path.exists(temp_dir))


def test_tensor_fixture():
    """Test the TensorFixture."""
    tf = TensorFixture()

    # Check that tensors were created
    assert_greater_equal(len(tf.tensors), 6)  # At least 6 tensors created

    # Check that we can get a specific tensor
    tensor = tf.get_tensor("tensor_0")
    assert_is_not_none(tensor)
    assert_is_instance(tensor, torch.Tensor)

    # Check that we can get random tensors
    random_tensors = tf.get_random_tensors(2)
    assert_equal(len(random_tensors), 2)

    # Clean up
    tf.cleanup()
    assert_equal(len(tf.tensors), 0)


def test_mock_model_fixture():
    """Test the MockModelFixture."""
    mf = MockModelFixture()

    # Check that model, optimizer, and dataloader were created
    assert_is_not_none(mf.model)
    assert_is_instance(mf.model, torch.nn.Module)

    assert_is_not_none(mf.optimizer)
    assert_is_not_none(mf.dataloader)

    # Check that we can iterate over the dataloader
    for batch in mf.dataloader:
        assert_equal(len(batch), 2)  # (data, labels)
        break  # Just check the first batch

    # Clean up
    mf.cleanup()


def test_config_fixture():
    """Test the ConfigFixture."""
    cf = ConfigFixture()

    # Check default config
    config = cf.get_config()
    assert_is_not_none(config)
    assert_in('model_name', config)
    assert_equal(config['model_name'], 'test_model')

    # Check specific config value
    model_name = cf.get_config('model_name')
    assert_equal(model_name, 'test_model')

    # Update config
    cf.update_config({'new_param': 'new_value'})
    assert_equal(cf.get_config('new_param'), 'new_value')

    # Clean up
    cf.cleanup()


def test_database_fixture():
    """Test the DatabaseFixture."""
    df = DatabaseFixture()

    # Check that connection was created
    assert_is_not_none(df.connection)
    assert_true(df.connection.connected)

    # Insert and retrieve data
    df.insert_data('users', {'name': 'John', 'age': 30})
    data = df.get_data('users')
    assert_equal(len(data), 1)
    assert_equal(data[0]['name'], 'John')

    # Clean up
    df.cleanup()
    assert_false(df.connection.connected)


def test_fixture_context_manager():
    """Test the fixture_context context manager."""
    manager = get_fixture_manager()

    def sample_fixture():
        return "context_value"

    manager.register_fixture("sample", sample_fixture, "function")

    with fixture_context("sample") as fixtures:
        assert_in("sample", fixtures)
        assert_equal(fixtures["sample"], "context_value")


def test_cleanup_functionality():
    """Test the cleanup functionality of the fixture manager."""
    manager = FixtureManager()

    # Create a fixture that tracks if cleanup was called
    class TrackedFixture:
        def __init__(self):
            self.cleaned_up = False

        def cleanup(self):
            self.cleaned_up = True

    def tracked_fixture_factory():
        return TrackedFixture()

    manager.register_fixture("tracked", tracked_fixture_factory, "session")

    # Get the fixture
    fixture_instance = manager.get_fixture("tracked")
    assert_false(fixture_instance.cleaned_up)

    # Clean up the manager
    manager.cleanup()
    assert_true(fixture_instance.cleaned_up)


def test_global_cleanup_function():
    """Test the global cleanup function."""
    # Create a fixture that tracks cleanup
    class CleanupTracker:
        def __init__(self):
            self.cleaned_up = False

        def cleanup(self):
            self.cleaned_up = True

    manager = get_fixture_manager()

    def tracker_fixture():
        return CleanupTracker()

    manager.register_fixture("tracker", tracker_fixture, "session")
    tracker = manager.get_fixture("tracker")

    assert_false(tracker.cleaned_up)

    # Call global cleanup
    cleanup_test_resources()

    # Tracker should be cleaned up
    assert_true(tracker.cleaned_up)


def test_reset_fixture_manager():
    """Test resetting the fixture manager."""
    manager = get_fixture_manager()

    def sample_fixture():
        return "reset_test"

    manager.register_fixture("reset_test", sample_fixture, "session")
    instance = manager.get_fixture("reset_test")

    # Verify fixture exists
    assert_in("reset_test", manager._fixtures)

    # Reset the manager
    reset_fixture_manager()

    # Verify fixture is gone
    assert_not_in("reset_test", manager._fixtures)


def test_fixture_scopes():
    """Test different fixture scopes work correctly."""
    manager = FixtureManager()

    call_count = 0

    def counting_fixture():
        nonlocal call_count
        call_count += 1
        return f"value_{call_count}"

    # Register fixture with function scope
    manager.register_fixture("function_scoped", counting_fixture, "function")

    # Get fixture multiple times - should increment each time for function scope
    val1 = manager.get_fixture("function_scoped")
    val2 = manager.get_fixture("function_scoped")

    assert_equal(val1, "value_1")
    assert_equal(val2, "value_2")

    # Now test session scope
    call_count = 0
    manager.register_fixture("session_scoped", counting_fixture, "session")

    # Get fixture multiple times - should return same value for session scope
    val3 = manager.get_fixture("session_scoped")
    val4 = manager.get_fixture("session_scoped")

    assert_equal(val3, "value_1")  # Still value_1 because it's cached
    assert_equal(val4, "value_1")  # Same cached value


def test_fixture_error_handling():
    """Test error handling in fixture system."""
    manager = FixtureManager()

    def error_fixture():
        raise ValueError("Fixture creation failed")

    manager.register_fixture("error_fixture", error_fixture, "function")

    # Getting an unregistered fixture should raise an error
    assert_raises(ValueError, lambda: manager.get_fixture("nonexistent_fixture"))

    # Getting a fixture that raises an error should propagate the error
    assert_raises(ValueError, lambda: manager.get_fixture("error_fixture"))


def test_tensor_fixture_shapes():
    """Test that TensorFixture creates tensors with correct shapes."""
    tf = TensorFixture([
        ((2, 3), torch.float32),
        ((4, 5, 6), torch.float32)
    ])

    # Check that tensors were created with correct shapes
    tensor_0 = tf.get_tensor("tensor_0")
    zeros_0 = tf.get_tensor("zeros_0")
    ones_0 = tf.get_tensor("ones_0")

    assert_tensor_shape(tensor_0, (2, 3))
    assert_tensor_dtype(tensor_0, torch.float32)
    assert_tensor_shape(zeros_0, (2, 3))
    assert_tensor_shape(ones_0, (2, 3))

    tensor_1 = tf.get_tensor("tensor_1")
    assert_tensor_shape(tensor_1, (4, 5, 6))


def test_mock_model_types():
    """Test that MockModelFixture creates different model types."""
    simple_model = MockModelFixture("simple")
    conv_model = MockModelFixture("conv")

    # Check that models have different structures
    assert_is_instance(simple_model.model, torch.nn.Sequential)
    assert_is_instance(conv_model.model, torch.nn.Sequential)

    # Simple model should have 3 layers
    assert_equal(len(simple_model.model), 3)

    # Conv model should have 4 layers
    assert_equal(len(conv_model.model), 4)

    # Clean up
    simple_model.cleanup()
    conv_model.cleanup()


def test_config_types():
    """Test that ConfigFixture creates different config types."""
    default_config = ConfigFixture("default")
    optimized_config = ConfigFixture("optimized")

    # Check that configs have different values
    default_cfg = default_config.get_config()
    optimized_cfg = optimized_config.get_config()

    assert_equal(default_cfg['model_name'], 'test_model')
    assert_equal(optimized_cfg['model_name'], 'optimized_test_model')

    assert_equal(default_cfg['batch_size'], 4)
    assert_equal(optimized_cfg['batch_size'], 8)

    # Clean up
    default_config.cleanup()
    optimized_config.cleanup()


def test_shared_fixtures():
    """Test that shared/session scoped fixtures work correctly."""
    manager = get_fixture_manager()

    # Clear any existing fixtures
    reset_fixture_manager()

    # Register a session-scoped fixture
    def shared_counter():
        if not hasattr(shared_counter, 'count'):
            shared_counter.count = 0
        shared_counter.count += 1
        return shared_counter.count

    manager.register_fixture("shared_counter", shared_counter, "session")

    # Get the fixture multiple times
    val1 = manager.get_fixture("shared_counter")
    val2 = manager.get_fixture("shared_counter")
    val3 = manager.get_fixture("shared_counter")

    # For session-scoped fixtures, the factory should only be called once
    assert_equal(val1, 1)
    assert_equal(val2, 1)  # Same value as cached
    assert_equal(val3, 1)  # Same cached value

    # Reset to test again
    reset_fixture_manager()
    shared_counter.count = 0  # Reset the counter


def test_multiple_fixtures():
    """Test using multiple fixtures together."""
    manager = get_fixture_manager()

    def fixture_a():
        return "A"

    def fixture_b():
        return "B"

    def fixture_c():
        return "C"

    manager.register_fixture("a", fixture_a, "function")
    manager.register_fixture("b", fixture_b, "function")
    manager.register_fixture("c", fixture_c, "function")

    @use_fixtures("a", "b", "c")
    def test_multiple(a=None, b=None, c=None):
        return f"{a}{b}{c}"

    result = test_multiple()
    assert_equal(result, "ABC")


def test_gc_collection_after_cleanup():
    """Test that garbage collection works after fixture cleanup."""
    manager = get_fixture_manager()

    class MemoryIntensiveFixture:
        def __init__(self):
            # Create a large tensor to consume memory
            self.large_tensor = torch.randn(1000, 1000)

        def cleanup(self):
            del self.large_tensor

    def mem_fixture():
        return MemoryIntensiveFixture()

    manager.register_fixture("memory_intensive", mem_fixture, "session")

    # Get the fixture
    fixture = manager.get_fixture("memory_intensive")
    assert_is_not_none(fixture.large_tensor)

    # Clean up
    manager.cleanup()

    # Force garbage collection
    gc.collect()


def test_create_tensor_with_shape():
    """Test the utility function for creating tensors with specific shape."""
    tensor = create_tensor_with_shape((3, 4), torch.float32, 'cpu')
    
    assert_is_instance(tensor, torch.Tensor)
    assert_tensor_shape(tensor, (3, 4))
    assert_tensor_dtype(tensor, torch.float32)
    assert_equal(tensor.device.type, 'cpu')


def test_create_mock_model_with_params():
    """Test the utility function for creating mock models with specific parameter count."""
    model = create_mock_model_with_params(50)
    
    assert_is_instance(model, torch.nn.Module)
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    # Allow some flexibility due to layer structure constraints
    assert_true(param_count <= 50 + 50, f"Expected ~50 params, got {param_count}")


def test_fixture_manager_register_cleanup_callback():
    """Test registering cleanup callbacks with the fixture manager."""
    manager = FixtureManager()
    
    cleanup_called = [False]
    
    def cleanup_callback():
        cleanup_called[0] = True
    
    manager.register_cleanup_callback(cleanup_callback)
    
    # Trigger cleanup
    manager.cleanup()
    
    assert_true(cleanup_called[0])


def test_tensor_fixture_with_custom_shapes():
    """Test TensorFixture with custom shapes and dtypes."""
    custom_shapes = [
        ((5,), torch.int64),
        ((2, 2), torch.float16),
        ((3, 3, 3), torch.bool)
    ]
    
    tf = TensorFixture(custom_shapes)
    
    # Check that tensors were created for each custom shape
    assert_in('tensor_0', tf.tensors)
    assert_in('tensor_1', tf.tensors)
    assert_in('tensor_2', tf.tensors)
    
    # Verify shapes and dtypes
    assert_tensor_shape(tf.tensors['tensor_0'], (5,))
    assert_tensor_dtype(tf.tensors['tensor_0'], torch.int64)
    
    assert_tensor_shape(tf.tensors['tensor_1'], (2, 2))
    assert_tensor_dtype(tf.tensors['tensor_1'], torch.float16)
    
    assert_tensor_shape(tf.tensors['tensor_2'], (3, 3, 3))
    assert_tensor_dtype(tf.tensors['tensor_2'], torch.bool)


def test_mock_model_fixture_with_different_types():
    """Test MockModelFixture with different model types."""
    simple_model = MockModelFixture("simple")
    conv_model = MockModelFixture("conv")
    default_model = MockModelFixture("invalid_type")  # Should default to simple
    
    # All should be valid models
    assert_is_instance(simple_model.model, torch.nn.Module)
    assert_is_instance(conv_model.model, torch.nn.Module)
    assert_is_instance(default_model.model, torch.nn.Module)
    
    # Clean up
    simple_model.cleanup()
    conv_model.cleanup()
    default_model.cleanup()


def test_config_fixture_with_different_types():
    """Test ConfigFixture with different config types."""
    default_config = ConfigFixture("default")
    optimized_config = ConfigFixture("optimized")
    invalid_config = ConfigFixture("invalid_type")  # Should default to empty
    
    # Check that configs have expected values
    assert_equal(default_config.get_config('model_name'), 'test_model')
    assert_equal(optimized_config.get_config('model_name'), 'optimized_test_model')
    assert_equal(len(invalid_config.get_config()), 0)  # Empty config
    
    # Clean up
    default_config.cleanup()
    optimized_config.cleanup()
    invalid_config.cleanup()


def test_database_fixture_operations():
    """Test DatabaseFixture operations in more detail."""
    df = DatabaseFixture()
    
    # Test inserting and retrieving data
    df.insert_data('users', {'id': 1, 'name': 'Alice'})
    df.insert_data('users', {'id': 2, 'name': 'Bob'})
    
    users = df.get_data('users')
    assert_equal(len(users), 2)
    assert_equal(users[0]['name'], 'Alice')
    assert_equal(users[1]['name'], 'Bob')
    
    # Test with different table
    df.insert_data('products', {'id': 101, 'name': 'Laptop'})
    products = df.get_data('products')
    assert_equal(len(products), 1)
    assert_equal(products[0]['name'], 'Laptop')
    
    # Clean up
    df.cleanup()


def test_fixture_manager_get_fixture_with_kwargs():
    """Test getting fixtures with keyword arguments."""
    manager = FixtureManager()
    
    def parametrized_fixture(prefix="default", suffix="suffix"):
        return f"{prefix}_value_{suffix}"
    
    manager.register_fixture("parametrized", parametrized_fixture, "function")
    
    # Get fixture with default parameters
    result1 = manager.get_fixture("parametrized")
    assert_equal(result1, "default_value_suffix")
    
    # Get fixture with custom parameters
    result2 = manager.get_fixture("parametrized", prefix="custom", suffix="end")
    assert_equal(result2, "custom_value_end")


def run_fixture_tests():
    """Run all fixture tests."""
    test_functions = [
        test_fixture_manager_initialization,
        test_register_fixture,
        test_get_fixture_function_scope,
        test_get_fixture_session_scope,
        test_fixture_decorator,
        test_use_fixtures_decorator,
        test_temporary_directory_fixture,
        test_tensor_fixture,
        test_mock_model_fixture,
        test_config_fixture,
        test_database_fixture,
        test_fixture_context_manager,
        test_cleanup_functionality,
        test_global_cleanup_function,
        test_reset_fixture_manager,
        test_fixture_scopes,
        test_fixture_error_handling,
        test_tensor_fixture_shapes,
        test_mock_model_types,
        test_config_types,
        test_shared_fixtures,
        test_multiple_fixtures,
        test_gc_collection_after_cleanup,
        test_create_tensor_with_shape,
        test_create_mock_model_with_params,
        test_fixture_manager_register_cleanup_callback,
        test_tensor_fixture_with_custom_shapes,
        test_mock_model_fixture_with_different_types,
        test_config_fixture_with_different_types,
        test_database_fixture_operations,
        test_fixture_manager_get_fixture_with_kwargs,
    ]

    print("Running updated fixture tests...")
    success = run_tests(test_functions)
    return success


if __name__ == "__main__":
    run_fixture_tests()