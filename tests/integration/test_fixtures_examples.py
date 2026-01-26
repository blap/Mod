"""
Example test file demonstrating the usage of the test fixture system
"""

import torch
import os
from unittest.mock import Mock

# Import the test utilities from the project
from src.inference_pio.test_utils import (
    assert_equal, assert_true, assert_false, assert_is_not_none, 
    assert_in, assert_greater, run_tests
)

# Import the fixture system
from src.inference_pio.test_fixtures import (
    fixture, use_fixtures, fixture_context, temp_dir, tensor_fixture,
    mock_model_fixture, config_fixture
)


def test_with_temp_dir_fixture():
    """Test using the temp_dir fixture."""
    with fixture_context('temp_dir') as fixtures:
        temp_dir_path = fixtures['temp_dir']
        
        # Create a file in the temporary directory
        test_file = os.path.join(temp_dir_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")
        
        # Verify the file exists
        assert_true(os.path.exists(test_file))
        
        # Verify we can read the content back
        with open(test_file, "r") as f:
            content = f.read()
        assert_equal(content, "test content")


@use_fixtures('tensor_fixture')
def test_with_tensor_fixture(tensor_fixture=None):
    """Test using the tensor_fixture."""
    # Get a specific tensor
    tensor = tensor_fixture.get_tensor("tensor_0")
    assert_is_not_none(tensor)
    assert_true(isinstance(tensor, torch.Tensor))
    
    # Get random tensors
    random_tensors = tensor_fixture.get_random_tensors(2)
    assert_equal(len(random_tensors), 2)
    
    # Verify tensor properties
    for t in random_tensors:
        assert_true(isinstance(t, torch.Tensor))
        assert_greater(t.nelement(), 0)


@use_fixtures('mock_model_fixture')
def test_with_mock_model_fixture(mock_model_fixture=None):
    """Test using the mock_model_fixture."""
    # Verify model components exist
    assert_is_not_none(mock_model_fixture.model)
    assert_is_not_none(mock_model_fixture.optimizer)
    assert_is_not_none(mock_model_fixture.dataloader)
    
    # Test model forward pass
    dummy_input = torch.randn(1, 10)
    output = mock_model_fixture.model(dummy_input)
    assert_is_not_none(output)
    
    # Test dataloader iteration
    batch_count = 0
    for batch in mock_model_fixture.dataloader:
        batch_count += 1
        assert_equal(len(batch), 2)  # (data, labels)
        if batch_count >= 2:  # Test only first 2 batches
            break
    assert_greater(batch_count, 0)


@use_fixtures('config_fixture')
def test_with_config_fixture(config_fixture=None):
    """Test using the config_fixture."""
    # Get the config
    config = config_fixture.get_config()
    assert_is_not_none(config)
    assert_in('model_name', config)
    assert_in('batch_size', config)
    
    # Test getting specific config values
    model_name = config_fixture.get_config('model_name')
    assert_is_not_none(model_name)
    
    batch_size = config_fixture.get_config('batch_size')
    assert_greater(batch_size, 0)
    
    # Test updating config
    config_fixture.update_config({'new_param': 'new_value'})
    new_param = config_fixture.get_config('new_param')
    assert_equal(new_param, 'new_value')


def test_combined_fixtures():
    """Test using multiple fixtures together."""
    with fixture_context('temp_dir', 'tensor_fixture', 'config_fixture') as fixtures:
        # Use temp directory
        temp_dir_path = fixtures['temp_dir']
        config = fixtures['config_fixture']
        tensor_fixture = fixtures['tensor_fixture']
        
        # Create a config file in temp directory
        config_file = os.path.join(temp_dir_path, "config.json")
        with open(config_file, "w") as f:
            import json
            json.dump(config.get_config(), f)
        
        # Verify file was created
        assert_true(os.path.exists(config_file))
        
        # Use tensor fixture
        tensor = tensor_fixture.get_tensor("tensor_0")
        assert_is_not_none(tensor)


def test_fixture_with_actual_model_training_scenario():
    """Test fixture usage in a scenario similar to model training."""
    with fixture_context('mock_model_fixture', 'config_fixture', 'tensor_fixture') as fixtures:
        model_fixture = fixtures['mock_model_fixture']
        config = fixtures['config_fixture']
        tensor_fx = fixtures['tensor_fixture']
        
        # Get model and config
        model = model_fixture.model
        optimizer = model_fixture.optimizer
        cfg = config.get_config()
        
        # Simulate a training step
        dummy_data, dummy_labels = next(iter(model_fixture.dataloader))
        
        # Forward pass
        outputs = model(dummy_data.float())
        
        # Verify outputs shape matches expected
        assert_equal(outputs.shape[0], dummy_data.shape[0])  # Batch dimension matches
        
        # Simulate loss calculation (just using MSE for demo)
        loss_fn = torch.nn.MSELoss()
        dummy_targets = torch.randn_like(outputs)
        loss = loss_fn(outputs, dummy_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Verify loss is a scalar
        assert_true(loss.dim() == 0)


def run_example_tests():
    """Run the example tests."""
    print("=" * 60)
    print("RUNNING FIXTURE EXAMPLE TESTS")
    print("=" * 60)
    
    # Run tests individually to demonstrate fixture usage
    tests = [
        test_with_temp_dir_fixture,
        lambda: test_with_tensor_fixture(tensor_fixture()),
        lambda: test_with_mock_model_fixture(mock_model_fixture()),
        lambda: test_with_config_fixture(config_fixture()),
        test_combined_fixtures,
        test_fixture_with_actual_model_training_scenario
    ]
    
    passed = 0
    failed = 0
    
    for i, test_func in enumerate(tests):
        try:
            print(f"Running test {i+1}: {test_func.__name__ if hasattr(test_func, '__name__') else f'test_{i+1}'}...", end="")
            test_func()
            print(" [PASS]")
            passed += 1
        except Exception as e:
            print(f" [FAIL]: {e}")
            failed += 1
    
    print("=" * 60)
    print("EXAMPLE TEST SUMMARY")
    print(f"Total: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    run_example_tests()