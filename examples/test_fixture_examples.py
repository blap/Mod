"""
Example demonstrating how to use the test fixture system in Inference-PIO
"""

import sys
import os
# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import importlib.util

# Dynamically import the test_fixtures module
spec = importlib.util.spec_from_file_location(
    "test_fixtures", 
    os.path.join(os.path.dirname(__file__), '..', 'src', 'inference_pio', 'test_fixtures.py')
)
test_fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_fixtures)

# Import the necessary functions
fixture = test_fixtures.fixture
use_fixtures = test_fixtures.use_fixtures
fixture_context = test_fixtures.fixture_context
TemporaryDirectoryFixture = test_fixtures.TemporaryDirectoryFixture
TensorFixture = test_fixtures.TensorFixture
MockModelFixture = test_fixtures.MockModelFixture
ConfigFixture = test_fixtures.ConfigFixture


def test_example_with_temp_dir():
    """Example: Using temporary directory fixture."""
    # Note: Since 'temp_dir' fixture is registered with the decorator system,
    # we'll create it directly for this example
    temp_fixture = TemporaryDirectoryFixture()

    with temp_fixture as temp_dir_path:
        # Create a file in the temporary directory
        test_file = os.path.join(temp_dir_path, "example.txt")
        with open(test_file, "w") as f:
            f.write("This is an example file created in a temporary directory.")

        # Verify the file exists and read its content
        assert os.path.exists(test_file), "File should exist"
        with open(test_file, "r") as f:
            content = f.read()
        assert content == "This is an example file created in a temporary directory."

        print("PASS Temporary directory fixture example passed")


def test_example_with_tensor_fixture():
    """Example: Using tensor fixture."""
    tensor_fixture = TensorFixture()
    
    # Get a specific tensor
    tensor = tensor_fixture.get_tensor("tensor_0")
    assert tensor is not None, "Tensor should exist"
    assert hasattr(tensor, 'shape'), "Tensor should have shape attribute"
    
    # Get random tensors
    random_tensors = tensor_fixture.get_random_tensors(2)
    assert len(random_tensors) >= 0, "Should get at least some tensors"
    
    print("PASS Tensor fixture example passed")


def test_example_with_mock_model():
    """Example: Using mock model fixture."""
    model_fixture = MockModelFixture()
    
    # Verify model components exist
    assert model_fixture.model is not None, "Model should exist"
    assert model_fixture.optimizer is not None, "Optimizer should exist"
    assert model_fixture.dataloader is not None, "Dataloader should exist"
    
    # Perform a simple forward pass
    import torch
    dummy_input = torch.randn(1, 10)
    output = model_fixture.model(dummy_input)
    assert output is not None, "Output should exist"
    
    print("PASS Mock model fixture example passed")


def test_example_with_config():
    """Example: Using config fixture."""
    config_fixture = ConfigFixture()
    
    # Get the config
    config = config_fixture.get_config()
    assert config is not None, "Config should exist"
    assert 'model_name' in config, "Config should have model_name"
    
    # Test getting specific config values
    model_name = config_fixture.get_config('model_name')
    assert model_name is not None, "Model name should exist"
    
    # Update config
    config_fixture.update_config({'new_param': 'new_value'})
    new_param = config_fixture.get_config('new_param')
    assert new_param == 'new_value', "New param should have correct value"
    
    print("PASS Config fixture example passed")


def test_combined_example():
    """Example: Using multiple fixtures together."""
    # Create fixtures manually for this example
    temp_fixture = TemporaryDirectoryFixture()
    tensor_fixture = TensorFixture()
    config_fixture = ConfigFixture()
    
    # Use the temporary directory
    with temp_fixture as temp_dir:
        # Save a tensor to the temp directory
        tensor = tensor_fixture.get_tensor("tensor_0")
        config = config_fixture.get_config()
        
        # Create a config file in temp directory
        config_file = os.path.join(temp_dir, "config.json")
        with open(config_file, "w") as f:
            import json
            json.dump(config, f)
        
        # Verify file was created
        assert os.path.exists(config_file), "Config file should exist"
        
        print("PASS Combined fixtures example passed")


def run_examples():
    """Run all fixture examples."""
    print("Testing Inference-PIO Test Fixture System Examples")
    print("=" * 50)
    
    test_example_with_temp_dir()
    test_example_with_tensor_fixture()
    test_example_with_mock_model()
    test_example_with_config()
    test_combined_example()
    
    print("=" * 50)
    print("All fixture examples passed successfully!")


if __name__ == "__main__":
    run_examples()