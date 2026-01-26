# Example Tests for Inference-PIO

## Unit Test Example

Unit tests focus on individual functions or methods in isolation:

```python
from src.inference_pio.test_utils import assert_equal, assert_true, assert_is_instance, run_tests

def test_addition_function():
    """Test a simple addition function."""
    def add(a, b):
        return a + b
    
    result = add(2, 3)
    assert_equal(result, 5, "Addition should return correct result")
    assert_is_instance(result, int, "Result should be an integer")

def test_string_formatting():
    """Test string formatting functionality."""
    template = "Hello, {}!"
    name = "World"
    result = template.format(name)
    
    assert_equal(result, "Hello, World!", "String formatting should work correctly")
    assert_true(len(result) > 0, "Formatted string should not be empty")

if __name__ == '__main__':
    run_tests([test_addition_function, test_string_formatting])
```

## Integration Test Example

Integration tests verify how multiple components work together:

```python
from src.inference_pio.test_utils import assert_true, assert_is_not_none, run_tests

class ComponentA:
    def __init__(self):
        self.data = []
    
    def add_item(self, item):
        self.data.append(item)
        return True

class ComponentB:
    def __init__(self, source_component):
        self.source = source_component
    
    def get_count(self):
        return len(self.source.data)

def test_component_integration():
    """Test that ComponentA and ComponentB work together."""
    comp_a = ComponentA()
    comp_b = ComponentB(comp_a)
    
    # Add an item using ComponentA
    success = comp_a.add_item("test_item")
    
    # Verify ComponentB can access the data
    count = comp_b.get_count()
    
    assert_true(success, "Adding item should succeed")
    assert_true(count == 1, "ComponentB should see the item added by ComponentA")
    assert_is_not_none(comp_a.data, "ComponentA data should not be None")

if __name__ == '__main__':
    run_tests([test_component_integration])
```

## Performance Test Example

Performance tests measure execution time and resource usage:

```python
import time
from src.inference_pio.test_utils import assert_less, run_tests

def performance_test_sorting():
    """Test sorting performance with different data sizes."""
    def sort_performance_test(size):
        # Create random data to sort
        import random
        data = [random.randint(1, 1000) for _ in range(size)]
        
        start_time = time.time()
        sorted_data = sorted(data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        return execution_time, len(sorted_data)
    
    # Test with different sizes
    sizes = [100, 1000, 5000]
    max_allowed_time_per_item = 0.0001  # 0.1ms per item
    
    for size in sizes:
        exec_time, result_len = sort_performance_test(size)
        expected_time_limit = size * max_allowed_time_per_item
        
        assert_less(exec_time, expected_time_limit, 
                  f"Sorting {size} items took too long: {exec_time:.4f}s")
        assert_equal(result_len, size, "Sorted result should have same length as input")

if __name__ == '__main__':
    run_tests([performance_test_sorting])
```

## Exception Testing Example

Test that appropriate exceptions are raised:

```python
from src.inference_pio.test_utils import assert_raises, assert_true, run_tests

def test_division_by_zero():
    """Test that division by zero raises appropriate exception."""
    def divide(a, b):
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return a / b
    
    # Test normal division
    result = divide(10, 2)
    assert_true(result == 5, "Normal division should work")
    
    # Test division by zero raises exception
    assert_raises(ZeroDivisionError, divide, 10, 0)

if __name__ == '__main__':
    run_tests([test_division_by_zero])
```

## Model-Specific Test Example

Example of testing a model-specific functionality:

```python
from src.inference_pio.test_utils import assert_equal, assert_true, run_tests

def test_model_configuration():
    """Test model configuration functionality."""
    # Simulated model configuration class
    class ModelConfig:
        def __init__(self):
            self.use_flash_attention = True
            self.batch_size = 1
            self.max_length = 512
        
        def update_batch_size(self, new_size):
            if new_size <= 0:
                raise ValueError("Batch size must be positive")
            self.batch_size = new_size
    
    config = ModelConfig()
    
    # Test initial values
    assert_true(config.use_flash_attention, "Flash attention should be enabled by default")
    assert_equal(config.batch_size, 1, "Default batch size should be 1")
    assert_equal(config.max_length, 512, "Default max length should be 512")
    
    # Test updating batch size
    config.update_batch_size(4)
    assert_equal(config.batch_size, 4, "Batch size should be updated to 4")
    
    # Test error case
    try:
        config.update_batch_size(-1)
        assert_true(False, "Should have raised ValueError for negative batch size")
    except ValueError:
        pass  # Expected behavior

if __name__ == '__main__':
    run_tests([test_model_configuration])
```

## Complete Test Suite Example

A complete test suite combining different types of tests:

```python
from src.inference_pio.test_utils import (
    assert_equal, assert_true, assert_false, 
    assert_is_none, assert_in, assert_greater,
    run_tests
)

# Unit tests
def test_math_operations():
    """Test basic mathematical operations."""
    assert_equal(2 + 2, 4)
    assert_true(5 > 3)
    assert_false(5 < 3)

def test_list_operations():
    """Test list manipulation operations."""
    items = ['a', 'b', 'c']
    assert_in('b', items)
    assert_equal(len(items), 3)

# Integration tests  
def test_data_processing_pipeline():
    """Test a simple data processing pipeline."""
    # Simulate a simple pipeline
    raw_data = "hello world"
    processed = raw_data.upper().replace(" ", "_")
    expected = "HELLO_WORLD"
    
    assert_equal(processed, expected)

# Performance tests
def test_string_operations_performance():
    """Test performance of string operations."""
    import time
    
    large_string = "x" * 10000
    
    start = time.time()
    result = large_string.replace("x", "y")
    end = time.time()
    
    # Verify correctness
    assert_true(all(c == 'y' for c in result))
    assert_equal(len(result), len(large_string))
    
    # Verify performance (should be fast for this operation)
    assert_true(end - start < 0.1, "String replacement should be fast")

if __name__ == '__main__':
    # Run all tests in the suite
    run_tests([
        test_math_operations,
        test_list_operations, 
        test_data_processing_pipeline,
        test_string_operations_performance
    ])
```