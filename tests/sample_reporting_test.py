"""
Sample test demonstrating the comprehensive test reporting system.
"""

import pytest
import time
import random


def test_fast_operation():
    """A fast test that should pass."""
    time.sleep(0.01)  # Simulate a small amount of work
    assert 1 + 1 == 2


def test_medium_operation():
    """A medium-duration test that should pass."""
    time.sleep(0.05)  # Simulate more work
    result = sum(range(100))
    assert result == 4950


@pytest.mark.slow
def test_slow_operation():
    """A slow test that should pass."""
    time.sleep(0.2)  # Simulate significant work
    # Perform a more complex calculation
    result = 1
    for i in range(1, 11):
        result *= i  # Calculate factorial of 10
    assert result == 3628800


def test_failing_operation():
    """A test that will fail to demonstrate error reporting."""
    time.sleep(0.02)
    assert 2 + 2 == 5, "This is intentionally wrong to show error reporting"


@pytest.mark.unit
def test_unit_feature():
    """A unit test with a specific marker."""
    time.sleep(0.03)
    data = {"key": "value"}
    assert "key" in data


@pytest.mark.integration
def test_integration_component():
    """An integration test with a specific marker."""
    time.sleep(0.04)
    # Simulate integration between components
    comp1_result = "processed"
    comp2_input = comp1_result.upper()
    assert comp2_input == "PROCESSED"


@pytest.mark.parametrize("input_val,expected", [(1, 2), (2, 4), (3, 6)])
def test_parametrized(input_val, expected):
    """A parametrized test to show multiple test cases."""
    time.sleep(0.01)
    result = input_val * 2
    assert result == expected


def test_with_exception():
    """A test that raises an exception."""
    time.sleep(0.02)
    with pytest.raises(ValueError):
        raise ValueError("This is an intentional error for testing")


@pytest.mark.performance
def test_performance_metric():
    """A performance test to demonstrate metrics collection."""
    start_time = time.time()
    # Simulate some computational work
    data = [random.random() for _ in range(1000)]
    processed = [x * 2 for x in data]
    end_time = time.time()
    
    duration = end_time - start_time
    assert len(processed) == 1000
    # Verify that the test took some measurable time
    assert duration >= 0  # Should be a positive duration


if __name__ == "__main__":
    # Run tests directly with reporting
    import subprocess
    import sys
    
    # Run this test file with the reporting system
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--generate-reports",
        "--reports-dir=sample_reports"
    ])
    
    if result.returncode == 0:
        print("Sample tests completed successfully with reports generated in 'sample_reports' directory")
    else:
        print(f"Sample tests failed with return code: {result.returncode}")