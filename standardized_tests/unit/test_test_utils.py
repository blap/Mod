"""
Test script to verify the enhanced test utilities work properly
"""
import sys
import os
from typing import Any, Callable, Dict, List, Tuple, Type, Union

# Add the src directory to the path to import test_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from inference_pio.test_utils import (
    assert_true, assert_false, assert_equal, assert_not_equal,
    assert_is_none, assert_is_not_none, assert_in, assert_not_in,
    assert_greater, assert_less, assert_raises, assert_is_instance,
    assert_greater_equal, assert_less_equal, assert_between,
    assert_is, assert_is_not, assert_almost_equal, assert_dict_contains,
    assert_list_contains, assert_tuple_contains, assert_set_contains,
    assert_positive, assert_negative, assert_non_negative, assert_non_positive,
    assert_zero, assert_not_zero, assert_close, assert_is_empty,
    assert_not_empty, assert_length, assert_min_length, assert_max_length,
    assert_tensor_equal, assert_tensor_close, assert_tensor_shape,
    assert_tensor_dtype, assert_tensor_all, assert_tensor_any,
    assert_starts_with, assert_ends_with, assert_has_key,
    assert_has_value, assert_is_subclass, assert_not_is_instance,
    assert_not_is_subclass, assert_finite, assert_items_equal,
    assert_sorted, assert_callable, assert_iterable, assert_has_attr
)

def test_basic_assertions() -> None:
    """Test basic assertion functions"""
    print("Testing basic assertions...")

    # Test assert_true
    assert_true(True, "True should pass")
    print("✓ assert_true works")

    # Test assert_false
    assert_false(False, "False should pass")
    print("✓ assert_false works")

    # Test assert_equal
    assert_equal(1, 1, "1 should equal 1")
    print("✓ assert_equal works")

    # Test assert_not_equal
    assert_not_equal(1, 2, "1 should not equal 2")
    print("✓ assert_not_equal works")

    # Test assert_is_none
    assert_is_none(None, "None should be None")
    print("✓ assert_is_none works")

    # Test assert_is_not_none
    assert_is_not_none("hello", "String should not be None")
    print("✓ assert_is_not_none works")

    # Test assert_in
    assert_in("a", ["a", "b", "c"], "a should be in list")
    print("✓ assert_in works")

    # Test assert_not_in
    assert_not_in("d", ["a", "b", "c"], "d should not be in list")
    print("✓ assert_not_in works")

    # Test assert_greater
    assert_greater(5, 3, "5 should be greater than 3")
    print("✓ assert_greater works")

    # Test assert_less
    assert_less(3, 5, "3 should be less than 5")
    print("✓ assert_less works")


def test_extended_assertions() -> None:
    """Test extended assertion functions"""
    print("\nTesting extended assertions...")

    # Test assert_greater_equal
    assert_greater_equal(5, 5, "5 should be greater than or equal to 5")
    assert_greater_equal(6, 5, "6 should be greater than or equal to 5")
    print("✓ assert_greater_equal works")

    # Test assert_less_equal
    assert_less_equal(5, 5, "5 should be less than or equal to 5")
    assert_less_equal(4, 5, "4 should be less than or equal to 5")
    print("✓ assert_less_equal works")

    # Test assert_between
    assert_between(5, 1, 10, "5 should be between 1 and 10")
    print("✓ assert_between works")

    # Test assert_is
    a: List[int] = [1, 2, 3]
    b: List[int] = a
    assert_is(a, b, "a and b should be the same object")
    print("✓ assert_is works")

    # Test assert_is_not
    c: List[int] = [1, 2, 3]
    assert_is_not(a, c, "a and c should be different objects")
    print("✓ assert_is_not works")

    # Test assert_almost_equal
    assert_almost_equal(1.0000001, 1.0000002, places=5, message="Values should be almost equal")
    print("✓ assert_almost_equal works")

    # Test assert_dict_contains
    assert_dict_contains({"a": 1, "b": 2}, "a", "Dict should contain key 'a'")
    print("✓ assert_dict_contains works")

    # Test assert_list_contains
    assert_list_contains([1, 2, 3], 2, "List should contain 2")
    print("✓ assert_list_contains works")

    # Test assert_tuple_contains
    assert_tuple_contains((1, 2, 3), 2, "Tuple should contain 2")
    print("✓ assert_tuple_contains works")

    # Test assert_set_contains
    assert_set_contains({1, 2, 3}, 2, "Set should contain 2")
    print("✓ assert_set_contains works")

    # Test assert_positive
    assert_positive(5, "5 should be positive")
    print("✓ assert_positive works")

    # Test assert_negative
    assert_negative(-5, "-5 should be negative")
    print("✓ assert_negative works")

    # Test assert_non_negative
    assert_non_negative(0, "0 should be non-negative")
    assert_non_negative(5, "5 should be non-negative")
    print("✓ assert_non_negative works")

    # Test assert_non_positive
    assert_non_positive(0, "0 should be non-positive")
    assert_non_positive(-5, "-5 should be non-positive")
    print("✓ assert_non_positive works")

    # Test assert_zero
    assert_zero(0, "0 should be zero")
    print("✓ assert_zero works")

    # Test assert_not_zero
    assert_not_zero(5, "5 should not be zero")
    print("✓ assert_not_zero works")

    # Test assert_close
    assert_close(1.0, 1.0000001, rel_tol=1e-5, message="Values should be close")
    print("✓ assert_close works")

    # Test assert_is_empty
    assert_is_empty([], "Empty list should be empty")
    print("✓ assert_is_empty works")

    # Test assert_not_empty
    assert_not_empty([1], "Non-empty list should not be empty")
    print("✓ assert_not_empty works")

    # Test assert_length
    assert_length([1, 2, 3], 3, "List should have length 3")
    print("✓ assert_length works")

    # Test assert_min_length
    assert_min_length([1, 2, 3], 2, "List should have minimum length 2")
    print("✓ assert_min_length works")

    # Test assert_max_length
    assert_max_length([1, 2, 3], 5, "List should have maximum length 5")
    print("✓ assert_max_length works")

    # Test assert_items_equal
    assert_items_equal([1, 2, 3], [3, 2, 1], "Lists should have same items")
    print("✓ assert_items_equal works")

    # Test assert_sorted
    assert_sorted([1, 2, 3, 4], "List should be sorted")
    print("✓ assert_sorted works")

    # Test assert_callable
    assert_callable(lambda x: x, "Lambda should be callable")
    print("✓ assert_callable works")

    # Test assert_iterable
    assert_iterable([1, 2, 3], "List should be iterable")
    print("✓ assert_iterable works")

    # Test assert_has_attr
    class TestClass:
        attr = 1
    assert_has_attr(TestClass, 'attr', "TestClass should have attr")
    print("✓ assert_has_attr works")


def test_string_assertions() -> None:
    """Test string-specific assertion functions"""
    print("\nTesting string assertions...")

    # Test assert_starts_with
    assert_starts_with("hello world", "hello", "String should start with 'hello'")
    print("✓ assert_starts_with works")

    # Test assert_ends_with
    assert_ends_with("hello world", "world", "String should end with 'world'")
    print("✓ assert_ends_with works")


def test_dict_assertions() -> None:
    """Test dictionary-specific assertion functions"""
    print("\nTesting dictionary assertions...")

    # Test assert_has_key
    assert_has_key({"a": 1, "b": 2}, "a", "Dict should have key 'a'")
    print("✓ assert_has_key works")

    # Test assert_has_value
    assert_has_value({"a": 1, "b": 2}, 1, "Dict should have value 1")
    print("✓ assert_has_value works")


def test_type_assertions() -> None:
    """Test type-specific assertion functions"""
    print("\nTesting type assertions...")

    # Test assert_is_subclass
    assert_is_subclass(list, object, "list should be subclass of object")
    print("✓ assert_is_subclass works")

    # Test assert_not_is_instance
    assert_not_is_instance(5, str, "5 should not be instance of str")
    print("✓ assert_not_is_instance works")

    # Test assert_not_is_subclass
    assert_not_is_subclass(int, str, "int should not be subclass of str")
    print("✓ assert_not_is_subclass works")


def test_exception_assertions() -> None:
    """Test exception-specific assertion functions"""
    print("\nTesting exception assertions...")

    # Test assert_raises
    def raise_value_error() -> None:
        raise ValueError("Test error")

    assert_raises(ValueError, raise_value_error)
    print("✓ assert_raises works")


def test_tensor_assertions() -> None:
    """Test tensor-specific assertion functions"""
    print("\nTesting tensor assertions...")

    import torch

    # Test assert_tensor_equal
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([1, 2, 3])
    assert_tensor_equal(t1, t2, "Tensors should be equal")
    print("✓ assert_tensor_equal works")

    # Test assert_tensor_close
    t3 = torch.tensor([1.0, 2.0, 3.0])
    t4 = torch.tensor([1.0001, 2.0001, 3.0001])
    assert_tensor_close(t3, t4, rtol=1e-3, atol=1e-5, message="Tensors should be close")
    print("✓ assert_tensor_close works")

    # Test assert_tensor_shape
    t5 = torch.tensor([[1, 2], [3, 4]])
    assert_tensor_shape(t5, (2, 2), "Tensor should have shape (2, 2)")
    print("✓ assert_tensor_shape works")

    # Test assert_tensor_dtype
    t6 = torch.tensor([1, 2, 3], dtype=torch.int32)
    assert_tensor_dtype(t6, torch.int32, "Tensor should have dtype int32")
    print("✓ assert_tensor_dtype works")

    # Test assert_tensor_all
    t7 = torch.tensor([1, 2, 3, 4])
    assert_tensor_all(t7, lambda x: x > 0, "All elements should be positive")
    print("✓ assert_tensor_all works")

    # Test assert_tensor_any
    t8 = torch.tensor([-1, 0, 1, 2])
    assert_tensor_any(t8, lambda x: x > 0, "Any element should be positive")
    print("✓ assert_tensor_any works")


def run_all_tests() -> bool:
    """Run all test functions"""
    print("Running tests for enhanced test utilities...\n")

    try:
        test_basic_assertions()
        test_extended_assertions()
        test_string_assertions()
        test_dict_assertions()
        test_type_assertions()
        test_exception_assertions()
        test_tensor_assertions()

        print("\n✅ All tests passed! Enhanced test utilities are working correctly.")
        return True
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)