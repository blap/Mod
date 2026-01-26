"""
Standalone test script to verify the enhanced test utilities work properly
"""
import sys
import os

# Add the src directory to the path to import test_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'inference_pio'))

from test_utils import (
    assert_true, assert_false, assert_equal, assert_not_equal,
    assert_is_none, assert_is_not_none, assert_in, assert_not_in,
    assert_greater, assert_less, assert_raises, assert_is_instance,
    assert_greater_equal, assert_less_equal, assert_between,
    assert_is, assert_is_not, assert_almost_equal, assert_dict_contains,
    assert_list_contains, assert_tuple_contains, assert_set_contains,
    assert_positive, assert_negative, assert_non_negative, assert_non_positive,
    assert_zero, assert_not_zero, assert_close, assert_is_empty,
    assert_not_empty, assert_length, assert_min_length, assert_max_length,
    assert_items_equal, assert_sorted, assert_callable, assert_iterable, 
    assert_has_attr, assert_finite, assert_starts_with, assert_ends_with,
    assert_has_key, assert_has_value, assert_is_subclass, 
    assert_not_is_instance, assert_not_is_subclass
)

def test_basic_assertions():
    """Test basic assertion functions"""
    print("Testing basic assertions...")
    
    # Test assert_true
    assert_true(True, "True should pass")
    print("PASS assert_true works")

    # Test assert_false
    assert_false(False, "False should pass")
    print("PASS assert_false works")

    # Test assert_equal
    assert_equal(1, 1, "1 should equal 1")
    print("PASS assert_equal works")

    # Test assert_not_equal
    assert_not_equal(1, 2, "1 should not equal 2")
    print("PASS assert_not_equal works")

    # Test assert_is_none
    assert_is_none(None, "None should be None")
    print("PASS assert_is_none works")

    # Test assert_is_not_none
    assert_is_not_none("hello", "String should not be None")
    print("PASS assert_is_not_none works")

    # Test assert_in
    assert_in("a", ["a", "b", "c"], "a should be in list")
    print("PASS assert_in works")

    # Test assert_not_in
    assert_not_in("d", ["a", "b", "c"], "d should not be in list")
    print("PASS assert_not_in works")

    # Test assert_greater
    assert_greater(5, 3, "5 should be greater than 3")
    print("PASS assert_greater works")

    # Test assert_less
    assert_less(3, 5, "3 should be less than 5")
    print("PASS assert_less works")


def test_extended_assertions():
    """Test extended assertion functions"""
    print("\nTesting extended assertions...")
    
    # Test assert_greater_equal
    assert_greater_equal(5, 5, "5 should be greater than or equal to 5")
    assert_greater_equal(6, 5, "6 should be greater than or equal to 5")
    print("PASS assert_greater_equal works")

    # Test assert_less_equal
    assert_less_equal(5, 5, "5 should be less than or equal to 5")
    assert_less_equal(4, 5, "4 should be less than or equal to 5")
    print("PASS assert_less_equal works")

    # Test assert_between
    assert_between(5, 1, 10, "5 should be between 1 and 10")
    print("PASS assert_between works")

    # Test assert_is
    a = [1, 2, 3]
    b = a
    assert_is(a, b, "a and b should be the same object")
    print("PASS assert_is works")

    # Test assert_is_not
    c = [1, 2, 3]
    assert_is_not(a, c, "a and c should be different objects")
    print("PASS assert_is_not works")

    # Test assert_almost_equal
    assert_almost_equal(1.0000001, 1.0000002, places=5, message="Values should be almost equal")
    print("PASS assert_almost_equal works")

    # Test assert_dict_contains
    assert_dict_contains({"a": 1, "b": 2}, "a", "Dict should contain key 'a'")
    print("PASS assert_dict_contains works")

    # Test assert_list_contains
    assert_list_contains([1, 2, 3], 2, "List should contain 2")
    print("PASS assert_list_contains works")

    # Test assert_tuple_contains
    assert_tuple_contains((1, 2, 3), 2, "Tuple should contain 2")
    print("PASS assert_tuple_contains works")

    # Test assert_set_contains
    assert_set_contains({1, 2, 3}, 2, "Set should contain 2")
    print("PASS assert_set_contains works")

    # Test assert_positive
    assert_positive(5, "5 should be positive")
    print("PASS assert_positive works")

    # Test assert_negative
    assert_negative(-5, "-5 should be negative")
    print("PASS assert_negative works")

    # Test assert_non_negative
    assert_non_negative(0, "0 should be non-negative")
    assert_non_negative(5, "5 should be non-negative")
    print("PASS assert_non_negative works")

    # Test assert_non_positive
    assert_non_positive(0, "0 should be non-positive")
    assert_non_positive(-5, "-5 should be non-positive")
    print("PASS assert_non_positive works")

    # Test assert_zero
    assert_zero(0, "0 should be zero")
    print("PASS assert_zero works")

    # Test assert_not_zero
    assert_not_zero(5, "5 should not be zero")
    print("PASS assert_not_zero works")

    # Test assert_close
    assert_close(1.0, 1.0000001, rel_tol=1e-5, message="Values should be close")
    print("PASS assert_close works")

    # Test assert_is_empty
    assert_is_empty([], "Empty list should be empty")
    print("PASS assert_is_empty works")

    # Test assert_not_empty
    assert_not_empty([1], "Non-empty list should not be empty")
    print("PASS assert_not_empty works")

    # Test assert_length
    assert_length([1, 2, 3], 3, "List should have length 3")
    print("PASS assert_length works")

    # Test assert_min_length
    assert_min_length([1, 2, 3], 2, "List should have minimum length 2")
    print("PASS assert_min_length works")

    # Test assert_max_length
    assert_max_length([1, 2, 3], 5, "List should have maximum length 5")
    print("PASS assert_max_length works")

    # Test assert_items_equal
    assert_items_equal([1, 2, 3], [3, 2, 1], "Lists should have same items")
    print("PASS assert_items_equal works")

    # Test assert_sorted
    assert_sorted([1, 2, 3, 4], "List should be sorted")
    print("PASS assert_sorted works")

    # Test assert_callable
    assert_callable(lambda x: x, "Lambda should be callable")
    print("PASS assert_callable works")

    # Test assert_iterable
    assert_iterable([1, 2, 3], "List should be iterable")
    print("PASS assert_iterable works")

    # Test assert_has_attr
    class TestClass:
        attr = 1
    assert_has_attr(TestClass, 'attr', "TestClass should have attr")
    print("PASS assert_has_attr works")

    # Test assert_finite
    assert_finite(5.0, "Float should be finite")
    print("PASS assert_finite works")


def test_string_assertions():
    """Test string-specific assertion functions"""
    print("\nTesting string assertions...")

    # Test assert_starts_with
    assert_starts_with("hello world", "hello", "String should start with 'hello'")
    print("PASS assert_starts_with works")

    # Test assert_ends_with
    assert_ends_with("hello world", "world", "String should end with 'world'")
    print("PASS assert_ends_with works")


def test_dict_assertions():
    """Test dictionary-specific assertion functions"""
    print("\nTesting dictionary assertions...")

    # Test assert_has_key
    assert_has_key({"a": 1, "b": 2}, "a", "Dict should have key 'a'")
    print("PASS assert_has_key works")

    # Test assert_has_value
    assert_has_value({"a": 1, "b": 2}, 1, "Dict should have value 1")
    print("PASS assert_has_value works")


def test_type_assertions():
    """Test type-specific assertion functions"""
    print("\nTesting type assertions...")

    # Test assert_is_subclass
    assert_is_subclass(list, object, "list should be subclass of object")
    print("PASS assert_is_subclass works")

    # Test assert_not_is_instance
    assert_not_is_instance(5, str, "5 should not be instance of str")
    print("PASS assert_not_is_instance works")

    # Test assert_not_is_subclass
    assert_not_is_subclass(int, str, "int should not be subclass of str")
    print("PASS assert_not_is_subclass works")


def test_exception_assertions():
    """Test exception-specific assertion functions"""
    print("\nTesting exception assertions...")

    # Test assert_raises
    def raise_value_error():
        raise ValueError("Test error")

    assert_raises(ValueError, raise_value_error)
    print("PASS assert_raises works")


def run_all_tests():
    """Run all test functions"""
    print("Running tests for enhanced test utilities...\n")

    try:
        test_basic_assertions()
        test_extended_assertions()
        test_string_assertions()
        test_dict_assertions()
        test_type_assertions()
        test_exception_assertions()

        print("\nSUCCESS: All tests passed! Enhanced test utilities are working correctly.")
        return True
    except Exception as e:
        print(f"\nERROR: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        sys.exit(1)