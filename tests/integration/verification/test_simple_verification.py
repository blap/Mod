"""
Final verification test for the enhanced test utilities
"""
import sys
import os
import torch
import numpy as np

# Add the src directory to the path to import test_utils directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'inference_pio'))

# Import the test utilities directly from the file
exec(open('tests.utils.test_utils.py').read())

def test_all_enhanced_functions():
    """Test all enhanced functions to ensure they work properly"""
    print("Testing all enhanced test utility functions...")

    # Basic assertions
    assert_true(True, "Basic assert_true should work")
    assert_false(False, "Basic assert_false should work")
    assert_equal(1, 1, "Basic assert_equal should work")
    assert_not_equal(1, 2, "Basic assert_not_equal should work")
    assert_is_none(None, "Basic assert_is_none should work")
    assert_is_not_none("hello", "Basic assert_is_not_none should work")
    assert_in("a", ["a", "b", "c"], "Basic assert_in should work")
    assert_not_in("d", ["a", "b", "c"], "Basic assert_not_in should work")
    assert_greater(5, 3, "Basic assert_greater should work")
    assert_less(3, 5, "Basic assert_less should work")
    print("PASS Basic assertions work")

    # Extended assertions
    assert_greater_equal(5, 5, "Extended assert_greater_equal should work")
    assert_less_equal(5, 5, "Extended assert_less_equal should work")
    assert_between(5, 1, 10, "Extended assert_between should work")
    a = [1, 2, 3]
    b = a
    assert_is(a, b, "Extended assert_is should work")
    c = [1, 2, 3]
    assert_is_not(a, c, "Extended assert_is_not should work")
    assert_almost_equal(1.0000001, 1.0000002, places=5, message="Extended assert_almost_equal should work")
    assert_dict_contains({"a": 1, "b": 2}, "a", "Extended assert_dict_contains should work")
    assert_list_contains([1, 2, 3], 2, "Extended assert_list_contains should work")
    assert_tuple_contains((1, 2, 3), 2, "Extended assert_tuple_contains should work")
    assert_set_contains({1, 2, 3}, 2, "Extended assert_set_contains should work")
    assert_positive(5, "Extended assert_positive should work")
    assert_negative(-5, "Extended assert_negative should work")
    assert_non_negative(0, "Extended assert_non_negative should work")
    assert_non_positive(0, "Extended assert_non_positive should work")
    assert_zero(0, "Extended assert_zero should work")
    assert_not_zero(5, "Extended assert_not_zero should work")
    assert_close(1.0, 1.0000001, rel_tol=1e-5, message="Extended assert_close should work")
    assert_is_empty([], "Extended assert_is_empty should work")
    assert_not_empty([1], "Extended assert_not_empty should work")
    assert_length([1, 2, 3], 3, "Extended assert_length should work")
    assert_min_length([1, 2, 3], 2, "Extended assert_min_length should work")
    assert_max_length([1, 2, 3], 5, "Extended assert_max_length should work")
    assert_items_equal([1, 2, 3], [3, 2, 1], "Extended assert_items_equal should work")
    assert_sorted([1, 2, 3, 4], "Extended assert_sorted should work")
    assert_callable(lambda x: x, "Extended assert_callable should work")
    assert_iterable([1, 2, 3], "Extended assert_iterable should work")

    class TestClass:
        attr = 1
    assert_has_attr(TestClass, 'attr', "Extended assert_has_attr should work")
    assert_finite(5.0, "Extended assert_finite should work")
    print("PASS Extended assertions work")

    # String assertions
    assert_starts_with("hello world", "hello", "String assert_starts_with should work")
    assert_ends_with("hello world", "world", "String assert_ends_with should work")
    print("PASS String assertions work")

    # Dictionary assertions
    assert_has_key({"a": 1, "b": 2}, "a", "Dictionary assert_has_key should work")
    assert_has_value({"a": 1, "b": 2}, 1, "Dictionary assert_has_value should work")
    print("PASS Dictionary assertions work")

    # Type assertions
    assert_is_subclass(list, object, "Type assert_is_subclass should work")
    assert_not_is_instance(5, str, "Type assert_not_is_instance should work")
    assert_not_is_subclass(int, str, "Type assert_not_is_subclass should work")
    print("PASS Type assertions work")

    # Exception assertions
    def raise_value_error():
        raise ValueError("Test error")

    assert_raises(ValueError, raise_value_error)
    print("PASS Exception assertions work")

    # Tensor assertions
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([1.0, 2.0, 3.0])
    assert_tensor_equal(t1, t2, "Tensor assert_tensor_equal should work")

    t3 = torch.tensor([1.0, 2.0, 3.0])
    t4 = torch.tensor([1.0001, 2.0001, 3.0001])
    assert_tensor_close(t3, t4, rtol=1e-3, message="Tensor assert_tensor_close should work")

    t5 = torch.tensor([[1, 2], [3, 4]])
    assert_tensor_shape(t5, (2, 2), "Tensor assert_tensor_shape should work")

    t6 = torch.tensor([1, 2, 3], dtype=torch.int32)
    assert_tensor_dtype(t6, torch.int32, "Tensor assert_tensor_dtype should work")

    assert_tensor_all(t5, lambda x: x > 0, "Tensor assert_tensor_all should work")
    assert_tensor_any(t5, lambda x: x > 2, "Tensor assert_tensor_any should work")
    print("PASS Tensor assertions work")

    # Advanced tensor assertions
    assert_tensor_all_elements_positive_in_sequence(torch.tensor([1.0, 2.0, 3.0]), message="Advanced tensor assertion should work")
    assert_tensor_any_elements_positive_in_sequence(torch.tensor([-1.0, 0.0, 1.0]), message="Advanced tensor assertion should work")
    assert_tensor_all_elements_finite_in_sequence(torch.tensor([1.0, 2.0, 3.0]), message="Advanced tensor assertion should work")
    print("PASS Advanced tensor assertions work")

    print("\nSUCCESS: All enhanced test utilities are working correctly!")
    print("The test_utils.py file has been successfully enhanced with comprehensive test utilities.")


if __name__ == "__main__":
    test_all_enhanced_functions()