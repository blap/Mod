"""
Enhanced test utilities for direct testing without external frameworks
"""

import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Callable, Type, Union
from enum import Enum


def assert_true(condition, message="Assertion failed"):
    """Assert that condition is True"""
    if not condition:
        raise AssertionError(message)


def assert_false(condition, message="Assertion failed"):
    """Assert that condition is False"""
    if condition:
        raise AssertionError(message)


def assert_equal(actual, expected, message="Values are not equal"):
    """Assert that actual equals expected"""
    if actual != expected:
        raise AssertionError(f"{message}: actual={actual}, expected={expected}")


def assert_not_equal(actual, expected, message="Values are equal but should not be"):
    """Assert that actual does not equal expected"""
    if actual == expected:
        raise AssertionError(f"{message}: actual={actual}, expected={expected}")


def assert_is_none(value, message="Value is not None"):
    """Assert that value is None"""
    if value is not None:
        raise AssertionError(f"{message}: value={value}")


def assert_is_not_none(value, message="Value is None"):
    """Assert that value is not None"""
    if value is None:
        raise AssertionError(message)


def assert_in(item, container, message="Item not found in container"):
    """Assert that item is in container"""
    if item not in container:
        raise AssertionError(f"{message}: item={item}, container={container}")


def assert_not_in(item, container, message="Item found in container but should not be"):
    """Assert that item is not in container"""
    if item in container:
        raise AssertionError(f"{message}: item={item}, container={container}")


def assert_greater(value, comparison, message="Value is not greater than comparison"):
    """Assert that value is greater than comparison"""
    if value <= comparison:
        raise AssertionError(f"{message}: value={value}, comparison={comparison}")


def assert_less(value, comparison, message="Value is not less than comparison"):
    """Assert that value is less than comparison"""
    if value >= comparison:
        raise AssertionError(f"{message}: value={value}, comparison={comparison}")


def assert_raises(exception_type, callable_func, *args, **kwargs):
    """Assert that calling the function raises the specified exception"""
    try:
        callable_func(*args, **kwargs)
        raise AssertionError(f"Expected {exception_type.__name__} but no exception was raised")
    except exception_type:
        pass  # Expected behavior
    except Exception as e:
        raise AssertionError(f"Expected {exception_type.__name__} but got {type(e).__name__}: {e}")


def assert_is_instance(obj, expected_class, message="Object is not instance of expected class"):
    """Assert that object is instance of expected class"""
    if not isinstance(obj, expected_class):
        raise AssertionError(f"{message}: expected={expected_class}, actual={type(obj)}")


def skip_test(reason="Test skipped"):
    """Skip a test with a reason"""
    raise SkipTestException(reason)


class SkipTestException(Exception):
    """Exception raised when a test is intentionally skipped"""
    pass


def run_test(test_func, test_name=None):
    """Run a single test function and report results"""
    if test_name is None:
        test_name = test_func.__name__

    try:
        print(f"Running test: {test_name}...", end="")
        test_func()
        print(" [PASS]")
        return True
    except SkipTestException as e:
        print(f" [SKIP]: {e}")
        return True
    except Exception as e:
        print(f" [FAIL]: {e}")
        return False


def run_tests(test_functions):
    """Run multiple test functions and report summary"""
    print("=" * 60)
    print("RUNNING TEST SUITE")
    print("=" * 60)

    passed = 0
    failed = 0
    skipped = 0

    for test_func in test_functions:
        try:
            test_name = test_func.__name__
            print(f"Running test: {test_name}...", end="")
            test_func()
            print(" [PASS]")
            passed += 1
        except SkipTestException as e:
            print(f" [SKIP]: {e}")
            skipped += 1
        except Exception as e:
            print(f" [FAIL]: {e}")
            failed += 1

    print("=" * 60)
    print("TEST SUMMARY")
    print(f"Total: {passed + failed + skipped}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print("=" * 60)

    return failed == 0


def assert_greater_equal(value, comparison, message="Value is not greater than or equal to comparison"):
    """Assert that value is greater than or equal to comparison"""
    if value < comparison:
        raise AssertionError(f"{message}: value={value}, comparison={comparison}")


def assert_less_equal(value, comparison, message="Value is not less than or equal to comparison"):
    """Assert that value is less than or equal to comparison"""
    if value > comparison:
        raise AssertionError(f"{message}: value={value}, comparison={comparison}")


def assert_between(value, lower_bound, upper_bound, message="Value is not between bounds"):
    """Assert that value is between lower and upper bounds (inclusive)"""
    if not (lower_bound <= value <= upper_bound):
        raise AssertionError(f"{message}: value={value}, bounds=[{lower_bound}, {upper_bound}]")


def assert_is(value, expected, message="Value is not identical to expected"):
    """Assert that value is identical to expected (using 'is' operator)"""
    if value is not expected:
        raise AssertionError(f"{message}: value={value}, expected={expected}")


def assert_is_not(value, expected, message="Value is identical to expected but should not be"):
    """Assert that value is not identical to expected (using 'is not' operator)"""
    if value is expected:
        raise AssertionError(f"{message}: value={value}, expected={expected}")


def assert_almost_equal(value, expected, places=7, message="Values are not almost equal"):
    """Assert that value is almost equal to expected (within decimal places)"""
    if not abs(value - expected) < 10**(-places):
        raise AssertionError(f"{message}: value={value}, expected={expected}, places={places}")


def assert_dict_contains(dictionary, key, message="Key not found in dictionary"):
    """Assert that dictionary contains the specified key"""
    if key not in dictionary:
        raise AssertionError(f"{message}: key={key}, dictionary={dictionary}")


def assert_list_contains(lst, item, message="Item not found in list"):
    """Assert that list contains the specified item"""
    if item not in lst:
        raise AssertionError(f"{message}: item={item}, list={lst}")


def assert_tuple_contains(tpl, item, message="Item not found in tuple"):
    """Assert that tuple contains the specified item"""
    if item not in tpl:
        raise AssertionError(f"{message}: item={item}, tuple={tpl}")


def assert_set_contains(st, item, message="Item not found in set"):
    """Assert that set contains the specified item"""
    if item not in st:
        raise AssertionError(f"{message}: item={item}, set={st}")


def assert_finite(value, message="Value is not finite"):
    """Assert that value is finite (not NaN or infinity)"""
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            raise AssertionError(f"{message}: value={value}")
    elif isinstance(value, torch.Tensor):
        if torch.isnan(value).any() or torch.isinf(value).any():
            raise AssertionError(f"{message}: tensor contains NaN or Inf values")
    else:
        raise TypeError(f"Unsupported type for assert_finite: {type(value)}")


def assert_positive(value, message="Value is not positive"):
    """Assert that value is positive"""
    if value <= 0:
        raise AssertionError(f"{message}: value={value}")


def assert_negative(value, message="Value is not negative"):
    """Assert that value is negative"""
    if value >= 0:
        raise AssertionError(f"{message}: value={value}")


def assert_non_negative(value, message="Value is negative"):
    """Assert that value is non-negative (>= 0)"""
    if value < 0:
        raise AssertionError(f"{message}: value={value}")


def assert_non_positive(value, message="Value is positive"):
    """Assert that value is non-positive (<= 0)"""
    if value > 0:
        raise AssertionError(f"{message}: value={value}")


def assert_zero(value, message="Value is not zero"):
    """Assert that value is zero"""
    if value != 0:
        raise AssertionError(f"{message}: value={value}")


def assert_not_zero(value, message="Value is zero but should not be"):
    """Assert that value is not zero"""
    if value == 0:
        raise AssertionError(f"{message}: value={value}")


def assert_close(value, expected, rel_tol=1e-09, abs_tol=0.0, message="Values are not close"):
    """Assert that value is close to expected within tolerance"""
    if not abs(value - expected) <= max(rel_tol * max(abs(value), abs(expected)), abs_tol):
        raise AssertionError(f"{message}: value={value}, expected={expected}, "
                             f"rel_tol={rel_tol}, abs_tol={abs_tol}")


def assert_is_empty(container, message="Container is not empty"):
    """Assert that container is empty"""
    if len(container) != 0:
        raise AssertionError(f"{message}: container={container}")


def assert_not_empty(container, message="Container is empty"):
    """Assert that container is not empty"""
    if len(container) == 0:
        raise AssertionError(f"{message}: container={container}")


def assert_length(container, expected_length, message="Container length is not as expected"):
    """Assert that container has the expected length"""
    actual_length = len(container)
    if actual_length != expected_length:
        raise AssertionError(f"{message}: actual_length={actual_length}, expected_length={expected_length}")


def assert_min_length(container, min_length, message="Container length is less than minimum"):
    """Assert that container has at least the minimum length"""
    if len(container) < min_length:
        raise AssertionError(f"{message}: length={len(container)}, min_length={min_length}")


def assert_max_length(container, max_length, message="Container length exceeds maximum"):
    """Assert that container has at most the maximum length"""
    if len(container) > max_length:
        raise AssertionError(f"{message}: length={len(container)}, max_length={max_length}")


def assert_items_equal(actual, expected, message="Items in containers are not equal"):
    """Assert that items in two containers are equal (order-independent)"""
    if sorted(actual) != sorted(expected):
        raise AssertionError(f"{message}: actual={actual}, expected={expected}")


def assert_sorted(container, message="Container is not sorted"):
    """Assert that container is sorted in ascending order"""
    if list(container) != sorted(container):
        raise AssertionError(f"{message}: container={container}")


def assert_callable(obj, message="Object is not callable"):
    """Assert that object is callable"""
    if not callable(obj):
        raise AssertionError(f"{message}: obj={obj}")


def assert_iterable(obj, message="Object is not iterable"):
    """Assert that object is iterable"""
    try:
        iter(obj)
    except TypeError:
        raise AssertionError(f"{message}: obj={obj}")


def assert_has_attr(obj, attr_name, message="Object does not have the specified attribute"):
    """Assert that object has the specified attribute"""
    if not hasattr(obj, attr_name):
        raise AssertionError(f"{message}: obj={obj}, attr_name={attr_name}")


def assert_tensor_equal(tensor1, tensor2, message="Tensors are not equal"):
    """Assert that two tensors are equal"""
    if not torch.equal(tensor1, tensor2):
        raise AssertionError(f"{message}: tensor1 and tensor2 differ")


def assert_tensor_close(tensor1, tensor2, rtol=1e-05, atol=1e-08, message="Tensors are not close"):
    """Assert that two tensors are close within tolerance"""
    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        raise AssertionError(f"{message}: tensor1 and tensor2 are not close")


def assert_tensor_shape(tensor, expected_shape, message="Tensor shape is not as expected"):
    """Assert that tensor has the expected shape"""
    if tensor.shape != expected_shape:
        raise AssertionError(f"{message}: actual_shape={tensor.shape}, expected_shape={expected_shape}")


def assert_tensor_dtype(tensor, expected_dtype, message="Tensor dtype is not as expected"):
    """Assert that tensor has the expected dtype"""
    if tensor.dtype != expected_dtype:
        raise AssertionError(f"{message}: actual_dtype={tensor.dtype}, expected_dtype={expected_dtype}")


def assert_tensor_all(tensor, condition_func, message="Not all tensor elements satisfy condition"):
    """Assert that all tensor elements satisfy the condition function"""
    if not condition_func(tensor).all():
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_any(tensor, condition_func, message="No tensor elements satisfy condition"):
    """Assert that any tensor element satisfies the condition function"""
    if not condition_func(tensor).any():
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_starts_with(text, prefix, message="Text does not start with prefix"):
    """Assert that text starts with the specified prefix"""
    if not text.startswith(prefix):
        raise AssertionError(f"{message}: text={text}, prefix={prefix}")


def assert_ends_with(text, suffix, message="Text does not end with suffix"):
    """Assert that text ends with the specified suffix"""
    if not text.endswith(suffix):
        raise AssertionError(f"{message}: text={text}, suffix={suffix}")


def assert_has_key(dictionary, key, message="Dictionary does not have the specified key"):
    """Assert that dictionary has the specified key"""
    if key not in dictionary:
        raise AssertionError(f"{message}: dictionary={dictionary}, key={key}")


def assert_has_value(dictionary, value, message="Dictionary does not have the specified value"):
    """Assert that dictionary has the specified value"""
    if value not in dictionary.values():
        raise AssertionError(f"{message}: dictionary={dictionary}, value={value}")


def assert_is_subclass(obj, expected_class, message="Object is not subclass of expected class"):
    """Assert that object is a subclass of expected class"""
    if not issubclass(obj, expected_class):
        raise AssertionError(f"{message}: obj={obj}, expected_class={expected_class}")


def assert_not_is_instance(obj, expected_class, message="Object is instance of expected class but should not be"):
    """Assert that object is not instance of expected class"""
    if isinstance(obj, expected_class):
        raise AssertionError(f"{message}: obj={obj}, expected_class={expected_class}")


def assert_not_is_subclass(obj, expected_class, message="Object is subclass of expected class but should not be"):
    """Assert that object is not subclass of expected class"""
    if issubclass(obj, expected_class):
        raise AssertionError(f"{message}: obj={obj}, expected_class={expected_class}")


def assert_match_regex(text, pattern, message="Text does not match regex pattern"):
    """Assert that text matches the regex pattern"""
    import re
    if not re.match(pattern, text):
        raise AssertionError(f"{message}: text={text}, pattern={pattern}")


def assert_not_match_regex(text, pattern, message="Text matches regex pattern but should not"):
    """Assert that text does not match the regex pattern"""
    import re
    if re.match(pattern, text):
        raise AssertionError(f"{message}: text={text}, pattern={pattern}")


def assert_file_exists(file_path, message="File does not exist"):
    """Assert that a file exists"""
    import os
    if not os.path.isfile(file_path):
        raise AssertionError(f"{message}: file_path={file_path}")


def assert_file_not_exists(file_path, message="File exists but should not"):
    """Assert that a file does not exist"""
    import os
    if os.path.isfile(file_path):
        raise AssertionError(f"{message}: file_path={file_path}")


def assert_dir_exists(dir_path, message="Directory does not exist"):
    """Assert that a directory exists"""
    import os
    if not os.path.isdir(dir_path):
        raise AssertionError(f"{message}: dir_path={dir_path}")


def assert_dir_not_exists(dir_path, message="Directory exists but should not"):
    """Assert that a directory does not exist"""
    import os
    if os.path.isdir(dir_path):
        raise AssertionError(f"{message}: dir_path={dir_path}")


def assert_path_exists(path, message="Path does not exist"):
    """Assert that a path exists"""
    import os
    if not os.path.exists(path):
        raise AssertionError(f"{message}: path={path}")


def assert_path_not_exists(path, message="Path exists but should not"):
    """Assert that a path does not exist"""
    import os
    if os.path.exists(path):
        raise AssertionError(f"{message}: path={path}")


def assert_is_file(path, message="Path is not a file"):
    """Assert that path is a file"""
    import os
    if not os.path.isfile(path):
        raise AssertionError(f"{message}: path={path}")


def assert_is_dir(path, message="Path is not a directory"):
    """Assert that path is a directory"""
    import os
    if not os.path.isdir(path):
        raise AssertionError(f"{message}: path={path}")


def assert_is_link(path, message="Path is not a symbolic link"):
    """Assert that path is a symbolic link"""
    import os
    if not os.path.islink(path):
        raise AssertionError(f"{message}: path={path}")


def assert_permission(path, permission, message="Path does not have required permission"):
    """Assert that path has the specified permission (r, w, x)"""
    import os
    perm_map = {
        'r': os.R_OK,
        'w': os.W_OK,
        'x': os.X_OK
    }
    if permission not in perm_map:
        raise ValueError(f"Invalid permission: {permission}. Use 'r', 'w', or 'x'.")
    
    if not os.access(path, perm_map[permission]):
        raise AssertionError(f"{message}: path={path}, permission={permission}")


def assert_readable(path, message="Path is not readable"):
    """Assert that path is readable"""
    assert_permission(path, 'r', message)


def assert_writable(path, message="Path is not writable"):
    """Assert that path is writable"""
    assert_permission(path, 'w', message)


def assert_executable(path, message="Path is not executable"):
    """Assert that path is executable"""
    assert_permission(path, 'x', message)


def assert_same_file(path1, path2, message="Paths do not refer to the same file"):
    """Assert that two paths refer to the same file"""
    import os
    if not os.path.samefile(path1, path2):
        raise AssertionError(f"{message}: path1={path1}, path2={path2}")


def assert_not_same_file(path1, path2, message="Paths refer to the same file but should not"):
    """Assert that two paths do not refer to the same file"""
    import os
    try:
        same = os.path.samefile(path1, path2)
        if same:
            raise AssertionError(f"{message}: path1={path1}, path2={path2}")
    except OSError:
        # If one of the paths doesn't exist, they can't be the same file
        pass


def assert_json_valid(json_str, message="JSON string is not valid"):
    """Assert that a string is valid JSON"""
    import json
    try:
        json.loads(json_str)
    except json.JSONDecodeError:
        raise AssertionError(f"{message}: json_str={json_str}")


def assert_xml_valid(xml_str, message="XML string is not valid"):
    """Assert that a string is valid XML"""
    import xml.etree.ElementTree as ET
    try:
        ET.fromstring(xml_str)
    except ET.ParseError:
        raise AssertionError(f"{message}: xml_str={xml_str}")


def assert_yaml_valid(yaml_str, message="YAML string is not valid"):
    """Assert that a string is valid YAML"""
    try:
        import yaml
        yaml.safe_load(yaml_str)
    except yaml.YAMLError:
        raise AssertionError(f"{message}: yaml_str={yaml_str}")


def assert_csv_valid(csv_str, message="CSV string is not valid"):
    """Assert that a string is valid CSV"""
    import csv
    from io import StringIO
    
    try:
        reader = csv.reader(StringIO(csv_str))
        # Try to read all rows to catch parsing errors
        for row in reader:
            pass
    except csv.Error:
        raise AssertionError(f"{message}: csv_str={csv_str}")


def assert_tsv_valid(tsv_str, message="TSV string is not valid"):
    """Assert that a string is valid TSV"""
    import csv
    from io import StringIO
    
    try:
        reader = csv.reader(StringIO(tsv_str), delimiter='\t')
        # Try to read all rows to catch parsing errors
        for row in reader:
            pass
    except csv.Error:
        raise AssertionError(f"{message}: tsv_str={tsv_str}")


def assert_json_equal(json1, json2, message="JSON objects are not equal"):
    """Assert that two JSON objects are equal"""
    import json
    
    parsed1 = json.loads(json1) if isinstance(json1, str) else json1
    parsed2 = json.loads(json2) if isinstance(json2, str) else json2
    
    if parsed1 != parsed2:
        raise AssertionError(f"{message}: json1={parsed1}, json2={parsed2}")


def assert_yaml_equal(yaml1, yaml2, message="YAML documents are not equal"):
    """Assert that two YAML documents are equal"""
    import yaml
    
    parsed1 = yaml.safe_load(yaml1) if isinstance(yaml1, str) else yaml1
    parsed2 = yaml.safe_load(yaml2) if isinstance(yaml2, str) else yaml2
    
    if parsed1 != parsed2:
        raise AssertionError(f"{message}: yaml1={parsed1}, yaml2={parsed2}")


def assert_csv_equal(csv1, csv2, message="CSV documents are not equal"):
    """Assert that two CSV documents are equal"""
    import csv
    from io import StringIO
    
    reader1 = csv.reader(StringIO(csv1) if isinstance(csv1, str) else StringIO(str(csv1)))
    reader2 = csv.reader(StringIO(csv2) if isinstance(csv2, str) else StringIO(str(csv2)))
    
    rows1 = list(reader1)
    rows2 = list(reader2)
    
    if rows1 != rows2:
        raise AssertionError(f"{message}: CSV documents are not equal")


def assert_tsv_equal(tsv1, tsv2, message="TSV documents are not equal"):
    """Assert that two TSV documents are equal"""
    import csv
    from io import StringIO
    
    reader1 = csv.reader(StringIO(tsv1) if isinstance(tsv1, str) else StringIO(str(tsv1)), delimiter='\t')
    reader2 = csv.reader(StringIO(tsv2) if isinstance(tsv2, str) else StringIO(str(tsv2)), delimiter='\t')
    
    rows1 = list(reader1)
    rows2 = list(reader2)
    
    if rows1 != rows2:
        raise AssertionError(f"{message}: TSV documents are not equal")


def assert_json_contains(json_obj, expected_subset, message="JSON object does not contain expected subset"):
    """Assert that a JSON object contains the expected subset"""
    import json
    
    if isinstance(json_obj, str):
        json_obj = json.loads(json_obj)
    
    if isinstance(expected_subset, str):
        expected_subset = json.loads(expected_subset)
    
    def _dict_contains(a, b):
        """Check if dict a contains all key-value pairs in dict b"""
        if not isinstance(a, dict) or not isinstance(b, dict):
            return a == b
        for key, value in b.items():
            if key not in a:
                return False
            if not _dict_contains(a[key], value):
                return False
        return True
    
    if not _dict_contains(json_obj, expected_subset):
        raise AssertionError(f"{message}: json_obj={json_obj}, expected_subset={expected_subset}")


def assert_tensor_all_close(tensor1, tensor2, rtol=1e-05, atol=1e-08, message="Tensors are not all close"):
    """Assert that all elements in two tensors are close within tolerance"""
    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        raise AssertionError(f"{message}: tensor1 and tensor2 are not close")


def assert_tensor_all_equal(tensor1, tensor2, message="Tensors are not all equal"):
    """Assert that all elements in two tensors are equal"""
    if not torch.equal(tensor1, tensor2):
        raise AssertionError(f"{message}: tensor1 and tensor2 are not equal")


def assert_tensor_all_finite(tensor, message="Tensor contains non-finite values"):
    """Assert that all elements in the tensor are finite"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise AssertionError(f"{message}: tensor contains NaN or Inf values")


def assert_tensor_all_positive(tensor, message="Tensor contains non-positive values"):
    """Assert that all elements in the tensor are positive"""
    if not (tensor > 0).all():
        raise AssertionError(f"{message}: tensor contains non-positive values")


def assert_tensor_all_negative(tensor, message="Tensor contains non-negative values"):
    """Assert that all elements in the tensor are negative"""
    if not (tensor < 0).all():
        raise AssertionError(f"{message}: tensor contains non-negative values")


def assert_tensor_all_non_negative(tensor, message="Tensor contains negative values"):
    """Assert that all elements in the tensor are non-negative"""
    if not (tensor >= 0).all():
        raise AssertionError(f"{message}: tensor contains negative values")


def assert_tensor_all_non_positive(tensor, message="Tensor contains positive values"):
    """Assert that all elements in the tensor are non-positive"""
    if not (tensor <= 0).all():
        raise AssertionError(f"{message}: tensor contains positive values")


def assert_tensor_all_zero(tensor, message="Tensor contains non-zero values"):
    """Assert that all elements in the tensor are zero"""
    if not (tensor == 0).all():
        raise AssertionError(f"{message}: tensor contains non-zero values")


def assert_tensor_all_not_zero(tensor, message="Tensor contains zero values"):
    """Assert that all elements in the tensor are not zero"""
    if not (tensor != 0).all():
        raise AssertionError(f"{message}: tensor contains zero values")


def assert_tensor_all_greater(tensor, comparison, message="Not all tensor elements are greater than comparison"):
    """Assert that all elements in the tensor are greater than comparison"""
    if not (tensor > comparison).all():
        raise AssertionError(f"{message}: tensor elements are not all greater than {comparison}")


def assert_tensor_all_less(tensor, comparison, message="Not all tensor elements are less than comparison"):
    """Assert that all elements in the tensor are less than comparison"""
    if not (tensor < comparison).all():
        raise AssertionError(f"{message}: tensor elements are not all less than {comparison}")


def assert_tensor_all_greater_equal(tensor, comparison, message="Not all tensor elements are greater than or equal to comparison"):
    """Assert that all elements in the tensor are greater than or equal to comparison"""
    if not (tensor >= comparison).all():
        raise AssertionError(f"{message}: tensor elements are not all greater than or equal to {comparison}")


def assert_tensor_all_less_equal(tensor, comparison, message="Not all tensor elements are less than or equal to comparison"):
    """Assert that all elements in the tensor are less than or equal to comparison"""
    if not (tensor <= comparison).all():
        raise AssertionError(f"{message}: tensor elements are not all less than or equal to {comparison}")


def assert_tensor_all_between(tensor, lower_bound, upper_bound, message="Not all tensor elements are between bounds"):
    """Assert that all elements in the tensor are between lower and upper bounds"""
    if not ((tensor >= lower_bound) & (tensor <= upper_bound)).all():
        raise AssertionError(f"{message}: tensor elements are not all between [{lower_bound}, {upper_bound}]")


def assert_tensor_any_positive(tensor, message="No tensor elements are positive"):
    """Assert that any element in the tensor is positive"""
    if not (tensor > 0).any():
        raise AssertionError(f"{message}: no tensor elements are positive")


def assert_tensor_any_negative(tensor, message="No tensor elements are negative"):
    """Assert that any element in the tensor is negative"""
    if not (tensor < 0).any():
        raise AssertionError(f"{message}: no tensor elements are negative")


def assert_tensor_any_non_negative(tensor, message="No tensor elements are non-negative"):
    """Assert that any element in the tensor is non-negative"""
    if not (tensor >= 0).any():
        raise AssertionError(f"{message}: no tensor elements are non-negative")


def assert_tensor_any_non_positive(tensor, message="No tensor elements are non-positive"):
    """Assert that any element in the tensor is non-positive"""
    if not (tensor <= 0).any():
        raise AssertionError(f"{message}: no tensor elements are non-positive")


def assert_tensor_any_zero(tensor, message="No tensor elements are zero"):
    """Assert that any element in the tensor is zero"""
    if not (tensor == 0).any():
        raise AssertionError(f"{message}: no tensor elements are zero")


def assert_tensor_any_not_zero(tensor, message="No tensor elements are non-zero"):
    """Assert that any element in the tensor is not zero"""
    if not (tensor != 0).any():
        raise AssertionError(f"{message}: no tensor elements are non-zero")


def assert_tensor_any_greater(tensor, comparison, message="No tensor elements are greater than comparison"):
    """Assert that any element in the tensor is greater than comparison"""
    if not (tensor > comparison).any():
        raise AssertionError(f"{message}: no tensor elements are greater than {comparison}")


def assert_tensor_any_less(tensor, comparison, message="No tensor elements are less than comparison"):
    """Assert that any element in the tensor is less than comparison"""
    if not (tensor < comparison).any():
        raise AssertionError(f"{message}: no tensor elements are less than {comparison}")


def assert_tensor_any_greater_equal(tensor, comparison, message="No tensor elements are greater than or equal to comparison"):
    """Assert that any element in the tensor is greater than or equal to comparison"""
    if not (tensor >= comparison).any():
        raise AssertionError(f"{message}: no tensor elements are greater than or equal to {comparison}")


def assert_tensor_any_less_equal(tensor, comparison, message="No tensor elements are less than or equal to comparison"):
    """Assert that any element in the tensor is less than or equal to comparison"""
    if not (tensor <= comparison).any():
        raise AssertionError(f"{message}: no tensor elements are less than or equal to {comparison}")


def assert_tensor_any_between(tensor, lower_bound, upper_bound, message="No tensor elements are between bounds"):
    """Assert that any element in the tensor is between lower and upper bounds"""
    if not ((tensor >= lower_bound) & (tensor <= upper_bound)).any():
        raise AssertionError(f"{message}: no tensor elements are between [{lower_bound}, {upper_bound}]")


def assert_tensor_mean(tensor, expected_mean, rtol=1e-05, atol=1e-08, message="Tensor mean is not as expected"):
    """Assert that the mean of the tensor is as expected"""
    actual_mean = tensor.mean()
    if not torch.allclose(actual_mean, torch.tensor(expected_mean), rtol=rtol, atol=atol):
        raise AssertionError(f"{message}: actual_mean={actual_mean}, expected_mean={expected_mean}")


def assert_tensor_std(tensor, expected_std, rtol=1e-05, atol=1e-08, message="Tensor std is not as expected"):
    """Assert that the std of the tensor is as expected"""
    actual_std = tensor.std()
    if not torch.allclose(actual_std, torch.tensor(expected_std), rtol=rtol, atol=atol):
        raise AssertionError(f"{message}: actual_std={actual_std}, expected_std={expected_std}")


def assert_tensor_var(tensor, expected_var, rtol=1e-05, atol=1e-08, message="Tensor var is not as expected"):
    """Assert that the var of the tensor is as expected"""
    actual_var = tensor.var()
    if not torch.allclose(actual_var, torch.tensor(expected_var), rtol=rtol, atol=atol):
        raise AssertionError(f"{message}: actual_var={actual_var}, expected_var={expected_var}")


def assert_tensor_sum(tensor, expected_sum, rtol=1e-05, atol=1e-08, message="Tensor sum is not as expected"):
    """Assert that the sum of the tensor is as expected"""
    actual_sum = tensor.sum()
    if not torch.allclose(actual_sum, torch.tensor(expected_sum), rtol=rtol, atol=atol):
        raise AssertionError(f"{message}: actual_sum={actual_sum}, expected_sum={expected_sum}")


def assert_tensor_min(tensor, expected_min, rtol=1e-05, atol=1e-08, message="Tensor min is not as expected"):
    """Assert that the min of the tensor is as expected"""
    actual_min = tensor.min()
    if not torch.allclose(actual_min, torch.tensor(expected_min), rtol=rtol, atol=atol):
        raise AssertionError(f"{message}: actual_min={actual_min}, expected_min={expected_min}")


def assert_tensor_max(tensor, expected_max, rtol=1e-05, atol=1e-08, message="Tensor max is not as expected"):
    """Assert that the max of the tensor is as expected"""
    actual_max = tensor.max()
    if not torch.allclose(actual_max, torch.tensor(expected_max), rtol=rtol, atol=atol):
        raise AssertionError(f"{message}: actual_max={actual_max}, expected_max={expected_max}")


def assert_tensor_unique(tensor, expected_unique_count, message="Tensor unique count is not as expected"):
    """Assert that the tensor has the expected number of unique values"""
    unique_count = torch.unique(tensor).numel()
    if unique_count != expected_unique_count:
        raise AssertionError(f"{message}: actual_unique_count={unique_count}, expected_unique_count={expected_unique_count}")


def assert_tensor_sorted(tensor, descending=False, message="Tensor is not sorted"):
    """Assert that the tensor is sorted"""
    if descending:
        is_sorted = torch.all(tensor[:-1] >= tensor[1:])
    else:
        is_sorted = torch.all(tensor[:-1] <= tensor[1:])
    
    if not is_sorted:
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_in_range(tensor, lower_bound, upper_bound, message="Tensor values are not in range"):
    """Assert that all tensor values are in the specified range"""
    if not ((tensor >= lower_bound) & (tensor <= upper_bound)).all():
        raise AssertionError(f"{message}: tensor values are not all in range [{lower_bound}, {upper_bound}]")


def assert_tensor_not_in_range(tensor, lower_bound, upper_bound, message="Tensor values are in range but should not be"):
    """Assert that no tensor values are in the specified range"""
    in_range = (tensor >= lower_bound) & (tensor <= upper_bound)
    if in_range.any():
        raise AssertionError(f"{message}: tensor values are in range [{lower_bound}, {upper_bound}] but should not be")


def assert_tensor_all_close_to_value(tensor, value, rtol=1e-05, atol=1e-08, message="Tensor values are not close to value"):
    """Assert that all tensor values are close to a specific value"""
    target_tensor = torch.full_like(tensor, value)
    close_mask = torch.abs(tensor - target_tensor) <= (atol + rtol * torch.abs(target_tensor))
    if not close_mask.all():
        raise AssertionError(f"{message}: not all tensor values are close to {value}")


def assert_tensor_any_close_to_value(tensor, value, rtol=1e-05, atol=1e-08, message="No tensor values are close to value"):
    """Assert that any tensor value is close to a specific value"""
    target_tensor = torch.full_like(tensor, value)
    close_mask = torch.abs(tensor - target_tensor) <= (atol + rtol * torch.abs(target_tensor))
    if not close_mask.any():
        raise AssertionError(f"{message}: no tensor values are close to {value}")


def assert_tensor_all_elements_in_range(tensor, min_val, max_val, message="Not all tensor elements are in the specified range"):
    """Assert that all tensor elements are in the specified range"""
    if not ((tensor >= min_val) & (tensor <= max_val)).all():
        raise AssertionError(f"{message}: tensor elements are not all in range [{min_val}, {max_val}]")


def assert_tensor_any_elements_in_range(tensor, min_val, max_val, message="No tensor elements are in the specified range"):
    """Assert that any tensor element is in the specified range"""
    if not ((tensor >= min_val) & (tensor <= max_val)).any():
        raise AssertionError(f"{message}: no tensor elements are in range [{min_val}, {max_val}]")


def assert_tensor_all_elements_outside_range(tensor, min_val, max_val, message="Not all tensor elements are outside the specified range"):
    """Assert that all tensor elements are outside the specified range"""
    if not ((tensor < min_val) | (tensor > max_val)).all():
        raise AssertionError(f"{message}: not all tensor elements are outside range [{min_val}, {max_val}]")


def assert_tensor_any_elements_outside_range(tensor, min_val, max_val, message="No tensor elements are outside the specified range"):
    """Assert that any tensor element is outside the specified range"""
    if not ((tensor < min_val) | (tensor > max_val)).any():
        raise AssertionError(f"{message}: no tensor elements are outside range [{min_val}, {max_val}]")


def assert_tensor_all_elements_unique(tensor, message="Not all tensor elements are unique"):
    """Assert that all tensor elements are unique"""
    unique_elements = torch.unique(tensor)
    if unique_elements.numel() != tensor.numel():
        raise AssertionError(f"{message}: tensor has duplicate elements")


def assert_tensor_all_elements_monotonic(tensor, message="Not all tensor elements are monotonic"):
    """Assert that all tensor elements are monotonic"""
    is_increasing = torch.all(tensor[1:] >= tensor[:-1])
    is_decreasing = torch.all(tensor[1:] <= tensor[:-1])
    
    if not (is_increasing or is_decreasing):
        raise AssertionError(f"{message}: tensor elements are not monotonic")


def assert_tensor_any_elements_monotonic(tensor, message="No tensor elements are monotonic"):
    """Assert that any tensor elements are monotonic"""
    # For this assertion, we'll check if the entire tensor is monotonic
    is_increasing = torch.all(tensor[1:] >= tensor[:-1])
    is_decreasing = torch.all(tensor[1:] <= tensor[:-1])
    
    if not (is_increasing or is_decreasing):
        raise AssertionError(f"{message}: tensor elements are not monotonic")


def assert_tensor_all_elements_strictly_monotonic(tensor, message="Not all tensor elements are strictly monotonic"):
    """Assert that all tensor elements are strictly monotonic"""
    is_increasing = torch.all(tensor[1:] > tensor[:-1])
    is_decreasing = torch.all(tensor[1:] < tensor[:-1])
    
    if not (is_increasing or is_decreasing):
        raise AssertionError(f"{message}: tensor elements are not strictly monotonic")


def assert_tensor_any_elements_strictly_monotonic(tensor, message="No tensor elements are strictly monotonic"):
    """Assert that any tensor elements are strictly monotonic"""
    # For this assertion, we'll check if the entire tensor is strictly monotonic
    is_increasing = torch.all(tensor[1:] > tensor[:-1])
    is_decreasing = torch.all(tensor[1:] < tensor[:-1])
    
    if not (is_increasing or is_decreasing):
        raise AssertionError(f"{message}: tensor elements are not strictly monotonic")


def assert_tensor_all_elements_match_condition(tensor, condition_func, message="Not all tensor elements match the condition"):
    """Assert that all tensor elements match the specified condition function"""
    if not condition_func(tensor).all():
        raise AssertionError(f"{message}: not all tensor elements match condition {condition_func}")


def assert_tensor_any_elements_match_condition(tensor, condition_func, message="No tensor elements match the specified condition"):
    """Assert that any tensor element matches the specified condition function"""
    if not condition_func(tensor).any():
        raise AssertionError(f"{message}: no tensor elements match condition {condition_func}")


def assert_tensor_all_elements_fail_condition(tensor, condition_func, message="All tensor elements match the condition but should not"):
    """Assert that all tensor elements fail the specified condition function"""
    if condition_func(tensor).any():
        raise AssertionError(f"{message}: all tensor elements match condition {condition_func} but should not")


def assert_tensor_any_elements_fail_condition(tensor, condition_func, message="No tensor elements fail the specified condition"):
    """Assert that any tensor element fails the specified condition function"""
    if not condition_func(tensor).all():
        raise AssertionError(f"{message}: no tensor elements fail condition {condition_func}")


def assert_tensor_all_elements_satisfy_predicate(tensor, predicate_func, message="Not all tensor elements satisfy the predicate"):
    """Assert that all tensor elements satisfy the specified predicate function"""
    flattened = tensor.flatten()
    for element in flattened:
        if not predicate_func(element.item()):
            raise AssertionError(f"{message}: element {element.item()} does not satisfy predicate {predicate_func}")


def assert_tensor_any_elements_satisfy_predicate(tensor, predicate_func, message="No tensor elements satisfy the predicate"):
    """Assert that any tensor element satisfies the specified predicate function"""
    flattened = tensor.flatten()
    for element in flattened:
        if predicate_func(element.item()):
            return  # Found one that satisfies the predicate
    raise AssertionError(f"{message}: no tensor elements satisfy predicate {predicate_func}")


def assert_tensor_all_elements_fail_predicate(tensor, predicate_func, message="All tensor elements satisfy the predicate but should not"):
    """Assert that all tensor elements fail the specified predicate function"""
    flattened = tensor.flatten()
    for element in flattened:
        if predicate_func(element.item()):
            raise AssertionError(f"{message}: element {element.item()} satisfies predicate {predicate_func} but should not")


def assert_tensor_any_elements_fail_predicate(tensor, predicate_func, message="No tensor elements fail the predicate"):
    """Assert that any tensor element fails the specified predicate function"""
    flattened = tensor.flatten()
    for element in flattened:
        if not predicate_func(element.item()):
            return  # Found one that fails the predicate
    raise AssertionError(f"{message}: all tensor elements satisfy predicate {predicate_func}, none fail")


def assert_tensor_all_elements_in_set(tensor, value_set, message="Not all tensor elements are in the specified set"):
    """Assert that all tensor elements are in the specified set"""
    flattened = tensor.flatten()
    for element in flattened:
        if element.item() not in value_set:
            raise AssertionError(f"{message}: element {element.item()} not in set {value_set}")


def assert_tensor_any_elements_in_set(tensor, value_set, message="No tensor elements are in the specified set"):
    """Assert that any tensor element is in the specified set"""
    flattened = tensor.flatten()
    for element in flattened:
        if element.item() in value_set:
            return  # Found one in the set
    raise AssertionError(f"{message}: no tensor elements are in set {value_set}")


def assert_tensor_all_elements_not_in_set(tensor, value_set, message="Not all tensor elements are outside the specified set"):
    """Assert that all tensor elements are outside the specified set"""
    flattened = tensor.flatten()
    for element in flattened:
        if element.item() in value_set:
            raise AssertionError(f"{message}: element {element.item()} is in set {value_set} but should not be")


def assert_tensor_any_elements_not_in_set(tensor, value_set, message="No tensor elements are outside the specified set"):
    """Assert that any tensor element is outside the specified set"""
    flattened = tensor.flatten()
    for element in flattened:
        if element.item() not in value_set:
            return  # Found one outside the set
    raise AssertionError(f"{message}: all tensor elements are in set {value_set}, none are outside")


def assert_tensor_all_elements_close_to_any_value_in_set(tensor, value_set, rtol=1e-05, atol=1e-08, message="Not all tensor values are close to any value in the set"):
    """Assert that all tensor values are close to any value in the specified set"""
    for val in tensor.flatten():
        close_to_any = False
        for set_val in value_set:
            if abs(val.item() - set_val) <= (atol + rtol * abs(set_val)):
                close_to_any = True
                break
        if not close_to_any:
            raise AssertionError(f"{message}: value {val.item()} is not close to any value in set {value_set}")


def assert_tensor_any_close_to_any_value_in_set(tensor, value_set, rtol=1e-05, atol=1e-08, message="No tensor values are close to any value in the set"):
    """Assert that any tensor value is close to any value in the specified set"""
    found_close = False
    for val in tensor.flatten():
        for set_val in value_set:
            if abs(val.item() - set_val) <= (atol + rtol * abs(set_val)):
                found_close = True
                break
        if found_close:
            break
    
    if not found_close:
        raise AssertionError(f"{message}: no tensor values are close to any value in set {value_set}")


def assert_tensor_all_elements_increasing(tensor, strict=False, message="Tensor values are not increasing"):
    """Assert that tensor values are increasing"""
    if strict:
        is_increasing = torch.all(tensor[1:] > tensor[:-1])
    else:
        is_increasing = torch.all(tensor[1:] >= tensor[:-1])
    
    if not is_increasing:
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_all_elements_decreasing(tensor, strict=False, message="Tensor values are not decreasing"):
    """Assert that tensor values are decreasing"""
    if strict:
        is_decreasing = torch.all(tensor[1:] < tensor[:-1])
    else:
        is_decreasing = torch.all(tensor[1:] <= tensor[:-1])
    
    if not is_decreasing:
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_any_elements_increasing(tensor, strict=False, message="No tensor values form an increasing sequence"):
    """Assert that any tensor values form an increasing sequence"""
    if strict:
        increasing_pairs = tensor[1:] > tensor[:-1]
    else:
        increasing_pairs = tensor[1:] >= tensor[:-1]
    
    if not increasing_pairs.any():
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_any_elements_decreasing(tensor, strict=False, message="No tensor values form a decreasing sequence"):
    """Assert that any tensor values form a decreasing sequence"""
    if strict:
        decreasing_pairs = tensor[1:] < tensor[:-1]
    else:
        decreasing_pairs = tensor[1:] <= tensor[:-1]
    
    if not decreasing_pairs.any():
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_all_elements_between_intervals(tensor, intervals, message="Not all tensor elements are in the specified intervals"):
    """Assert that all tensor elements are in one of the specified intervals"""
    for val in tensor.flatten():
        in_any_interval = False
        for start, end in intervals:
            if start <= val.item() <= end:
                in_any_interval = True
                break
        if not in_any_interval:
            raise AssertionError(f"{message}: value {val.item()} is not in any of the intervals {intervals}")


def assert_tensor_any_elements_between_intervals(tensor, intervals, message="No tensor elements are in the specified intervals"):
    """Assert that any tensor element is in one of the specified intervals"""
    for val in tensor.flatten():
        for start, end in intervals:
            if start <= val.item() <= end:
                return  # Found one in an interval
    raise AssertionError(f"{message}: no tensor elements are in any of the intervals {intervals}")


def assert_tensor_all_elements_outside_intervals(tensor, intervals, message="Not all tensor elements are outside the specified intervals"):
    """Assert that all tensor elements are outside all specified intervals"""
    for val in tensor.flatten():
        for start, end in intervals:
            if start <= val.item() <= end:
                raise AssertionError(f"{message}: value {val.item()} is in interval [{start}, {end}] but should not be")


def assert_tensor_any_elements_outside_intervals(tensor, intervals, message="No tensor elements are outside the specified intervals"):
    """Assert that any tensor element is outside all specified intervals"""
    for val in tensor.flatten():
        in_all_intervals = True
        for start, end in intervals:
            if start <= val.item() <= end:
                in_all_intervals = True
                break
            else:
                in_all_intervals = False
        if not in_all_intervals:
            return  # Found one outside all intervals
    raise AssertionError(f"{message}: all tensor elements are in at least one of the intervals {intervals}")


def assert_tensor_all_elements_match_pattern(tensor, pattern_func, message="Not all tensor elements match the specified pattern"):
    """Assert that all tensor elements match the specified pattern function"""
    flattened = tensor.flatten()
    for element in flattened:
        if not pattern_func(element.item()):
            raise AssertionError(f"{message}: element {element.item()} does not match pattern {pattern_func}")


def assert_tensor_any_elements_match_pattern(tensor, pattern_func, message="No tensor elements match the specified pattern"):
    """Assert that any tensor element matches the specified pattern function"""
    flattened = tensor.flatten()
    for element in flattened:
        if pattern_func(element.item()):
            return  # Found one that matches
    raise AssertionError(f"{message}: no tensor elements match pattern {pattern_func}")


def assert_tensor_all_elements_fail_pattern(tensor, pattern_func, message="All tensor elements match the pattern but should not"):
    """Assert that all tensor elements fail the specified pattern function"""
    flattened = tensor.flatten()
    for element in flattened:
        if pattern_func(element.item()):
            raise AssertionError(f"{message}: element {element.item()} matches pattern {pattern_func} but should not")


def assert_tensor_any_elements_fail_pattern(tensor, pattern_func, message="No tensor elements fail the pattern"):
    """Assert that any tensor element fails the specified pattern function"""
    flattened = tensor.flatten()
    for element in flattened:
        if not pattern_func(element.item()):
            return  # Found one that fails
    raise AssertionError(f"{message}: all tensor elements match pattern {pattern_func}, none fail")


def assert_tensor_all_elements_meet_criteria(tensor, criteria_func, message="Not all tensor elements meet the specified criteria"):
    """Assert that all tensor elements meet the specified criteria function"""
    flattened = tensor.flatten()
    for element in flattened:
        if not criteria_func(element.item()):
            raise AssertionError(f"{message}: element {element.item()} does not meet criteria {criteria_func}")


def assert_tensor_any_elements_meet_criteria(tensor, criteria_func, message="No tensor elements meet the specified criteria"):
    """Assert that any tensor element meets the specified criteria function"""
    flattened = tensor.flatten()
    for element in flattened:
        if criteria_func(element.item()):
            return  # Found one that meets criteria
    raise AssertionError(f"{message}: no tensor elements meet criteria {criteria_func}")


def assert_tensor_all_elements_not_meet_criteria(tensor, criteria_func, message="All tensor elements meet the criteria but should not"):
    """Assert that all tensor elements do not meet the specified criteria function"""
    flattened = tensor.flatten()
    for element in flattened:
        if criteria_func(element.item()):
            raise AssertionError(f"{message}: element {element.item()} meets criteria {criteria_func} but should not")


def assert_tensor_any_elements_not_meet_criteria(tensor, criteria_func, message="No tensor elements fail to meet the specified criteria"):
    """Assert that any tensor element does not meet the specified criteria function"""
    flattened = tensor.flatten()
    for element in flattened:
        if not criteria_func(element.item()):
            return  # Found one that doesn't meet criteria
    raise AssertionError(f"{message}: all tensor elements meet criteria {criteria_func}, none fail to meet them")


def assert_tensor_all_elements_pass_test(tensor, test_func, message="Not all tensor elements pass the specified test"):
    """Assert that all tensor elements pass the specified test function"""
    flattened = tensor.flatten()
    for element in flattened:
        if not test_func(element.item()):
            raise AssertionError(f"{message}: element {element.item()} does not pass test {test_func}")


def assert_tensor_any_elements_pass_test(tensor, test_func, message="No tensor elements pass the specified test"):
    """Assert that any tensor element passes the specified test function"""
    flattened = tensor.flatten()
    for element in flattened:
        if test_func(element.item()):
            return  # Found one that passes
    raise AssertionError(f"{message}: no tensor elements pass test {test_func}")


def assert_tensor_all_elements_fail_test(tensor, test_func, message="All tensor elements pass the test but should not"):
    """Assert that all tensor elements fail the specified test function"""
    flattened = tensor.flatten()
    for element in flattened:
        if test_func(element.item()):
            raise AssertionError(f"{message}: element {element.item()} passes test {test_func} but should not")


def assert_tensor_any_elements_fail_test(tensor, test_func, message="No tensor elements fail the specified test"):
    """Assert that any tensor element fails the specified test function"""
    flattened = tensor.flatten()
    for element in flattened:
        if not test_func(element.item()):
            return  # Found one that fails
    raise AssertionError(f"{message}: all tensor elements pass test {test_func}, none fail")


def assert_tensor_all_elements_satisfy_condition_func(tensor, condition_func, message="Not all tensor elements satisfy the specified condition function"):
    """Assert that all tensor elements satisfy the specified condition function"""
    flattened = tensor.flatten()
    for element in flattened:
        if not condition_func(element.item()):
            raise AssertionError(f"{message}: element {element.item()} does not satisfy condition {condition_func}")


def assert_tensor_any_elements_satisfy_condition_func(tensor, condition_func, message="No tensor elements satisfy the specified condition function"):
    """Assert that any tensor element satisfies the specified condition function"""
    flattened = tensor.flatten()
    for element in flattened:
        if condition_func(element.item()):
            return  # Found one that satisfies
    raise AssertionError(f"{message}: no tensor elements satisfy condition {condition_func}")


def assert_tensor_all_elements_fail_condition_func(tensor, condition_func, message="All tensor elements satisfy the condition function but should not"):
    """Assert that all tensor elements fail the specified condition function"""
    flattened = tensor.flatten()
    for element in flattened:
        if condition_func(element.item()):
            raise AssertionError(f"{message}: element {element.item()} satisfies condition {condition_func} but should not")


def assert_tensor_any_elements_fail_condition_func(tensor, condition_func, message="No tensor elements fail the specified condition function"):
    """Assert that any tensor element fails the specified condition function"""
    flattened = tensor.flatten()
    for element in flattened:
        if not condition_func(element.item()):
            return  # Found one that fails
    raise AssertionError(f"{message}: all tensor elements satisfy condition {condition_func}, none fail")


def assert_tensor_all_elements_increasing_sequence(tensor, strict=False, message="Tensor elements do not form an increasing sequence"):
    """Assert that tensor elements form an increasing sequence"""
    if strict:
        is_increasing = torch.all(tensor[1:] > tensor[:-1])
    else:
        is_increasing = torch.all(tensor[1:] >= tensor[:-1])
    
    if not is_increasing:
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_all_elements_decreasing_sequence(tensor, strict=False, message="Tensor elements do not form a decreasing sequence"):
    """Assert that tensor elements form a decreasing sequence"""
    if strict:
        is_decreasing = torch.all(tensor[1:] < tensor[:-1])
    else:
        is_decreasing = torch.all(tensor[1:] <= tensor[:-1])
    
    if not is_decreasing:
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_any_elements_increasing_sequence(tensor, strict=False, message="No tensor elements form an increasing sequence"):
    """Assert that any tensor elements form an increasing sequence"""
    if strict:
        is_increasing = torch.any(tensor[1:] > tensor[:-1])
    else:
        is_increasing = torch.any(tensor[1:] >= tensor[:-1])
    
    if not is_increasing:
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_any_elements_decreasing_sequence(tensor, strict=False, message="No tensor elements form a decreasing sequence"):
    """Assert that any tensor elements form a decreasing sequence"""
    if strict:
        is_decreasing = torch.any(tensor[1:] < tensor[:-1])
    else:
        is_decreasing = torch.any(tensor[1:] <= tensor[:-1])
    
    if not is_decreasing:
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_all_elements_monotonic_sequence(tensor, message="Tensor elements do not form a monotonic sequence"):
    """Assert that tensor elements form a monotonic sequence"""
    is_increasing = torch.all(tensor[1:] >= tensor[:-1])
    is_decreasing = torch.all(tensor[1:] <= tensor[:-1])
    
    if not (is_increasing or is_decreasing):
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_any_elements_monotonic_sequence(tensor, message="No tensor elements form a monotonic sequence"):
    """Assert that any tensor elements form a monotonic sequence"""
    # For this assertion, we'll check if the entire tensor is monotonic
    is_increasing = torch.all(tensor[1:] >= tensor[:-1])
    is_decreasing = torch.all(tensor[1:] <= tensor[:-1])
    
    if not (is_increasing or is_decreasing):
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_all_elements_strictly_monotonic_sequence(tensor, message="Tensor elements do not form a strictly monotonic sequence"):
    """Assert that tensor elements form a strictly monotonic sequence"""
    is_increasing = torch.all(tensor[1:] > tensor[:-1])
    is_decreasing = torch.all(tensor[1:] < tensor[:-1])
    
    if not (is_increasing or is_decreasing):
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_any_elements_strictly_monotonic_sequence(tensor, message="No tensor elements form a strictly monotonic sequence"):
    """Assert that any tensor elements form a strictly monotonic sequence"""
    # For this assertion, we'll check if the entire tensor is strictly monotonic
    is_increasing = torch.all(tensor[1:] > tensor[:-1])
    is_decreasing = torch.all(tensor[1:] < tensor[:-1])
    
    if not (is_increasing or is_decreasing):
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_all_elements_sorted_sequence(tensor, descending=False, message="Tensor elements are not sorted in sequence"):
    """Assert that tensor elements are sorted in sequence"""
    if descending:
        is_sorted = torch.all(tensor[:-1] >= tensor[1:])
    else:
        is_sorted = torch.all(tensor[:-1] <= tensor[1:])
    
    if not is_sorted:
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_any_elements_sorted_sequence(tensor, descending=False, message="No tensor elements form a sorted sequence"):
    """Assert that any tensor elements form a sorted sequence"""
    # For this assertion, we'll check if the entire tensor is sorted
    if descending:
        is_sorted = torch.all(tensor[:-1] >= tensor[1:])
    else:
        is_sorted = torch.all(tensor[:-1] <= tensor[1:])
    
    if not is_sorted:
        raise AssertionError(f"{message}: tensor={tensor}")


def assert_tensor_all_elements_unique_in_sequence(tensor, message="Not all tensor elements in sequence are unique"):
    """Assert that all tensor elements in sequence are unique"""
    unique_elements = torch.unique(tensor)
    if unique_elements.numel() != tensor.numel():
        raise AssertionError(f"{message}: tensor has duplicate elements")


def assert_tensor_any_elements_unique_in_sequence(tensor, message="No tensor elements in sequence are unique (all are duplicates)"):
    """Assert that any tensor element in sequence is unique (not a duplicate)"""
    # This is a tricky assertion since if all elements are the same, none are unique
    # So we'll check if there are at least some different values
    unique_elements = torch.unique(tensor)
    if unique_elements.numel() == 1 and tensor.numel() > 1:
        raise AssertionError(f"{message}: all tensor elements are the same")


def assert_tensor_all_elements_finite_in_sequence(tensor, message="Not all tensor elements in sequence are finite"):
    """Assert that all tensor elements in sequence are finite"""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise AssertionError(f"{message}: tensor contains NaN or Inf values")


def assert_tensor_any_elements_finite_in_sequence(tensor, message="No tensor elements in sequence are finite"):
    """Assert that any tensor element in sequence is finite"""
    finite_mask = torch.isfinite(tensor)
    if not finite_mask.any():
        raise AssertionError(f"{message}: no tensor elements are finite")


def assert_tensor_all_elements_positive_in_sequence(tensor, message="Not all tensor elements in sequence are positive"):
    """Assert that all tensor elements in sequence are positive"""
    if not (tensor > 0).all():
        raise AssertionError(f"{message}: not all tensor elements are positive")


def assert_tensor_any_elements_positive_in_sequence(tensor, message="No tensor elements in sequence are positive"):
    """Assert that any tensor element in sequence is positive"""
    if not (tensor > 0).any():
        raise AssertionError(f"{message}: no tensor elements are positive")


def assert_tensor_all_elements_negative_in_sequence(tensor, message="Not all tensor elements in sequence are negative"):
    """Assert that all tensor elements in sequence are negative"""
    if not (tensor < 0).all():
        raise AssertionError(f"{message}: not all tensor elements are negative")


def assert_tensor_any_elements_negative_in_sequence(tensor, message="No tensor elements in sequence are negative"):
    """Assert that any tensor element in sequence is negative"""
    if not (tensor < 0).any():
        raise AssertionError(f"{message}: no tensor elements are negative")


def assert_tensor_all_elements_non_negative_in_sequence(tensor, message="Not all tensor elements in sequence are non-negative"):
    """Assert that all tensor elements in sequence are non-negative"""
    if not (tensor >= 0).all():
        raise AssertionError(f"{message}: not all tensor elements are non-negative")


def assert_tensor_any_elements_non_negative_in_sequence(tensor, message="No tensor elements in sequence are non-negative"):
    """Assert that any tensor element in sequence is non-negative"""
    if not (tensor >= 0).any():
        raise AssertionError(f"{message}: no tensor elements are non-negative")


def assert_tensor_all_elements_non_positive_in_sequence(tensor, message="Not all tensor elements in sequence are non-positive"):
    """Assert that all tensor elements in sequence are non-positive"""
    if not (tensor <= 0).all():
        raise AssertionError(f"{message}: not all tensor elements are non-positive")


def assert_tensor_any_elements_non_positive_in_sequence(tensor, message="No tensor elements in sequence are non-positive"):
    """Assert that any tensor element in sequence is non-positive"""
    if not (tensor <= 0).any():
        raise AssertionError(f"{message}: no tensor elements are non-positive")


def assert_tensor_all_elements_zero_in_sequence(tensor, message="Not all tensor elements in sequence are zero"):
    """Assert that all tensor elements in sequence are zero"""
    if not (tensor == 0).all():
        raise AssertionError(f"{message}: not all tensor elements are zero")


def assert_tensor_any_elements_zero_in_sequence(tensor, message="No tensor elements in sequence are zero"):
    """Assert that any tensor element in sequence is zero"""
    if not (tensor == 0).any():
        raise AssertionError(f"{message}: no tensor elements are zero")


def assert_tensor_all_elements_not_zero_in_sequence(tensor, message="Not all tensor elements in sequence are non-zero"):
    """Assert that all tensor elements in sequence are non-zero"""
    if not (tensor != 0).all():
        raise AssertionError(f"{message}: not all tensor elements are non-zero")


def assert_tensor_any_elements_not_zero_in_sequence(tensor, message="No tensor elements in sequence are non-zero"):
    """Assert that any tensor element in sequence is non-zero"""
    if not (tensor != 0).any():
        raise AssertionError(f"{message}: no tensor elements are non-zero")


def assert_tensor_all_elements_greater_than_in_sequence(tensor, value, message="Not all tensor elements in sequence are greater than the specified value"):
    """Assert that all tensor elements in sequence are greater than the specified value"""
    if not (tensor > value).all():
        raise AssertionError(f"{message}: not all tensor elements are greater than {value}")


def assert_tensor_any_elements_greater_than_in_sequence(tensor, value, message="No tensor elements in sequence are greater than the specified value"):
    """Assert that any tensor element in sequence is greater than the specified value"""
    if not (tensor > value).any():
        raise AssertionError(f"{message}: no tensor elements are greater than {value}")


def assert_tensor_all_elements_less_than_in_sequence(tensor, value, message="Not all tensor elements in sequence are less than the specified value"):
    """Assert that all tensor elements in sequence are less than the specified value"""
    if not (tensor < value).all():
        raise AssertionError(f"{message}: not all tensor elements are less than {value}")


def assert_tensor_any_elements_less_than_in_sequence(tensor, value, message="No tensor elements in sequence are less than the specified value"):
    """Assert that any tensor element in sequence is less than the specified value"""
    if not (tensor < value).any():
        raise AssertionError(f"{message}: no tensor elements are less than {value}")


def assert_tensor_all_elements_greater_or_equal_in_sequence(tensor, value, message="Not all tensor elements in sequence are greater than or equal to the specified value"):
    """Assert that all tensor elements in sequence are greater than or equal to the specified value"""
    if not (tensor >= value).all():
        raise AssertionError(f"{message}: not all tensor elements are greater than or equal to {value}")


def assert_tensor_any_elements_greater_or_equal_in_sequence(tensor, value, message="No tensor elements in sequence are greater than or equal to the specified value"):
    """Assert that any tensor element in sequence is greater than or equal to the specified value"""
    if not (tensor >= value).any():
        raise AssertionError(f"{message}: no tensor elements are greater than or equal to {value}")


def assert_tensor_all_elements_less_or_equal_in_sequence(tensor, value, message="Not all tensor elements in sequence are less than or equal to the specified value"):
    """Assert that all tensor elements in sequence are less than or equal to the specified value"""
    if not (tensor <= value).all():
        raise AssertionError(f"{message}: not all tensor elements are less than or equal to {value}")


def assert_tensor_any_elements_less_or_equal_in_sequence(tensor, value, message="No tensor elements in sequence are less than or equal to the specified value"):
    """Assert that any tensor element in sequence is less than or equal to the specified value"""
    if not (tensor <= value).any():
        raise AssertionError(f"{message}: no tensor elements are less than or equal to {value}")


def assert_tensor_all_elements_between_in_sequence(tensor, lower_bound, upper_bound, message="Not all tensor elements in sequence are between the specified bounds"):
    """Assert that all tensor elements in sequence are between the specified bounds"""
    if not ((tensor >= lower_bound) & (tensor <= upper_bound)).all():
        raise AssertionError(f"{message}: not all tensor elements are between [{lower_bound}, {upper_bound}]")


def assert_tensor_any_elements_between_in_sequence(tensor, lower_bound, upper_bound, message="No tensor elements in sequence are between the specified bounds"):
    """Assert that any tensor element in sequence is between the specified bounds"""
    if not ((tensor >= lower_bound) & (tensor <= upper_bound)).any():
        raise AssertionError(f"{message}: no tensor elements are between [{lower_bound}, {upper_bound}]")


def assert_tensor_all_elements_not_between_in_sequence(tensor, lower_bound, upper_bound, message="Not all tensor elements in sequence are outside the specified bounds"):
    """Assert that all tensor elements in sequence are outside the specified bounds"""
    if not ((tensor < lower_bound) | (tensor > upper_bound)).all():
        raise AssertionError(f"{message}: not all tensor elements are outside [{lower_bound}, {upper_bound}]")


def assert_tensor_any_elements_not_between_in_sequence(tensor, lower_bound, upper_bound, message="No tensor elements in sequence are outside the specified bounds"):
    """Assert that any tensor element in sequence is outside the specified bounds"""
    if not ((tensor < lower_bound) | (tensor > upper_bound)).any():
        raise AssertionError(f"{message}: no tensor elements are outside [{lower_bound}, {upper_bound}]")


def assert_tensor_all_elements_in_set_in_sequence(tensor, value_set, message="Not all tensor elements in sequence are in the specified set"):
    """Assert that all tensor elements in sequence are in the specified set"""
    flattened = tensor.flatten()
    for element in flattened:
        if element.item() not in value_set:
            raise AssertionError(f"{message}: element {element.item()} not in set {value_set}")


def assert_tensor_any_elements_in_set_in_sequence(tensor, value_set, message="No tensor elements in sequence are in the specified set"):
    """Assert that any tensor element in sequence is in the specified set"""
    flattened = tensor.flatten()
    for element in flattened:
        if element.item() in value_set:
            return  # Found one in the set
    raise AssertionError(f"{message}: no tensor elements are in set {value_set}")


def assert_tensor_all_elements_not_in_set_in_sequence(tensor, value_set, message="Not all tensor elements in sequence are outside the specified set"):
    """Assert that all tensor elements in sequence are outside the specified set"""
    flattened = tensor.flatten()
    for element in flattened:
        if element.item() in value_set:
            raise AssertionError(f"{message}: element {element.item()} is in set {value_set} but should not be")


def assert_tensor_any_elements_not_in_set_in_sequence(tensor, value_set, message="No tensor elements in sequence are outside the specified set"):
    """Assert that any tensor element in sequence is outside the specified set"""
    flattened = tensor.flatten()
    for element in flattened:
        if element.item() not in value_set:
            return  # Found one outside the set
    raise AssertionError(f"{message}: all tensor elements are in set {value_set}, none are outside")


def assert_tensor_all_elements_close_to_value_in_sequence(tensor, value, rtol=1e-05, atol=1e-08, message="Not all tensor elements in sequence are close to the specified value"):
    """Assert that all tensor elements in sequence are close to the specified value"""
    target_tensor = torch.full_like(tensor, value)
    close_mask = torch.abs(tensor - target_tensor) <= (atol + rtol * torch.abs(target_tensor))
    if not close_mask.all():
        raise AssertionError(f"{message}: not all tensor elements are close to {value}")


def assert_tensor_any_elements_close_to_value_in_sequence(tensor, value, rtol=1e-05, atol=1e-08, message="No tensor elements in sequence are close to the specified value"):
    """Assert that any tensor element in sequence is close to the specified value"""
    target_tensor = torch.full_like(tensor, value)
    close_mask = torch.abs(tensor - target_tensor) <= (atol + rtol * torch.abs(target_tensor))
    if not close_mask.any():
        raise AssertionError(f"{message}: no tensor elements are close to {value}")


def assert_tensor_all_elements_match_regex(tensor, pattern, message="Not all tensor elements match the specified regex pattern"):
    """Assert that all tensor elements match the specified regex pattern"""
    import re
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if not re.match(pattern, element_str):
            raise AssertionError(f"{message}: element {element.item()} does not match pattern {pattern}")


def assert_tensor_any_elements_match_regex(tensor, pattern, message="No tensor elements match the specified regex pattern"):
    """Assert that any tensor element matches the specified regex pattern"""
    import re
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if re.match(pattern, element_str):
            return  # Found one that matches
    raise AssertionError(f"{message}: no tensor elements match pattern {pattern}")


def assert_tensor_all_elements_not_match_regex(tensor, pattern, message="All tensor elements match the regex pattern but should not"):
    """Assert that all tensor elements do not match the specified regex pattern"""
    import re
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if re.match(pattern, element_str):
            raise AssertionError(f"{message}: element {element.item()} matches pattern {pattern} but should not")


def assert_tensor_any_elements_not_match_regex(tensor, pattern, message="No tensor elements fail to match the specified regex pattern"):
    """Assert that any tensor element does not match the specified regex pattern"""
    import re
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if not re.match(pattern, element_str):
            return  # Found one that doesn't match
    raise AssertionError(f"{message}: all tensor elements match pattern {pattern}, none fail to match")


def assert_tensor_all_elements_contain_substring(tensor, substring, message="Not all tensor elements contain the specified substring"):
    """Assert that all tensor elements contain the specified substring"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if substring not in element_str:
            raise AssertionError(f"{message}: element {element.item()} does not contain substring '{substring}'")


def assert_tensor_any_elements_contain_substring(tensor, substring, message="No tensor elements contain the specified substring"):
    """Assert that any tensor element contains the specified substring"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if substring in element_str:
            return  # Found one that contains substring
    raise AssertionError(f"{message}: no tensor elements contain substring '{substring}'")


def assert_tensor_all_elements_not_contain_substring(tensor, substring, message="All tensor elements contain the substring but should not"):
    """Assert that all tensor elements do not contain the specified substring"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if substring in element_str:
            raise AssertionError(f"{message}: element {element.item()} contains substring '{substring}' but should not")


def assert_tensor_any_elements_not_contain_substring(tensor, substring, message="No tensor elements fail to contain the specified substring"):
    """Assert that any tensor element does not contain the specified substring"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if substring not in element_str:
            return  # Found one that doesn't contain substring
    raise AssertionError(f"{message}: all tensor elements contain substring '{substring}', none fail to contain it")


def assert_tensor_all_elements_start_with(tensor, prefix, message="Not all tensor elements start with the specified prefix"):
    """Assert that all tensor elements start with the specified prefix"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if not element_str.startswith(prefix):
            raise AssertionError(f"{message}: element {element.item()} does not start with prefix '{prefix}'")


def assert_tensor_any_elements_start_with(tensor, prefix, message="No tensor elements start with the specified prefix"):
    """Assert that any tensor element starts with the specified prefix"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if element_str.startswith(prefix):
            return  # Found one that starts with prefix
    raise AssertionError(f"{message}: no tensor elements start with prefix '{prefix}'")


def assert_tensor_all_elements_end_with(tensor, suffix, message="Not all tensor elements end with the specified suffix"):
    """Assert that all tensor elements end with the specified suffix"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if not element_str.endswith(suffix):
            raise AssertionError(f"{message}: element {element.item()} does not end with suffix '{suffix}'")


def assert_tensor_any_elements_end_with(tensor, suffix, message="No tensor elements end with the specified suffix"):
    """Assert that any tensor element ends with the specified suffix"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if element_str.endswith(suffix):
            return  # Found one that ends with suffix
    raise AssertionError(f"{message}: no tensor elements end with suffix '{suffix}'")


def assert_tensor_all_elements_have_length(tensor, expected_length, message="Not all tensor elements have the expected length"):
    """Assert that all tensor elements have the expected length when converted to string"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if len(element_str) != expected_length:
            raise AssertionError(f"{message}: element {element.item()} has length {len(element_str)} but expected {expected_length}")


def assert_tensor_any_elements_have_length(tensor, expected_length, message="No tensor elements have the expected length"):
    """Assert that any tensor element has the expected length when converted to string"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if len(element_str) == expected_length:
            return  # Found one with expected length
    raise AssertionError(f"{message}: no tensor elements have expected length {expected_length}")


def assert_tensor_all_elements_not_have_length(tensor, expected_length, message="All tensor elements have the length but should not"):
    """Assert that all tensor elements do not have the specified length when converted to string"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if len(element_str) == expected_length:
            raise AssertionError(f"{message}: element {element.item()} has length {len(element_str)} but should not")


def assert_tensor_any_elements_not_have_length(tensor, expected_length, message="No tensor elements fail to have the specified length"):
    """Assert that any tensor element does not have the specified length when converted to string"""
    flattened = tensor.flatten()
    for element in flattened:
        element_str = str(element.item())
        if len(element_str) != expected_length:
            return  # Found one that doesn't have expected length
    raise AssertionError(f"{message}: all tensor elements have expected length {expected_length}, none fail to have it")


def assert_tensor_all_elements_type(tensor, expected_type, message="Not all tensor elements are of the expected type"):
    """Assert that all tensor elements are of the expected type when converted to Python values"""
    flattened = tensor.flatten()
    for element in flattened:
        element_val = element.item()
        if not isinstance(element_val, expected_type):
            raise AssertionError(f"{message}: element {element_val} is of type {type(element_val)} but expected {expected_type}")


def assert_tensor_any_elements_type(tensor, expected_type, message="No tensor elements are of the expected type"):
    """Assert that any tensor element is of the expected type when converted to Python values"""
    flattened = tensor.flatten()
    for element in flattened:
        element_val = element.item()
        if isinstance(element_val, expected_type):
            return  # Found one with expected type
    raise AssertionError(f"{message}: no tensor elements are of expected type {expected_type}")


def assert_tensor_all_elements_not_type(tensor, expected_type, message="All tensor elements are of the type but should not be"):
    """Assert that all tensor elements are not of the specified type when converted to Python values"""
    flattened = tensor.flatten()
    for element in flattened:
        element_val = element.item()
        if isinstance(element_val, expected_type):
            raise AssertionError(f"{message}: element {element_val} is of type {type(element_val)} but should not be")


def assert_tensor_any_elements_not_type(tensor, expected_type, message="No tensor elements fail to be of the specified type"):
    """Assert that any tensor element is not of the specified type when converted to Python values"""
    flattened = tensor.flatten()
    for element in flattened:
        element_val = element.item()
        if not isinstance(element_val, expected_type):
            return  # Found one that isn't of expected type
    raise AssertionError(f"{message}: all tensor elements are of expected type {expected_type}, none fail to be of it")


def assert_tensor_all_elements_in_range_type(tensor, min_val, max_val, expected_type, message="Not all tensor elements are in range and of expected type"):
    """Assert that all tensor elements are in range and of the expected type"""
    flattened = tensor.flatten()
    for element in flattened:
        element_val = element.item()
        if not (min_val <= element_val <= max_val) or not isinstance(element_val, expected_type):
            raise AssertionError(f"{message}: element {element_val} is not in range [{min_val}, {max_val}] or not of type {expected_type}")


def assert_tensor_any_elements_in_range_type(tensor, min_val, max_val, expected_type, message="No tensor elements are in range and of expected type"):
    """Assert that any tensor element is in range and of the expected type"""
    flattened = tensor.flatten()
    for element in flattened:
        element_val = element.item()
        if min_val <= element_val <= max_val and isinstance(element_val, expected_type):
            return  # Found one in range and of expected type
    raise AssertionError(f"{message}: no tensor elements are in range [{min_val}, {max_val}] and of type {expected_type}")


def assert_tensor_all_elements_not_in_range_type(tensor, min_val, max_val, expected_type, message="All tensor elements are in range and of expected type but should not be"):
    """Assert that all tensor elements are not in range or not of the expected type"""
    flattened = tensor.flatten()
    for element in flattened:
        element_val = element.item()
        if min_val <= element_val <= max_val and isinstance(element_val, expected_type):
            raise AssertionError(f"{message}: element {element_val} is in range [{min_val}, {max_val}] and of type {expected_type} but should not be")


def assert_tensor_any_elements_not_in_range_type(tensor, min_val, max_val, expected_type, message="No tensor elements fail to be in range or not of expected type"):
    """Assert that any tensor element is not in range or not of the expected type"""
    flattened = tensor.flatten()
    for element in flattened:
        element_val = element.item()
        if not (min_val <= element_val <= max_val and isinstance(element_val, expected_type)):
            return  # Found one that fails the condition
    raise AssertionError(f"{message}: all tensor elements are in range [{min_val}, {max_val}] and of type {expected_type}, none fail")