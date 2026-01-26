import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Step 1: Testing import of centralized utilities...")

try:
    from tests.test_utilities import assert_equal
    print("SUCCESS: Import of assert_equal from test_utilities succeeded")
    
    # Actually use the function
    assert_equal(1, 1)
    print("SUCCESS: assert_equal function works correctly")
    
except Exception as e:
    print(f"ERROR importing from test_utilities: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 2: Testing import from old location (compatibility)...")

try:
    from tests.core.test_utils import assert_equal as old_assert_equal
    print("SUCCESS: Import from old test_utils location succeeded")
    
    # Actually use the function
    old_assert_equal(1, 1)
    print("SUCCESS: Old location function works correctly")
    
except Exception as e:
    print(f"ERROR importing from old location: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 3: Testing other centralized functions...")

try:
    from tests.test_utilities import TestRunner
    runner = TestRunner()
    print("SUCCESS: TestRunner class instantiated successfully")
    
    from tests.test_utilities import BaseQwen3VLTestCase
    base_test = BaseQwen3VLTestCase()
    print("SUCCESS: BaseQwen3VLTestCase class instantiated successfully")
    
except Exception as e:
    print(f"ERROR with other functions: {e}")
    import traceback
    traceback.print_exc()

print("\nVerification completed!")