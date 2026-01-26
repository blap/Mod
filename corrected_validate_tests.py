import os
import sys
import subprocess
import tempfile
from pathlib import Path

def validate_test_file(file_path):
    """Validate a single test file by importing it."""
    try:
        # Create a temporary Python script that imports the test file
        temp_script = f'''
import sys
sys.path.insert(0, r'{os.path.dirname(file_path)}')
sys.path.insert(0, r'{str(Path(file_path).parent.parent.parent.absolute())}')

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("{Path(file_path).stem}", r"{file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {{e}}")
'''

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(temp_script)
            temp_file = f.name

        # Run the temporary script
        result = subprocess.run([sys.executable, temp_file],
                              capture_output=True, text=True, timeout=30)

        # Clean up
        os.unlink(temp_file)

        output = result.stdout.strip()
        if output == "SUCCESS":
            return True, ""
        else:
            error_msg = output.replace("ERROR: ", "")
            return False, error_msg

    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)

def validate_all_tests():
    """Validate all test files in the project."""
    # Find all test files
    test_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.startswith('test_') or '_test' in file:
                if file.endswith('.py'):
                    test_files.append(os.path.join(root, file))

    print(f"Validating {len(test_files)} test files...")

    results = []
    for i, file_path in enumerate(test_files):
        print(f"[{i+1}/{len(test_files)}] Validating: {file_path}")
        success, error = validate_test_file(file_path)
        results.append((file_path, success, error))

        if not success:
            print(f"  FAILED: {error}")
        else:
            print(f"  OK")

    # Summary
    successful = sum(1 for _, success, _ in results if success)
    failed = len(results) - successful

    print(f"\\nValidation Summary:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(results)}")

    if failed > 0:
        print(f"\\nFailed files:")
        for file_path, success, error in results:
            if not success:
                print(f"  - {file_path}: {error}")

    return results

if __name__ == "__main__":
    validate_all_tests()