# Inference-PIO Test Report

Generated on: 2026-01-25 21:01:29

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | 11 |
| Passed | 10 |
| Failed | 1 |
| Skipped | 0 |
| Pass Rate | 90.91% |
| Execution Time | 0.41s |

## System Information

- OS: nt (win32)
- Python Version: 3.12.10
- Torch Version: 2.7.1+cu128
- CUDA Available: Yes
- CPU Count: 8
- Memory Total: 10.30 GB

## Performance Metrics

- Average Duration: 0.04s
- Min Duration: 0.00s
- Max Duration: 0.20s
- CPU Usage: 14.4%
- Memory Usage: 50.9%

## Test Results

| Test Name | Status | Duration (s) | Error Message |
|-----------|--------|--------------|---------------|
| tests/sample_reporting_test.py::test_fast_operation | PASS | 0.01 |  |
| tests/sample_reporting_test.py::test_medium_operation | PASS | 0.05 |  |
| tests/sample_reporting_test.py::test_slow_operation | PASS | 0.20 |  |
| tests/sample_reporting_test.py::test_failing_operation | FAIL | 0.02 | def test_failing_operation():
        """A test th... |
| tests/sample_reporting_test.py::test_unit_feature | PASS | 0.03 |  |
| tests/sample_reporting_test.py::test_integration_component | PASS | 0.04 |  |
| tests/sample_reporting_test.py::test_parametrized[1-2] | PASS | 0.01 |  |
| tests/sample_reporting_test.py::test_parametrized[2-4] | PASS | 0.01 |  |
| tests/sample_reporting_test.py::test_parametrized[3-6] | PASS | 0.01 |  |
| tests/sample_reporting_test.py::test_with_exception | PASS | 0.02 |  |
| tests/sample_reporting_test.py::test_performance_metric | PASS | 0.00 |  |

## Failed Tests Details

### tests/sample_reporting_test.py::test_failing_operation

**Error:** def test_failing_operation():
        """A test that will fail to demonstrate error reporting."""
        time.sleep(0.02)
>       assert 2 + 2 == 5, "This is intentionally wrong to show error reporting"
E       AssertionError: This is intentionally wrong to show error reporting
E       assert (2 + 2) == 5

tests\sample_reporting_test.py:37: AssertionError

**Traceback:**
```
None
```

