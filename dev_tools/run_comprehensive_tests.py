"""
Test Runner for All Model Optimizations

This script runs comprehensive tests for all 4 models (GLM-4-7, Qwen3-4b-instruct-2507, 
Qwen3-coder-30b, Qwen3-vl-2b) covering all the new optimizations implemented.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference_pio.common.comprehensive_optimization_tests import (
    TestStructuredPruning,
    TestAdaptiveSparseAttention,
    TestAdaptiveBatching,
    TestContinuousNAS,
    TestStreamingComputation,
    TestTensorDecomposition,
    TestSparseNeuralNetworks,
    TestFeedbackController,
    TestModularOptimizations,
    TestModelSurgery,
    TestIntegration
)

from src.inference_pio.tests.basic_tests import (
    TestGLM47Plugin,
    TestQwen3Coder30BPlugin,
    TestQwen3VL2BPlugin,
    TestQwen34BInstruct2507Plugin
)


def create_test_suite():
    """Create a comprehensive test suite for all optimizations."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add basic plugin tests
    suite.addTests(loader.loadTestsFromTestCase(TestGLM47Plugin))
    suite.addTests(loader.loadTestsFromTestCase(TestQwen3Coder30BPlugin))
    suite.addTests(loader.loadTestsFromTestCase(TestQwen3VL2BPlugin))
    suite.addTests(loader.loadTestsFromTestCase(TestQwen34BInstruct2507Plugin))
    
    # Add optimization-specific tests
    suite.addTests(loader.loadTestsFromTestCase(TestStructuredPruning))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveSparseAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveBatching))
    suite.addTests(loader.loadTestsFromTestCase(TestContinuousNAS))
    suite.addTests(loader.loadTestsFromTestCase(TestStreamingComputation))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorDecomposition))
    suite.addTests(loader.loadTestsFromTestCase(TestSparseNeuralNetworks))
    suite.addTests(loader.loadTestsFromTestCase(TestFeedbackController))
    suite.addTests(loader.loadTestsFromTestCase(TestModularOptimizations))
    suite.addTests(loader.loadTestsFromTestCase(TestModelSurgery))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    return suite


def run_tests():
    """Run all tests and report results."""
    print("=" * 80)
    print("COMPREHENSIVE OPTIMIZATION TEST SUITE FOR ALL MODELS")
    print("=" * 80)
    print()
    
    # Create test suite
    suite = create_test_suite()
    
    # Create test runner
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        buffer=True
    )
    
    # Run tests
    result = runner.run(suite)
    
    print()
    print("=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"\n{test}")
            print(trace)
    
    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"\n{test}")
            print(trace)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)